import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse
import torch.sparse
import vipy
from collections import OrderedDict
from vipy.util import try_import
import keynet.torch
import keynet.sparse
from keynet.sparse import sparse_permutation_matrix, sparse_identity_matrix, sparse_identity_matrix_like, diagonal_affine_to_linear
from keynet.sparse import sparse_uniform_random_diagonal_matrix, sparse_gaussian_random_diagonal_matrix, sparse_random_diagonally_dominant_doubly_stochastic_matrix
from keynet.sparse import sparse_channelorder_to_blockorder_matrix, sparse_affine_to_linear, sparse_block_diagonal, sparse_orthogonal_block_diagonal
from keynet.sparse import sparse_orthogonal_matrix
from keynet.blockpermute import hierarchical_block_permutation_matrix
import keynet.layer
import keynet.fiberbundle
from keynet.util import blockview
from keynet.globals import verbose


class KeyedModel(object):
    def __init__(self, net, inshape, inkey, f_layername_to_keypair, f_module_to_keyedmodule=None, do_output_encryption=False):
        # Assign layerkeys using provided lambda function
        net.eval()
        netshape = keynet.torch.netshape(net, inshape)
        (i,o) = (netshape['input'], netshape['output'])              
        layerkey = {k:{'outkeypair':f_layername_to_keypair(k, v['outshape']),
                       'prevlayer':v['prevlayer']}
                    for (k,v) in netshape.items() if k not in set(['input', 'output'])}

        leafkey = {'input':inkey, 'output':layerkey[o]['outkeypair'][1] if do_output_encryption else None}  # private
        layerkey = {k:{'A':v['outkeypair'][0] if k!=o or (k==o and do_output_encryption) else None,
                       'Ainv':inkey if v['prevlayer'] == 'input' else layerkey[v['prevlayer']]['outkeypair'][1]}
                    for (k,v) in layerkey.items()}
        layerkey.update(leafkey)

        # Iterate over named layers and replace with keyed versions
        layernames = set([k for (k,m) in net.named_children()])        
        d_name_to_keyedmodule = OrderedDict()        
        for (k,m) in net.named_children():
            if verbose():
                print('[keynet.layers.KeyNet]: Keying "%s"' % k)
            assert k in layerkey, 'Key not found for layer "%s"' % k
            assert k in netshape, 'Layer name not found in net shape for layer "%s"' % k
            assert 'A' in layerkey[k] and 'Ainv' in layerkey[k], 'Keys not specified for layer "%s"' % k
            assert 'inshape' in netshape[k], 'Layer input shape not specified for layer "%s"' % k

            # Replace torch layers with keyed layers            
            if isinstance(m, nn.BatchNorm2d):
                assert ('_' in k) and hasattr(net, k.split('_')[0]), "Batchnorm layers must be named 'mylayername_bn' for corresponding linear layer mylayername.  (e.g. 'conv3_bn')"
                k_prev = k.split('_')[0]
                assert k_prev in d_name_to_keyedmodule, "Batchnorm layer named 'mylayer_bn' must come after 'mylayer' (e.g. 'conv3_bn' must come after 'conv3')"
                m_prev = getattr(net, k_prev)
                (bn_weight, bn_bias) = keynet.torch.fuse_conv2d_and_bn(m_prev.weight, m_prev.bias,
                                                                       m.running_mean, m.running_var, 1E-5,
                                                                       m.weight, m.bias)
                # Replace module k_prev with fused weights
                (m_prev.weight, m_prev.bias) = (bn_weight, bn_bias)
                d_name_to_keyedmodule[k_prev] = f_module_to_keyedmodule(m_prev, netshape[k]['inshape'], netshape[k]['outshape'], layerkey[k_prev]['A'], layerkey[k_prev]['Ainv'])
            elif isinstance(m, nn.Dropout):
                pass  # identity matrix at test time, ignore me
            elif '%s_bn' % k not in layernames:
                d_name_to_keyedmodule[k] = f_module_to_keyedmodule(m, netshape[k]['inshape'], netshape[k]['outshape'], layerkey[k]['A'], layerkey[k]['Ainv'])

        self._keynet = nn.Sequential(d_name_to_keyedmodule)
        self._embeddingkey = layerkey['output']
        self._imagekey = layerkey['input']
        self._layernames = layernames
        self._outshape = netshape[netshape['output']]['outshape']

    def __repr__(self):
        return self._keynet.__repr__()
    
    def __getattr__(self, attr):
        try:
            return self._keynet.__getattr__(attr)
        except:
            return self.__getattribute__(attr)
    
    def forward(self, img_cipher, outkey=None):
        outkey = outkey if outkey is not None else self.embeddingkey()
        y_cipher = self._keynet.forward(img_cipher)
        return keynet.torch.linear_to_affine(self.decrypt(y_cipher, outkey) if outkey is not None else y_cipher, self._outshape)
    
    def decrypt(self, y_cipher, outkey=None): 
        outkey = outkey if outkey is not None else self.embeddingkey()
        return keynet.layer.KeyedLayer(W=outkey).forward(y_cipher) if outkey is not None else y_cipher

    def imagekey(self):
        """Return key for decryption of image (if desired)"""
        return self._imagekey

    def embeddingkey(self):
        """Return key for decryption of output embedding layer"""
        return self._embeddingkey
    
    def public(self):
        """When publicly releasing the keynet, remove keys (if present)"""
        self._imagekey = None
        self._embeddingkey = None        
        return self
        
    def num_parameters(self):
        return sum([c.nnz() for (k,c) in self._keynet.named_children() if hasattr(c, 'nnz')])

    def layers(self):
        return self._layernames


class KeyedSensor(keynet.layer.KeyedLayer):
    def __init__(self, inshape, keypair):
        assert isinstance(inshape, tuple) and len(inshape) == 3
        (self._encryptkey, self._decryptkey) = keypair
        self._inshape = (1, *inshape)  # 1xCxHxW
        self._tensor = None
        self._im = None
        self.W = keynet.sparse.SparseMatrix(self._encryptkey)
        self._layertype = 'input'
        
    def __repr__(self):
        return str('<KeySensor: height=%d, width=%d, channels=%d>' % (self._inshape[2], self._inshape[3], self._inshape[1]))
    
    def load(self, imgfile):
        im = vipy.image.Image(imgfile).resize(height=self._inshape[2], width=self._inshape[3])
        if self._inshape[1] == 1:
            im = im.grey()
        self._tensor = im.float().torch().contiguous()  # HxWxC -> 1xCxHxW
        self._im = im
        return self

    def fromimage(self, im):
        assert (1, im.channels(), im.height(), im.width()) == self._inshape        
        self._im = im
        self._tensor = im.float().torch().contiguous()
        return self
    
    def fromtensor(self, x):
        if x is not None:
            self._tensor = x.clone().type(torch.FloatTensor)
        return self

    def astensor(self):
        return self._tensor
    
    def asimage(self):
        x_torch = self._tensor        
        if self.isencrypted():
            x_torch = keynet.torch.linear_to_affine(x_torch, self._inshape)
            print(x_torch.shape)            
        im = self._im.fromtorch(x_torch).mat2gray()  # 1xCxHxW -> HxWxC
        return im.rgb() if im.iscolor() else im.lum()  # uint8

    def keypair(self):
        return (self._encryptkey, self._decryptkey)

    def key(self):
        return self._decryptkey
    
    def isencrypted(self):
        """An encrypted image is converted from NxCxHxW tensor to Nx(C*H*W+1)"""
        return self.isloaded() and self._tensor.ndim == 2 and self._tensor.shape == (1, np.prod(self._inshape)+1)

    def isloaded(self):
        return self._tensor is not None
    
    def encrypt(self, x_raw=None):
        """img_tensor is NxCxHxW, return Nx(C*H*W+1) homogenized and encrypted"""
        assert self.isloaded() or x_raw is not None, "Load image first"
        self.fromtensor(x_raw)
        if not self.isencrypted():
            self._tensor = self.forward(keynet.torch.affine_to_linear(self._tensor))
        return self
        
    def decrypt(self, x_cipher=None):
        """x_cipher is Nx(C*H*W+1) homogenized, convert to NxCxHxW decrypted"""
        assert self.isloaded() or x_cipher is not None, "Load image first"        
        self.fromtensor(x_cipher)        
        if self.isencrypted():
            x_raw = super(KeyedSensor, self).decrypt(self._decryptkey, self._tensor)
            self._tensor = keynet.torch.linear_to_affine(x_raw, self._inshape)
        return self

    
class OpticalFiberBundle(KeyedSensor):
    def __init__(self, inshape=(3,512,512), keypair=None):
        (encryptkey, decryptkey) = keygen(inshape, global_photometric='identity', local_photometric='identity', global_geometric='identity', local_geometric='identity')
        super(OpticalFiberBundle, self).__init__(inshape, (encryptkey, decryptkey))
    
    def load(self, imgfile):
        (N,C,H,W) = self._inshape
        img_color = vipy.image.Image(imgfile).maxdim(max(H,W)).centercrop(height=H, width=W).numpy()
        img_sim = keynet.fiberbundle.simulation(img_color, h_xtalk=0.05, v_xtalk=0.05, fiber_core_x=16, fiber_core_y=16, do_camera_noise=True)        
        self._im = vipy.image.Image(array=np.uint8(img_sim), colorspace='rgb')
        return self
    
    def image(self):
        return self._im
    

def layergen(module, inshape, outshape, A, Ainv, tileshape=None, backend='scipy'):
    if backend == 'scipy':
        return keynet.layer.KeyedLayer(module, inshape, outshape, A, Ainv, tileshape=tileshape)
    else:
        raise ValueError('invalid backend "%s"' % backend)


def keygen(shape, global_geometric, local_geometric, global_photometric, local_photometric, memoryorder='channel', alpha=None, beta=None, gamma=None, seed=None, hierarchical_blockshape=None, hierarchical_permute_at_level=None, blocksize=None):
    allowable_memoryorder = set(['channel', 'block'])
    allowable_global_geometric = set(['identity', 'permutation', 'hierarchical_permutation', 'hierarchical_rotation', 'givens_orthogonal'])    
    allowable_local_geometric = set(['identity', 'permutation', 'doubly_stochastic', 'givens_orthogonal'])
    allowable_photometric = set(['identity', 'uniform_random_gain', 'uniform_random_affine', 'uniform_random_bias', 'constant_bias', 'linear_bias'])

    (channels, height, width) = shape
    blocknumel = blocksize*blocksize if blocksize is not None else None
    N = np.prod(shape)
    
    if seed is not None:
        np.random.seed(seed)    
    
    if memoryorder == 'channel':
        (c, cinv) = (sparse_identity_matrix(N), sparse_identity_matrix(N))
    elif memoryorder == 'block':
        assert blocksize is not None
        (c, cinv) = sparse_channelorder_to_blockorder_matrix(shape, blocksize, withinverse=True)
    else:
        raise ValueError("Invalid memory order '%s' - must be in '%s'" % (memoryorder, str(allowable_memoryorder)))
    (C, Cinv) = (sparse_affine_to_linear(c), sparse_affine_to_linear(cinv))
    
    if global_geometric == 'identity':
        (G, Ginv) = (sparse_identity_matrix(N), sparse_identity_matrix(N))
    elif global_geometric == 'permutation':
        (G, Ginv) = sparse_permutation_matrix(N, withinverse=True)
    elif global_geometric == 'hierarchical_permutation':
        assert hierarchical_blockshape is not None and hierarchical_permute_at_level is not None
        (A, Ainv) = keynet.sparse.sparse_channelorder_to_pixelorder_matrix((channels, height, width), withinverse=True)
        (G, Ginv) = hierarchical_block_permutation_matrix((height, width, channels), hierarchical_blockshape, hierarchical_permute_at_level, min_blocksize=8, seed=seed, twist=False, withinverse=True)
        (G, Ginv) = (Ainv.dot(G).dot(A), Ainv.dot(Ginv).dot(A))  # CxHxW -> HxWxC -> hierarchical permute in HxWxC order -> CxHxW
        if memoryorder != 'channel':
            (G, Ginv) = (c.dot(G).dot(cinv), c.dot(Ginv).dot(cinv))        
    elif global_geometric == 'hierarchical_rotation':
        assert hierarchical_blockshape is not None and hierarchical_permute_at_level is not None
        (A, Ainv) = keynet.sparse.sparse_channelorder_to_pixelorder_matrix((channels, height, width), withinverse=True)                
        (G, Ginv) = hierarchical_block_permutation_matrix((height, width, channels), hierarchical_blockshape, hierarchical_permute_at_level, min_blocksize=8, seed=seed, twist=True, withinverse=True)
        (G, Ginv) = (Ainv.dot(G).dot(A), Ainv.dot(Ginv).dot(A))  # CxHxW -> HxWxC -> hierarchical permute in HxWxC order -> CxHxW
        if memoryorder != 'channel':
            (G, Ginv) = (c.dot(G).dot(cinv), c.dot(Ginv).dot(cinv))        
    elif global_geometric == 'givens_orthogonal':
        assert alpha is not None
        (G, Ginv) = sparse_orthogonal_matrix(N, int(alpha), balanced=True, withinverse=True)        
    else:
        raise ValueError("Invalid global geometric transform '%s' - must be in '%s'" % (global_geometric, str(allowable_global_geometric)))
    (G, Ginv) = (sparse_affine_to_linear(G), sparse_affine_to_linear(Ginv))
    
    if local_geometric == 'identity':
        (g, ginv) = (sparse_identity_matrix(N), sparse_identity_matrix(N))        
    elif local_geometric == 'permutation':
        assert blocksize is not None
        g = keynet.sparse.SparseTiledMatrix(tilediag=sparse_permutation_matrix(blocknumel), shape=(N,N)).tocoo().astype(np.float32)
        ginv = g.transpose()
    elif local_geometric == 'doubly_stochastic':
        assert blocksize is not None and blocksize < 4096 and alpha is not None
        (g, ginv) = sparse_random_diagonally_dominant_doubly_stochastic_matrix(blocknumel, int(alpha), withinverse=True)  # expensive inverse
        g = keynet.sparse.SparseTiledMatrix(tilediag=g, shape=(N,N)).tocoo()
        ginv = keynet.sparse.SparseTiledMatrix(tilediag=ginv, shape=(N,N)).tocoo()
    elif local_geometric == 'givens_orthogonal':
        assert alpha is not None and blocksize is not None
        (g, ginv) = sparse_orthogonal_matrix(blocknumel, int(alpha), balanced=True, withinverse=True)
        (A, Ainv) = sparse_permutation_matrix(blocknumel, withinverse=True)
        (g, ginv) = (A.dot(g), ginv.dot(Ainv))
        g = keynet.sparse.SparseTiledMatrix(tilediag=g, shape=(N,N)).tocoo().astype(np.float32)
        ginv = keynet.sparse.SparseTiledMatrix(tilediag=ginv, shape=(N,N)).tocoo().astype(np.float32)
    else:
        raise ValueError("Invalid local geometric transform '%s' - must be in '%s'" % (local_geometric, str(allowable_local_geometric)))        
    (g, ginv) = (sparse_affine_to_linear(g), sparse_affine_to_linear(ginv))
    
    if global_photometric == 'identity':
        (P, Pinv) = (sparse_affine_to_linear(sparse_identity_matrix(N)), sparse_affine_to_linear(sparse_identity_matrix(N)))
    elif global_photometric == 'uniform_random_gain':
        assert beta is not None and beta > 0
        (P, Pinv) = sparse_uniform_random_diagonal_matrix(N, beta, withinverse=True)
        (P, Pinv) = (sparse_affine_to_linear(P), sparse_affine_to_linear(Pinv))
    elif global_photometric == 'uniform_random_bias':
        assert gamma is not None and gamma > 0
        (P, Pinv) = diagonal_affine_to_linear(sparse_identity_matrix(N), gamma*np.random.rand(N,1), withinverse=True)
    elif global_photometric == 'linear_bias':
        assert gamma is not None and gamma > 0
        (P, Pinv) = diagonal_affine_to_linear(sparse_identity_matrix(N), (gamma/float(N))*np.array(range(0,N)).reshape(N,1), withinverse=True)
    elif global_photometric == 'uniform_random_affine':
        assert beta is not None and beta > 0 and gamma is not None and gamma > 0
        P = sparse_uniform_random_diagonal_matrix(N, beta)
        (P, Pinv) = diagonal_affine_to_linear(P, gamma*np.random.rand(N,1), withinverse=True)
    else:
        raise ValueError("Invalid global photometric transform '%s' - must be in '%s'" % (global_photometric, str(allowable_photometric)))                

    if local_photometric == 'identity':
        (p, pinv) = (sparse_affine_to_linear(sparse_identity_matrix(N)), sparse_affine_to_linear(sparse_identity_matrix(N)))
    elif local_photometric == 'uniform_random_gain':
        assert blocksize is not None
        assert beta is not None and beta > 0        
        (p, pinv) = sparse_uniform_random_diagonal_matrix(blocknumel, beta, withinverse=True)
        (p, pinv) = (sparse_block_diagonal(p, shape=(N,N)), sparse_block_diagonal(pinv, shape=(N,N)))
        (p, pinv) = (sparse_affine_to_linear(p), sparse_affine_to_linear(pinv))
    elif local_photometric == 'uniform_random_bias':
        # FIXME: local bias does not respect memoryorder
        assert blocksize is not None 
        assert gamma is not None and gamma > 0
        bias = np.repeat(gamma*np.random.rand(blocknumel, 1), np.ceil(N / blocknumel))[0:N].reshape(N,1)
        (p, pinv) = diagonal_affine_to_linear(sparse_identity_matrix(N), bias=bias, withinverse=True)        
    elif local_photometric == 'uniform_random_affine':
        assert blocksize is not None 
        assert beta is not None and beta > 0 and gamma is not None and gamma > 0
        p = sparse_uniform_random_diagonal_matrix(blocknumel, beta)
        bias = np.repeat(gamma*np.random.rand(blocknumel, 1), np.ceil(N / blocknumel))[0:N].reshape(N,1)        
        (p, pinv) = diagonal_affine_to_linear(sparse_block_diagonal(p, shape=(N,N)), bias=bias, withinverse=True)
    else:
        raise ValueError("Invalid local photometric transform '%s' - must be in '%s'" % (local_photometric, str(allowable_photometric)))                
    
    # Compose!
    A = Cinv.dot(p.dot(g.dot(P.dot(G.dot(C)))))
    Ainv = Cinv.dot(Ginv.dot(Pinv.dot(ginv.dot(pinv.dot(C)))))
    return (A, Ainv)


def Keynet(inshape, net=None, backend='scipy', global_photometric='identity', local_photometric='identity', global_geometric='identity', local_geometric='identity', memoryorder='channel',
           do_output_encryption=False, alpha=None, beta=None, gamma=None, hierarchical_blockshape=None, hierarchical_permute_at_level=None, blocksize=None, tileshape=None):
    
    f_layergen = lambda module, inshape, outshape, A, Ainv: layergen(module, inshape, outshape, A, Ainv, tileshape=tileshape, backend=backend)
    f_keypair = lambda layername, shape:  keygen(shape, 
                                                 global_photometric=global_photometric if 'relu' not in layername or global_photometric == 'identity' else 'identity',
                                                 local_photometric=local_photometric if 'relu' not in layername or local_photometric == 'identity' else 'identity',
                                                 global_geometric=global_geometric if 'relu' not in layername or global_geometric == 'identity' else 'permutation',
                                                 local_geometric=local_geometric if 'relu' not in layername or local_geometric == 'identity' else 'permutation',
                                                 memoryorder=memoryorder,                                                                                                  
                                                 blocksize=blocksize, alpha=alpha, beta=beta, gamma=gamma, hierarchical_blockshape=hierarchical_blockshape, hierarchical_permute_at_level=hierarchical_permute_at_level)
    
    sensor = KeyedSensor(inshape, f_keypair('input', inshape))
    model = KeyedModel(net, inshape, sensor.key(), f_keypair, f_layergen, do_output_encryption=do_output_encryption) if net is not None else None
    return (sensor, model)


def IdentityKeynet(inshape, net, backend='scipy'):
    return Keynet(inshape, net, backend=backend)


def PermutationKeynet(inshape, net, do_output_encryption=False):
    return Keynet(inshape, net, global_geometric='permutation', do_output_encryption=do_output_encryption)


def TiledIdentityKeynet(inshape, net, tilesize):
    return Keynet(inshape, net, backend='scipy-tiled', blocksize=tilesize)


def TiledPermutationKeynet(inshape, net, tilesize):
    return Keynet(inshape, net, global_geometric='permutation', backend='scipy-tiled', blocksize=tilesize)


def OpticalFiberBundleKeynet(inshape, net):
    f_keypair = keygen('identity', 'scipy')  # FIXME
    sensor = OpticalFiberBundle(inshape, f_keypair('input', inshape))
    model = KeyedModel(net, inshape, sensor.key(), f_keypair, do_output_encryption=do_output_encryption, verbose=verbose)
    return (sensor, model)
    
