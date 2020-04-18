import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse
import torch.sparse
import vipy
from collections import OrderedDict
from vipy.util import try_import, tolist
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
import copy 
from vipy.util import Stopwatch
import warnings


class KeyedModel(object):
    def __init__(self, net, inshape, inkey, f_layername_to_keypair, f_module_to_keyedmodule=None, do_output_encryption=False):
        # Assign layerkeys using provided lambda function
        net.eval()
        netshape = keynet.torch.netshape(net, inshape)
        
        # Remove identity layers from keying (doubly linked list - delete node)
        identity_layers = ['dropout']
        for prefix in identity_layers:
            netshape = {k:v for (k,v) in netshape.items() if k not in prefix}  # remove identity layer
            for (k,v) in netshape.items():
                if v['nextlayer'] is not None and prefix in v['nextlayer']:
                    v['nextlayer'] = netshape[v['nextlayer']]['nextlayer']  # bypass next layer
                elif v['prevlayer'] is not None and prefix in v['prevlayer']:
                    v['prevlayer'] = netshape[v['prevlayer']]['prevlayer']  # bypass prev layer

        # Generate keypairs
        (i,o) = (netshape['input']['nextlayer'], netshape['output']['prevlayer'])              
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
                assert '_bn' in k, "Batchnorm layers must be named 'mylayername_bn' for corresponding linear layer mylayername.  (e.g. 'conv3_bn')"
                k_prev = k.split('_')[0]
                assert netshape[k]['prevlayer'] == k_prev, "Batchnorm layer named 'mylayer_bn' must come right after 'mylayer' (e.g. 'conv3_bn' must come right after 'conv3')"

                # Fuse batchnorm weights with previous layer (which was skipped)
                m_prev = copy.deepcopy(getattr(net, k_prev))  # do not overwrite
                (bn_weight, bn_bias) = keynet.torch.fuse_conv2d_and_bn(m_prev.weight, m_prev.bias,
                                                                       m.running_mean, m.running_var, 1E-5,
                                                                       m.weight, m.bias)

                # Replace module k_prev with fused weights, do not include batchnorm in final network
                (m_prev.weight, m_prev.bias) = (torch.nn.Parameter(bn_weight), torch.nn.Parameter(bn_bias))
                B = layerkey[k]['A'].dot(layerkey[k]['Ainv'])  # use batchnorm outkey
                d_name_to_keyedmodule[k_prev] = f_module_to_keyedmodule(m_prev, netshape[k_prev]['inshape'], netshape[k]['outshape'], B.dot(layerkey[k_prev]['A']), layerkey[k_prev]['Ainv'])
                if verbose():
                    print('[keynet.layers.KeyNet]:     %s' % str(d_name_to_keyedmodule[k_prev]))
                    print('[keynet.layers.KeyNet]:     %s' % k)

            elif isinstance(m, nn.ReLU):
                # Apply key to previous layer (which was skipped) and make this layer unkeyed, forward is ReLU only
                k_prev = netshape[k]['prevlayer']
                if '_bn' not in k_prev:
                    m_prev = getattr(net, k_prev)
                    B = layerkey[k]['A'].dot(layerkey[k]['Ainv'])  # use relu outkey
                    d_name_to_keyedmodule[k_prev] = f_module_to_keyedmodule(m_prev, netshape[k_prev]['inshape'], netshape[k_prev]['outshape'], B.dot(layerkey[k_prev]['A']), layerkey[k_prev]['Ainv'])                
                    d_name_to_keyedmodule[k] = copy.deepcopy(m)  # unkeyed, ReLU only
                    if verbose():
                        print('[keynet.layers.KeyNet]:     %s' % str(d_name_to_keyedmodule[k_prev]))
                        print('[keynet.layers.KeyNet]:     %s' % str(d_name_to_keyedmodule[k]))
                else:
                    # If previous layer is batchnorm, then we need to include an explicit ReLU layer, this is expensive
                    warnings.warn('Keying ReLU since previous layer "%s" is already keyed - Avoid sequential batchnorm and ReLU layers for efficient keying' % k_prev)
                    d_name_to_keyedmodule[k] = f_module_to_keyedmodule(m, netshape[k]['inshape'], netshape[k]['outshape'], layerkey[k]['A'], layerkey[k]['Ainv'])                
                    if verbose():
                        print('[keynet.layers.KeyNet]:     %s' % str(d_name_to_keyedmodule[k]))

            elif isinstance(m, nn.Dropout):    
                if verbose():
                    print('[keynet.layers.KeyNet]:     Skipping...')
                pass  # Dropout is identity in final eval() network, so key assignment skips dropout and removes from keyed network

            elif netshape[k]['nextlayer'] is not None and (('%s_bn' % k) == netshape[k]['nextlayer'] or 'relu' in netshape[k]['nextlayer']):                
                pass  # Key this layer by merging with next layer 
            else:
                d_name_to_keyedmodule[k] = f_module_to_keyedmodule(m, netshape[k]['inshape'], netshape[k]['outshape'], layerkey[k]['A'], layerkey[k]['Ainv'])
                if verbose():
                    print('[keynet.layers.KeyNet]:     %s' % str(d_name_to_keyedmodule[k]))

        self._keynet = nn.Sequential(d_name_to_keyedmodule)  # layers in insertion order
        self._embeddingkey = layerkey['output']
        self._imagekey = layerkey['input']
        self._layernames = layernames
        self._outshape = netshape['output']['outshape']

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
        return sum([c.nnz() for (k,c) in self._keynet.named_children() if isinstance(c, keynet.layer.KeyedLayer)])

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

    def save(self, outfile='/tmp/out.png'):
        assert self.isencrypted()
        x_numpy_linear = self._tensor.detach().numpy().transpose()
        x_numpy_affine = x_numpy_linear.flatten()[:-1]
        (A, Ainv) = keynet.sparse.mat2gray(x_numpy_affine)
        x_torch_linear = torch.as_tensor(A.dot(x_numpy_linear)).t()
        x_torch_affine = keynet.torch.linear_to_affine(x_torch_linear, self._inshape)
        vipy.image.Image().fromtorch(x_torch_affine).colorspace('float').rgb().saveas(outfile)
        return (outfile, self._decryptkey.dot(Ainv))
        
    def load(self, imgfile, imagekey=None):
        im = vipy.image.Image(imgfile)        
        if imagekey is not None:
            if self._inshape[1] == 1:
                im = im.red()  # one channel only, no scaling
            assert (1, im.channels(), im.height(), im.width()) == self._inshape            
            x_torch_affine = im.float().torch().contiguous()  # HxWxC -> 1xCxHxW                    
            x_torch_linear = keynet.torch.affine_to_linear((1.0/255.0)*x_torch_affine)  # [0,255] -> [0,1]
            x_numpy_linear = imagekey.dot(x_torch_linear.detach().numpy().transpose())  # DecryptKey*(1/mat2gray)
            x_torch_linear = torch.as_tensor(x_numpy_linear).t()
            self._tensor = keynet.torch.linear_to_affine(x_torch_linear, self._inshape)
            self._im.fromtorch(self._tensor)            
        else:
            im = im.resize(height=self._inshape[2], width=self._inshape[3])
            if self._inshape[1] == 1:
                im = im.grey()            
            self._im = im            
            self._tensor = im.float().torch().contiguous()  # HxWxC -> 1xCxHxW        
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
    def totensor(self):
        return self.astensor()
    
    def asimage(self):
        x_torch = self._tensor        
        if self.isencrypted():
            x_torch = keynet.torch.linear_to_affine(x_torch, self._inshape)
        im = self._im.fromtorch(x_torch).mat2gray()  # 1xCxHxW -> HxWxC
        return im.rgb() if im.iscolor() else im.lum()  # uint8
    def toimage(self):
        return self.asimage()

    def show(self):
        self.asimage().show()
        return self
        
    def keypair(self):
        return (self._encryptkey, self._decryptkey)

    def key(self):
        return self._decryptkey
    
    def isencrypted(self):
        """An encrypted image is converted from NxCxHxW tensor to Nx(C*H*W+1)"""
        return self.isloaded() and self._tensor.ndim == 2 and self._tensor.shape == (1, np.prod(self._inshape)+1)

    def isloaded(self):
        return self._tensor is not None
    
    def encrypt(self):
        """img_tensor is NxCxHxW, return Nx(C*H*W+1) homogenized and encrypted"""
        assert self.isloaded(), "Load image first"
        if not self.isencrypted():
            self._tensor = self.forward(keynet.torch.affine_to_linear(self._tensor))
        return self
        
    def decrypt(self):
        """x_cipher is Nx(C*H*W+1) homogenized, convert to NxCxHxW decrypted"""
        assert self.isloaded(), "Load image first"
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
    if tileshape is not None:
        new_tileshape = (keynet.util.find_closest_positive_divisor(outshape[1], tileshape[0]),  # force non-ragged spatial tileshape
                         keynet.util.find_closest_positive_divisor(inshape[1], tileshape[1])) 
        if verbose() and new_tileshape != tileshape:
            print('[layergen]: Ragged spatial tileshape=%s, forcing non-ragged tileshape "%s" for inshape="%s", outshape="%s"' % (str(tileshape), str(new_tileshape), str(inshape), str(outshape)))
        tileshape = new_tileshape

    if backend == 'scipy':
        return keynet.layer.KeyedLayer(module, inshape, outshape, A, Ainv, tileshape=tileshape)
    else:
        raise ValueError('invalid backend "%s"' % backend)


def keygen(shape, global_geometric, local_geometric, global_photometric, local_photometric, memoryorder='channel', alpha=None, beta=None, gamma=None, seed=None, hierarchical_blockshape=None, hierarchical_permute_at_level=None, blocksize=None, tileshape=None, strict=False):
    allowable_memoryorder = set(['channel', 'block'])
    allowable_global_geometric = set(['identity', 'permutation', 'hierarchical_permutation', 'hierarchical_rotation', 'givens_orthogonal'])    
    allowable_local_geometric = set(['identity', 'permutation', 'doubly_stochastic', 'givens_orthogonal'])
    allowable_photometric = set(['identity', 'uniform_random_gain', 'uniform_random_affine', 'uniform_random_bias', 'constant_bias', 'linear_bias', 'blockwise_constant_bias'])

    (channels, height, width) = shape
    N = np.prod(shape)
    
    if seed is not None:
        np.random.seed(seed)    

    if blocksize is not None:
        if tileshape is not None:
            assert blocksize == tileshape[0] and blocksize == tileshape[1]
        if height == 1 and width == 1:
            blocksize = np.prod(shape)
            H = N  # global transformation
            blocknumel = N  # global transformation
        elif not strict and (height % blocksize != 0 or width % blocksize != 0):
            assert height == width, "Image must be square to correct ragged blocksize"
            new_blocksize = keynet.util.find_closest_positive_divisor(height, blocksize)  # force evenly divisible, if possible
            if verbose():
                print('[keynet.system]: Ragged blocksize %d for image shape %s, setting blocksize=%d' % (blocksize, str((height, width)), new_blocksize))
            blocksize = new_blocksize
            H = height*width  # channel repeated transformation
            blocknumel = blocksize*blocksize  # spatially repeated transformation
        else:
            H = height*width  # channel repeated transformation
            blocknumel = blocksize*blocksize  # spatially repeated transformation

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
        assert tileshape is None, "Global permutation is not tile compressible"
        (G, Ginv) = sparse_permutation_matrix(N, withinverse=True)
    elif global_geometric == 'hierarchical_permutation':
        assert hierarchical_blockshape is not None and hierarchical_permute_at_level is not None
        hierarchical_permute_at_level = tolist(hierarchical_permute_at_level) 
        hierarchical_permute_at_level = hierarchical_permute_at_level if max(height,width)/np.power(2, max(hierarchical_permute_at_level)) >= 8 else []
        hierarchical_permute_at_level = [] if (height==1 and width==1) else hierarchical_permute_at_level
        (A, Ainv) = keynet.sparse.sparse_channelorder_to_pixelorder_matrix((channels, height, width), withinverse=True) 
        (G, Ginv) = hierarchical_block_permutation_matrix((height, width, channels), hierarchical_blockshape, hierarchical_permute_at_level, min_blocksize=8, seed=seed, twist=False, withinverse=True, strict=False)
        (G, Ginv) = (Ainv.dot(G).dot(A), Ainv.dot(Ginv).dot(A))  # CxHxW -> HxWxC -> hierarchical permute in HxWxC order -> CxHxW
        if memoryorder != 'channel':
            (G, Ginv) = (c.dot(G).dot(cinv), c.dot(Ginv).dot(cinv))        
    elif global_geometric == 'hierarchical_rotation':
        assert hierarchical_blockshape is not None and hierarchical_permute_at_level is not None
        hierarchical_permute_at_level = tolist(hierarchical_permute_at_level) 
        hierarchical_permute_at_level = hierarchical_permute_at_level if max(height,width)/np.power(2, max(hierarchical_permute_at_level)) >= 8 else []
        hierarchical_permute_at_level = [] if (height==1 and width==1) else hierarchical_permute_at_level
        (A, Ainv) = keynet.sparse.sparse_channelorder_to_pixelorder_matrix((channels, height, width), withinverse=True)                
        (G, Ginv) = hierarchical_block_permutation_matrix((height, width, channels), hierarchical_blockshape, hierarchical_permute_at_level, min_blocksize=8, seed=seed, twist=True, withinverse=True, strict=False)
        (G, Ginv) = (Ainv.dot(G).dot(A), Ainv.dot(Ginv).dot(A))  # CxHxW -> HxWxC -> hierarchical permute in HxWxC order -> CxHxW
        if memoryorder != 'channel':
            (G, Ginv) = (c.dot(G).dot(cinv), c.dot(Ginv).dot(cinv))        
    elif global_geometric == 'givens_orthogonal':
        assert alpha is not None
        assert tileshape is None, "Global givens rotation orthogonal matrix is not tile compressible"
        (G, Ginv) = sparse_orthogonal_matrix(N, int(alpha), balanced=True, withinverse=True)        
    else:
        raise ValueError("Invalid global geometric transform '%s' - must be in '%s'" % (global_geometric, str(allowable_global_geometric)))
    (G, Ginv) = (sparse_affine_to_linear(G), sparse_affine_to_linear(Ginv))
    
    if local_geometric == 'identity':
        (g, ginv) = (sparse_identity_matrix(N), sparse_identity_matrix(N))        
    elif local_geometric == 'permutation':
        assert blocksize is not None and height==width
        g = keynet.sparse.DiagonalTiledMatrix(sparse_permutation_matrix(blocknumel), shape=(H, H)).tocoo().astype(np.float32)   # spatial repeat
        g = keynet.sparse.DiagonalTiledMatrix(g, shape=(N,N)).tocoo().astype(np.float32)  # channel repeat
        ginv = g.transpose()

    elif local_geometric == 'doubly_stochastic':
        assert blocksize is not None and alpha is not None and height == width
        assert blocksize < 8192, "Blocksize %d must be less than 8192, since doubly_stochastic requires the direct inverse of a dense matrix" % blocksize
        (g, ginv) = sparse_random_diagonally_dominant_doubly_stochastic_matrix(blocknumel, int(alpha), withinverse=True)  # expensive inverse
        g = keynet.sparse.DiagonalTiledMatrix(keynet.sparse.DiagonalTiledMatrix(g, shape=(H, H)).tocoo(), shape=(N,N)).tocoo()  # spatial, channel repeat
        ginv = keynet.sparse.DiagonalTiledMatrix(keynet.sparse.DiagonalTiledMatrix(ginv, shape=(H, H)).tocoo(), shape=(N,N)).tocoo()  # spatial, channel repeat
    elif local_geometric == 'givens_orthogonal':
        assert alpha is not None and blocksize is not None and height == width
        (g, ginv) = sparse_orthogonal_matrix(blocknumel, int(alpha), balanced=True, withinverse=True)
        (A, Ainv) = sparse_permutation_matrix(blocknumel, withinverse=True)
        (g, ginv) = (A.dot(g), ginv.dot(Ainv))
        g = keynet.sparse.DiagonalTiledMatrix(keynet.sparse.DiagonalTiledMatrix(g, shape=(H, H)).tocoo(), shape=(N,N)).tocoo().astype(np.float32)  # spatial, channel repeat
        ginv = keynet.sparse.DiagonalTiledMatrix(keynet.sparse.DiagonalTiledMatrix(ginv, shape=(H, H)).tocoo(), shape=(N,N)).tocoo().astype(np.float32)  # spatial, channel repeat
    else:
        raise ValueError("Invalid local geometric transform '%s' - must be in '%s'" % (local_geometric, str(allowable_local_geometric)))        
    (g, ginv) = (sparse_affine_to_linear(g), sparse_affine_to_linear(ginv))
    
    if global_photometric == 'identity':
        (P, Pinv) = (sparse_affine_to_linear(sparse_identity_matrix(N)), sparse_affine_to_linear(sparse_identity_matrix(N)))
    elif global_photometric == 'uniform_random_gain':
        assert tileshape is None, "Global permutation is not tile compressible"
        assert beta is not None and beta > 0
        (P, Pinv) = sparse_uniform_random_diagonal_matrix(N, beta, bias=1, withinverse=True)
        (P, Pinv) = (sparse_affine_to_linear(P), sparse_affine_to_linear(Pinv))
    elif global_photometric == 'uniform_random_bias':
        assert gamma is not None and gamma > 0
        (P, Pinv) = diagonal_affine_to_linear(sparse_identity_matrix(N), gamma*np.random.rand(N,1), withinverse=True)
    elif global_photometric == 'linear_bias':
        assert gamma is not None and gamma > 0
        (P, Pinv) = diagonal_affine_to_linear(sparse_identity_matrix(N), (gamma/float(N))*np.array(range(0,N)).reshape(N,1), withinverse=True)
    elif global_photometric == 'uniform_random_affine':
        assert tileshape is None, "Global permutation is not tile compressible"
        assert beta is not None and beta > 0 and gamma is not None and gamma > 0
        P = sparse_uniform_random_diagonal_matrix(N, beta, bias=1)
        (P, Pinv) = diagonal_affine_to_linear(P, gamma*np.random.rand(N,1), withinverse=True)
    elif global_photometric == 'blockwise_constant_bias':
        assert gamma is not None and gamma > 0
        assert blocksize is not None
        bias = gamma*np.random.rand(int(np.ceil(N//blocksize)), 1).dot(np.ones( (1, blocknumel))).flatten()[0:N].reshape(N,1)
        (P, Pinv) = diagonal_affine_to_linear(sparse_identity_matrix(N), bias, withinverse=True)        
    else:
        raise ValueError("Invalid global photometric transform '%s' - must be in '%s'" % (global_photometric, str(allowable_photometric)))                

    if local_photometric == 'identity':
        (p, pinv) = (sparse_affine_to_linear(sparse_identity_matrix(N)), sparse_affine_to_linear(sparse_identity_matrix(N)))
    elif local_photometric == 'uniform_random_gain':
        assert blocksize is not None
        assert beta is not None and beta > 0
        (p, pinv) = sparse_uniform_random_diagonal_matrix(blocknumel, beta, bias=1, withinverse=True)
        (p, pinv) = (sparse_block_diagonal(p, shape=(N,N)), sparse_block_diagonal(pinv, shape=(N,N)))
        (p, pinv) = (sparse_affine_to_linear(p), sparse_affine_to_linear(pinv))
    elif local_photometric == 'uniform_random_bias':
        # FIXME: local bias does not respect memoryorder
        assert blocksize is not None 
        assert gamma is not None and gamma > 0
        bias = np.tile(gamma*np.random.rand(blocknumel), int(np.ceil(N / blocknumel)))[0:N].reshape(N,1)
        (p, pinv) = diagonal_affine_to_linear(sparse_identity_matrix(N), bias=bias, withinverse=True)        
    elif local_photometric == 'uniform_random_affine':
        assert blocksize is not None 
        assert beta is not None and beta > 0 and gamma is not None and gamma > 0
        p = sparse_uniform_random_diagonal_matrix(blocknumel, beta, bias=1)
        bias = np.tile(gamma*np.random.rand(blocknumel), int(np.ceil(N / blocknumel)))[0:N].reshape(N,1)
        (p, pinv) = diagonal_affine_to_linear(sparse_block_diagonal(p, shape=(N,N)), bias=bias, withinverse=True)
    elif local_photometric == 'blockwise_constant_bias':
        raise ValueError('blockwise_constant_bias supported for global_photometric testing only')
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
                                                 local_photometric=local_photometric if 'relu' not in layername or local_photometric == 'identity' else 'uniform_random_gain',
                                                 global_geometric=global_geometric if 'relu' not in layername or global_geometric == 'identity' else 'identity',
                                                 local_geometric=local_geometric if 'relu' not in layername or local_geometric == 'identity' else 'permutation',
                                                 memoryorder=memoryorder,                                                                                                  
                                                 blocksize=blocksize, tileshape=tileshape, alpha=alpha, beta=beta, gamma=gamma, hierarchical_blockshape=hierarchical_blockshape, hierarchical_permute_at_level=hierarchical_permute_at_level)
    
    sensor = KeyedSensor(inshape, f_keypair('input', inshape))
    model = KeyedModel(net, inshape, sensor.key(), f_keypair, f_layergen, do_output_encryption=do_output_encryption) if net is not None else None
    return (sensor, model)


def IdentityKeynet(inshape, net, backend='scipy'):
    return Keynet(inshape, net, backend=backend)


def PermutationKeynet(inshape, net, do_output_encryption=False):
    return Keynet(inshape, net, global_geometric='permutation', do_output_encryption=do_output_encryption)


def TiledIdentityKeynet(inshape, net, tilesize):
    return Keynet(inshape, net, backend='scipy', tileshape=(tilesize, tilesize))


def TiledPermutationKeynet(inshape, net, tilesize):
    return Keynet(inshape, net, local_geometric='permutation', backend='scipy', tileshape=(tilesize, tilesize), blocksize=tilesize)

def TiledOrthogonalKeynet(inshape, net, tilesize, hierarchical_permute_at_level=(0,1)):
    return Keynet(inshape, net, tileshape=(tilesize, tilesize), 
                  global_geometric='hierarchical_permutation', hierarchical_blockshape=(2,2), hierarchical_permute_at_level=hierarchical_permute_at_level,
                  global_photometric='identity',
                  local_geometric='givens_orthogonal', alpha=tilesize, blocksize=tilesize,
                  local_photometric='uniform_random_affine', beta=0.1, gamma=100.0,
                  memoryorder='block')

def OpticalFiberBundleKeynet(inshape, net):
    f_keypair = keygen('identity', 'scipy')  # FIXME
    sensor = OpticalFiberBundle(inshape, f_keypair('input', inshape))
    model = KeyedModel(net, inshape, sensor.key(), f_keypair, do_output_encryption=do_output_encryption, verbose=verbose)
    return (sensor, model)
    
