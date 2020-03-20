import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse
import torch.sparse
import vipy
from collections import OrderedDict
from keynet.torch import homogenize, dehomogenize
from vipy.util import try_import
from keynet.torch import homogenize_matrix, scipy_coo_to_torch_sparse
from keynet.sparse import sparse_permutation_matrix_with_inverse, sparse_permutation_matrix, sparse_generalized_permutation_block_matrix_with_inverse, sparse_identity_matrix
from keynet.sparse import sparse_stochastic_matrix_with_inverse, sparse_generalized_stochastic_matrix_with_inverse, sparse_generalized_permutation_matrix_with_inverse, sparse_identity_matrix_like
from keynet.sparse import sparse_permutation_tiled_matrix_with_inverse, sparse_identity_tiled_matrix_with_inverse, sparse_generalized_permutation_tiled_matrix_with_inverse
from keynet.sparse import sparse_generalized_stochastic_tiled_matrix_with_inverse, sparse_block_permutation_tiled_matrix_with_inverse, sparse_channelorder_to_blockorder
import keynet.torch
import keynet.layer
import keynet.fiberbundle
from keynet.util import blockview
import keynet.sparse
import keynet.torch


class KeyedModel(object):
    def __init__(self, net, inshape, inkey, f_layername_to_keypair, do_output_encryption=False, verbose=True):
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
            if verbose:
                print('[keynet.layers.KeyNet]: Keying "%s"' % k)
            assert k in layerkey, 'Key not found for layer "%s"' % k
            assert k in netshape, 'Layer name not found in net shape for layer "%s"' % k
            assert 'A' in layerkey[k] and 'Ainv' in layerkey[k], 'Keys not specified for layer "%s"' % k
            assert 'inshape' in netshape[k], 'Layer input shape not specified for layer "%s"' % k

            # Replace torch layers with keyed layers 
            if isinstance(m, nn.Conv2d):
                assert m.padding[0] == m.kernel_size[0]//2 and m.padding[1] == m.kernel_size[1]//2, "Padding is assumed to be equal to (kernelsize-1)/2"
                m_keyed = keynet.layer.KeyedConv2d(inshape=netshape[k]['inshape'],
                                                   out_channels=m.out_channels,
                                                   in_channels=m.in_channels,
                                                   kernel_size=m.kernel_size,
                                                   stride=m.stride)
                d_name_to_keyedmodule[k] = m_keyed
                if '%s_bn' % k not in layernames:                
                    d_name_to_keyedmodule[k] = d_name_to_keyedmodule[k].key(m.weight, m.bias, layerkey[k]['A'], layerkey[k]['Ainv'])
            elif isinstance(m, nn.AvgPool2d):
                m_keyed = keynet.layer.KeyedAvgpool2d(inshape=netshape[k]['inshape'],
                                                      kernel_size=m.kernel_size,
                                                      stride=m.stride)
                d_name_to_keyedmodule[k] = m_keyed.key(layerkey[k]['A'], layerkey[k]['Ainv'])  
            elif isinstance(m, nn.ReLU):
                m_keyed = keynet.layer.KeyedReLU()
                d_name_to_keyedmodule[k] = m_keyed.key(layerkey[k]['A'], layerkey[k]['Ainv'])                                
            elif isinstance(m, nn.Linear):                
                m_keyed = keynet.layer.KeyedLinear(out_features=m.out_features,
                                                   in_features=m.in_features)
                if '%s_bn' % k not in layernames:                                
                    d_name_to_keyedmodule[k] = m_keyed.key(m.weight, m.bias, layerkey[k]['A'], layerkey[k]['Ainv'])   
            elif isinstance(m, nn.BatchNorm2d):
                assert ('_' in k) and hasattr(net, k.split('_')[0]), "Batchnorm layers must be named 'mylayername_bn' for corresponding linear layer mylayername.  (e.g. 'conv3_bn')"
                k_prev = k.split('_')[0]
                assert k_prev in d_name_to_keyedmodule, "Batchnorm layer named 'mylayer_bn' must come after 'mylayer' (e.g. 'conv3_bn' must come after 'conv3')"
                m_prev = getattr(net, k_prev)
                (bn_weight, bn_bias) = keynet.torch.fuse_conv2d_and_bn(m_prev.weight, m_prev.bias,
                                                                       m.running_mean, m.running_var, 1E-5,
                                                                       m.weight, m.bias)
                # Replace module k_prev with fused weights
                d_name_to_keyedmodule[k_prev].key(bn_weight, bn_bias, layerkey[k_prev]['A'], layerkey[k_prev]['Ainv'])
            elif isinstance(m, nn.Dropout):
                pass  # identity matrix at test time, ignore me
            else:
                raise ValueError('unsupported layer type "%s"' % str(type(m)))

        self._keynet = nn.Sequential(d_name_to_keyedmodule)
        self._embeddingkey = layerkey['output']
        self._imagekey = layerkey['input']
        
    def forward(self, img_cipher, outkey=None):
        outkey = outkey if outkey is not None else self.embeddingkey()
        y_cipher = self._keynet.forward(img_cipher)
        return dehomogenize(self.decrypt(y_cipher, outkey) if outkey is not None else y_cipher)
    
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

    def num_parameters(self):
        return sum([c.nnz() for (k,c) in self._keynet.named_children() if hasattr(c, 'nnz')])
    

class KeyedSensor(keynet.layer.KeyedLayer):
    def __init__(self, inshape, keypair, reorder=None):
        super(KeyedSensor, self).__init__() 
        (self._encryptkey, self._decryptkey) = keypair
        self._reorderkey = reorder.transpose() if reorder is not None else None
        self._inshape = inshape
        self._tensor = None
        
    def __repr__(self):
        return str('<KeySensor: height=%d, width=%d, channels=%d>' % (self._inshape[1], self._inshape[2], self._inshape[0]))
    
    def load(self, imgfile):
        im = vipy.image.Image(imgfile).resize(height=self._inshape[1], width=self._inshape[2])
        if self._inshape[0] == 1:
            im = im.grey()
        self._tensor = im.float().torch().contiguous()
        return self

    def tensor(self, x=None):
        if x is None:
            return self._tensor
        else:
            self._tensor = x.clone().type(torch.FloatTensor)
            return self

    def image(self):
        if self.isencrypted():
            x = self._tensor if self._reorderkey is None else self._reorderkey.torchdot(self._tensor.t()).t()
            img = dehomogenize(x).reshape(1, *self._inshape)
        else:
            img = self._tensor

        img = np.squeeze(img.permute(2,3,1,0).numpy())  # 1xCxHxW -> HxWxC
        colorspace = 'float' if img.dtype == np.float32 else None
        colorspace = 'rgb' if img.dtype == np.uint8 and img.shape[2] == 3 else colorspace
        colorspace = 'lum' if img.dtype == np.uint8 and img.shape[2] == 1 else colorspace
        return vipy.image.Image(array=img, colorspace=colorspace)

    def keypair(self):
        return (self._encryptkey, self._decryptkey)

    def key(self):
        return self._decryptkey
    
    def isencrypted(self):
        """An encrypted image is converted from NxCxHxW tensor to Nx(C*H*W+1)"""
        return self.isloaded() and self._tensor.ndim == 2

    def isloaded(self):
        return self._tensor is not None
    
    def encrypt(self, x_raw=None):
        """img_tensor is NxCxHxW, return Nx(C*H*W+1) homogenized and encrypted"""
        self.tensor(x_raw) 
        assert self.isloaded(), "Load image first"
        self.W = self._encryptkey    # Used in super().forward()
        self._tensor = super(KeyedSensor, self).forward(homogenize(self._tensor)) if not self.isencrypted() else self._tensor
        return self
        
    def decrypt(self, x_cipher=None):
        """x_cipher is Nx(C*H*W+1) homogenized, convert to NxCxHxW decrypted"""
        self.tensor(x_cipher)
        assert self.isloaded(), "Load image first"        
        self.W = self._decryptkey   # Used in super().forward()               
        self._tensor = dehomogenize(super(KeyedSensor, self).forward(self._tensor)).reshape(self._tensor.shape[0], *self._inshape) if self.isencrypted() else self._tensor
        return self

    
class OpticalFiberBundle(KeyedSensor):
    def __init__(self, inshape=(3,512,512), keypair=None):
        (encryptkey, decryptkey) = keypair if keypair is not None else keygen('identity', 'scipy')('input', inshape)      
        super(OpticalFiberBundle, self).__init__(inshape, (encryptkey, decryptkey))
    
    def load(self, imgfile):
        img_color = vipy.image.Image(imgfile).maxdim(max(self._inshape)).centercrop(height=self._inshape[1], width=self._inshape[2]).numpy()
        img_sim = keynet.fiberbundle.simulation(img_color, h_xtalk=0.05, v_xtalk=0.05, fiber_core_x=16, fiber_core_y=16, do_camera_noise=True)
        return vipy.image.Image(array=np.uint8(img_sim), colorspace='rgb')
    

class KeyPair(object):
    def __init__(self, backend='scipy', n_processes=1):
        self._SparseMatrix = keynet.sparse.SparseMatrix if backend=='scipy' else keynet.torch.SparseMatrix
        self._SparseTiledMatrix = keynet.sparse.SparseTiledMatrix if backend=='scipy' else keynet.torch.SparseTiledMatrix
        self._n_processes = n_processes
        
    def identity(self):
        return lambda layername, outshape: (self._SparseMatrix(sparse_identity_matrix(np.prod(outshape)+1).tocoo()),
                                            self._SparseMatrix(sparse_identity_matrix(np.prod(outshape)+1).tocoo()))
    def permutation(self):
        return lambda layername, outshape: tuple(self._SparseMatrix(m) for m in sparse_permutation_matrix_with_inverse(np.prod(outshape)+1))

    def stochastic(self, alpha, beta):
        assert alpha is not None and beta is not None, "Invalid (alpha, beta)"
        return lambda layername, outshape: (tuple(self._SparseMatrix(m) for m in sparse_generalized_stochastic_matrix_with_inverse(np.prod(outshape)+1, alpha, beta)) if 'relu' not in layername else 
                                            tuple(self._SparseMatrix(m) for m in sparse_generalized_permutation_matrix_with_inverse(np.prod(outshape)+1, beta)))
    def tiled_identity(self, tilesize):
        assert tilesize is not None, "invalid tilesize"
        return lambda layername, outshape: tuple(A.parallel(self._n_processes) for A in sparse_identity_tiled_matrix_with_inverse(np.prod(outshape)+1, tilesize, tiler=self._SparseTiledMatrix))
    
    def tiled_permutation(self, tilesize):
        assert tilesize is not None, "invalid tilesize"
        return lambda layername, outshape: tuple(A.parallel(self._n_processes) for A in sparse_permutation_tiled_matrix_with_inverse(np.prod(outshape)+1, tilesize, tiler=self._SparseTiledMatrix))

    def tiled_stochastic(self, alpha, beta, tilesize):    
        assert tilesize is not None, "invalid tilesize"
        assert alpha is not None and beta is not None, "Invalid (alpha, beta)"
        return lambda layername, outshape: (tuple(A.parallel(self._n_processes) for A in sparse_generalized_stochastic_tiled_matrix_with_inverse(np.prod(outshape)+1, tilesize, alpha, beta, tiler=self._SparseTiledMatrix)) if 'relu' not in layername else 
                                            tuple(A.parallel(self._n_processes) for A in sparse_generalized_permutation_tiled_matrix_with_inverse(np.prod(outshape)+1, tilesize, beta, tiler=self._SparseTiledMatrix)))
    
    def tiled_block_permutation(self, tilesize): 
        assert tilesize is not None, "invalid tilesize"
        
        def _f_reorder(outshape):
            B_channel_to_block = self._SparseMatrix(sparse_channelorder_to_blockorder(outshape, tilesize, homogenize=True)).parallel(self._n_processes)
            return B_channel_to_block
        
        def _f_keypair(layername, outshape):
            (A, Ainv) = sparse_block_permutation_tiled_matrix_with_inverse(np.prod(outshape)+1, tilesize*tilesize, tiler=self._SparseTiledMatrix)
            (A, Ainv) = (A.parallel(self._n_processes), Ainv.parallel(self._n_processes))
            B_channel_to_block = _f_reorder(outshape)            
            return (A.matmul(B_channel_to_block), Ainv.transpose().matmul(B_channel_to_block).transpose())
        
        return (_f_keypair, _f_reorder)

    
def keygen(format, backend, alpha=None, beta=0, tilesize=None, n_processes=1):
    formats = set(['identity', 'permutation', 'stochastic', 'tiled-identity', 'tiled-permutation', 'tiled-stochastic', 'tiled-blockpermutation'])
    backends = set(['torch', 'scipy'])
    assert format in formats, "Invalid format '%s' - must be in '%s'" % (format, str(formats))
    assert backend in backends, "Invalid backend '%s' - must be in '%s'" % (backend, str(backends))

    keypair = KeyPair(backend, n_processes)
    if format == 'identity':
        return keypair.identity()
    elif format == 'permutation':
        return keypair.permutation()    
    elif format == 'stochastic':
        return keypair.stochastic(alpha, beta)
    elif format == 'tiled-identity':
        return keypair.tiled_identity(tilesize)
    elif format == 'tiled-permutation':
        return keypair.tiled_permutation(tilesize)            
    elif format == 'tiled-stochastic':
        return keypair.tiled_stochastic(alpha, beta, tilesize)                
    elif format == 'tiled-blockpermutation':
        return keypair.tiled_block_permutation(tilesize)
    else:
        raise ValueError("Invalid format '%s' - must be in '%s'" % (format, str(formats)))

    
def Keynet(inshape, net, format, backend, do_output_encryption=False, verbose=True, alpha=None, beta=None, tilesize=None, n_processes=1):
    f_keypair = keygen(format, backend, alpha, beta, tilesize, n_processes=n_processes)
    sensor = KeyedSensor(inshape, f_keypair('input', inshape))
    model = KeyedModel(net, inshape, sensor.key(), f_keypair, do_output_encryption=do_output_encryption, verbose=verbose)
    return (sensor, model)


def IdentityKeynet(inshape, net):
    return Keynet(inshape, net, 'identity', 'scipy')


def PermutationKeynet(inshape, net, do_output_encryption=False):
    return Keynet(inshape, net, 'permutation', 'scipy', do_output_encryption)    


def StochasticKeynet(inshape, net, alpha, beta=0, do_output_encryption=False):
    return Keynet(inshape, net, 'stochastic', 'scipy', do_output_encryption, alpha=alpha, beta=beta)    


def TiledIdentityKeynet(inshape, net, tilesize, n_processes=1):
    return Keynet(inshape, net, 'tiled-identity', 'scipy', do_output_encryption=False, tilesize=tilesize, n_processes=n_processes)


def TiledPermutationKeynet(inshape, net, tilesize, do_output_encryption=False, n_processes=1):
    return Keynet(inshape, net, 'tiled-permutation', 'scipy', do_output_encryption, tilesize=tilesize, n_processes=n_processes)


def TiledStochasticKeynet(inshape, net, tilesize, alpha, beta=0, do_output_encryption=False, n_processes=1):
    return Keynet(inshape, net, 'tiled-stochastic', 'scipy', do_output_encryption, tilesize=tilesize, alpha=alpha, beta=beta, n_processes=n_processes)


def TiledStochasticKeySensor(inshape, tilesize, alpha, beta=0):
    f_keypair = keygen('tiled-stochastic', 'scipy', alpha, beta, tilesize)
    return KeyedSensor(inshape, f_keypair('input', inshape))


def TiledBlockPermutationKeySensor(inshape, tilesize):
    (f_keypair, f_reorder) = keygen('tiled-blockpermutation', 'scipy', tilesize=tilesize)
    return KeyedSensor(inshape, f_keypair('input', inshape), reorder=f_reorder(inshape))


def OpticalFiberBundleKeynet(inshape, net):
    f_keypair = keygen('identity', 'scipy')  # FIXME
    sensor = OpticalFiberBundle(inshape, f_keypair('input', inshape))
    model = KeyedModel(net, inshape, sensor.key(), f_keypair, do_output_encryption=do_output_encryption, verbose=verbose)
    return (sensor, model)
    
