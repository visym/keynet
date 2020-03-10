from torch import nn
from numpy.linalg import multi_dot 
from keynet.torch import homogenize_matrix, sparse_toeplitz_conv2d, sparse_toeplitz_avgpool2d, scipy_coo_to_torch_sparse
from keynet.sparse import sparse_permutation_matrix_with_inverse, sparse_permutation_matrix, sparse_generalized_permutation_block_matrix_with_inverse, sparse_identity_matrix
from keynet.sparse import sparse_stochastic_matrix_with_inverse, sparse_generalized_permutation_matrix_with_inverse, sparse_identity_matrix_like
from keynet.sparse import sparse_permutation_tiled_matrix_with_inverse, SparseTiledMatrix, sparse_identity_tiled_matrix_with_inverse
import torch
import numpy as np
import scipy.sparse
import torch.nn.functional as F
import torch.sparse
from torch import nn, optim
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import warnings
from keynet.sparse import SparseTiledMatrix
import keynet.torch
import vipy
from collections import OrderedDict
from keynet.torch import homogenize, dehomogenize
from vipy.util import try_import
try:
    import cupyx
    import cupy
except:    
    pass  # Exception on init if cupy backend used


class KeyedLayer(nn.Module):
    def __init__(self, backend, W=None):
        super(KeyedLayer, self).__init__()        
        assert backend in set(['torch', 'cupy', 'scipy', 'tiled']), "Unknown backend='%s' - Must be in ['torch', 'cupy', 'scipy', 'tiled']" % backend
        self._backend = backend
        self.W = W

    def forward(self, x_affine):
        """Keyed sparse linear transformation, using the backend specified for sparse matrix multiplication

           Input: 
             x_affine is (N x C*U*V+1) torch tensor

           Output:
             y=(W*x^T)^T, (N x -1) torch tensor, right multiplied

        """
        if self._backend == 'torch':
            if isinstance(self.W, torch.sparse.FloatTensor):
                return torch.sparse.mm(self.W, x_affine.t()).t()   # torch.sparse, right multiply
            else:
                return torch.matmul(self.W, x_affine.t()).t()      # torch dense, right multiply
        elif self._backend == 'scipy':
            return torch.as_tensor(self.W.dot(x_affine.detach().numpy().transpose())).t()  # scipy.sparse, right multiply required
        elif self._backend == 'cupy':
            try_import(package='cupy', pipname='cupy, cupyx')
            return from_dlpack(self.W.dot(cupy.fromDlpack(to_dlpack(x_affine.t()))).toDlpack()).t()
        elif self._backend == 'tiled':
            return self.W.dot(x_affine.t()).t()  # right multiply
        else:
            raise ValueError('Invalid backend "%s"' % self._backend)

    def key(self, W, A, Ainv):
        #assert self._backend == 'tiled' or (scipy.sparse.issparse(A) and scipy.sparse.issparse(Ainv)), "(A,Ainv) must be scipy.sparse matrices"
        #assert Ainv.shape[0] == Ainv.shape[1], "A*Ainv must be conformal"        
        #assert np.prod(inshape)+1 == Ainv.shape[0], "Ainv does not have conformal shape"
        
        if self._backend == 'torch':
            if isinstance(W, torch.sparse.FloatTensor):            
                A = scipy_coo_to_torch_sparse(A, device='cuda')
                Ainv = scipy_coo_to_torch_sparse(Ainv, device='cuda')
                self.W = scipy_coo_to_torch_sparse(W.tocoo(), device='cuda')
                self.W = Ainv.t().dot(self.W.dot(A.t()))  # transpose for left multiply
            else:
                self.W = torch.matmul(torch.as_tensor(Ainv.todense().transpose()), W)
                if A is not None:
                    self.W = torch.matmul(self.W, A.todense().transpose())
                self.W = torch.as_tensor(self.W)                
                        
        elif self._backend == 'scipy':
            if scipy.sparse.issparse(W) or W is None:
                if A is None:
                    self.W = W.dot(Ainv)  # right multiply in scipy
                elif W is None:
                    self.W = A.dot(Ainv)  # right multiply in scipy
                else:
                    self.W = A.dot(W.dot(Ainv))  # right multiply in scipy                    
            elif isinstance(W, torch.FloatTensor):
                if W is None:
                    self.W = Ainv
                else:
                    self.W = Ainv.transpose().dot(W.t().detach().numpy()).transpose()   # right multiply for scipy. yuck
                if A is not None:
                    self.W = A.dot(self.W)   # right multiply
            else:
                raise ValueError('Invalid W - Should be torch or numpy')

        elif self._backend == 'cupy':
            A = cupyx.scipy.sparse.csc_matrix(A.tocsc())
            Ainv = cupyx.scipy.sparse.csc_matrix(Ainv.tocsc())            
            self.W = cupyx.scipy.sparse.csc_matrix(W.tocsc())  # csc for left mulipl
            self.W = Ainv.t().dot(self.W.dot(A.t()))  # transpose for left multiply
            
        elif self._backend == 'tiled':
            # assert isinstance(A, SparseTiledMatrix) and isinstance(Ainv, SparseTiledMatrix), "backend='tiled' requires SparseTiledMatrix inputs"
            # assert A.tilesize() == Ainv.tilesize(), "Invalid keys

            if not scipy.sparse.issparse(W) and W is not None:
                W = scipy.sparse.coo_matrix(W.detach().numpy())
            
            if W is None:
                self.W = A.prod(Ainv)  # right multiply                    
            elif A is None:
                self.W = (SparseTiledMatrix(coo_matrix=W.tocoo(), tilesize=Ainv.tilesize()).prod(Ainv))   # right multiply                    
            else:
                T = SparseTiledMatrix(coo_matrix=W.tocoo(), tilesize=Ainv.tilesize())
                TAinv = T.prod(Ainv)
                self.W = A.prod(TAinv)  # right multiply
        else:
            raise ValueError('Invalid backend "%s"' % self._backend)
        return self

        
class KeyedConv2d(KeyedLayer):
    def __init__(self, inshape, in_channels, out_channels, kernel_size, stride, backend='torch_sparse'):
        super(KeyedConv2d, self).__init__(backend=backend)

        assert len(kernel_size)==1 or len(kernel_size)==2 and (kernel_size[0] == kernel_size[1]), "Kernel must be square"
        assert len(stride)==1 or len(stride)==2 and (stride[0] == stride[1]), "Strides must be isotropic"
        
        self.stride = stride[0] if len(stride)==2 else stride
        self.kernel_size = kernel_size[0] if len(kernel_size) == 2 else kernel_size        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inshape = inshape
        assert len(inshape) == 3, "Inshape must be (C,H,W) for the shape of the tensor at the input to this layer"""
        
    def extra_repr(self):
        return str('<KeyedConv2d: W=%s, in_channels=%d, out_channels=%d, kernel_size=%s, stride=%s, backend=%s>' (str(self.W.shape), self.in_channels, self.out_channels, str(self.kernel_size), str(self.stride), str(self.backend)))
        
    def key(self, w, b, A, Ainv):
        """Assign key to conv2d
        
        Inputs:
            -w: torch weight parameter tensor for conv2d
            -b: torch bias parameter tensor for conv2d (may be None)
            -A:  Layer output key, must be numpy sparse matrix in COO format
            -Ainv:  Layer input key, must be numpy sparse matrix in COO format such that A.dot(Ainv) = I
            -inshape:  (C,H,W) tuple defining the input shape of a forward tensor

        Output:
            -None

        """
        assert (w.shape[0] == self.out_channels and
                w.shape[1] == self.in_channels and
                w.shape[2] == self.kernel_size and
                b.shape[0] == self.out_channels), "Invalid input"
        W = sparse_toeplitz_conv2d(self.inshape, w.detach().numpy(), bias=b.detach().numpy(), stride=self.stride)   # Right multiply
        return super(KeyedConv2d, self).key(W, A, Ainv)

        
class KeyedLinear(KeyedLayer):
    def __init__(self, in_features, out_features, backend='scipy'):
        super(KeyedLinear, self).__init__(backend=backend)
        self.in_features = in_features
        self.out_features = out_features

    def extra_repr(self):
        return str('<KeyedLinear: W=%s, in_features=%d, out_features=%d, backend=%s>' (str(self.W.shape), self.in_features, self.out_features, str(self.backend)))
        
    def key(self, w, b, A, Ainv):
        W = homogenize_matrix(w, b).t()  # transposed for right multiply
        return super(KeyedLinear, self).key(W, A, Ainv)

    
class KeyedReLU(KeyedLayer):
    def __init__(self, backend='scipy'):
        super(KeyedReLU, self).__init__(backend=backend)

    def extra_repr(self):
        return str('<KeyedReLU: W=%s, backend=%s>' (str(self.W.shape), str(self.backend)))
        
    def key(self, B, Ainv):
        return super(KeyedReLU, self).key(None, B, Ainv)
        
    def forward(self, x_affine):
        return F.relu(super(KeyedReLU, self).forward(x_affine))

    
class KeyedAvgpool2d(KeyedLayer):
    def __init__(self, inshape, kernel_size, stride, backend='scipy'):
        super(KeyedAvgpool2d, self).__init__(backend=backend)
        self.kernel_size = kernel_size
        self.stride = stride
        self.inshape = inshape

    def extra_repr(self):
        return str('<KeyedAvgpool2d: W=%s, kernel_size=%s, stride=%s, backend=%s>' (str(self.W.shape), str(self.kernel_size), str(self.stride), str(self.backend)))
    
    def key(self, A, Ainv):
        W = sparse_toeplitz_avgpool2d(self.inshape, (self.inshape[0], self.inshape[0], self.kernel_size, self.kernel_size), self.stride)  # right multiply
        return super(KeyedAvgpool2d, self).key(W, A, Ainv)


    
class KeyNet(object):
    def __init__(self, net, inshape, inkey, f_layername_to_keypair, do_output_encryption=False, backend='scipy', verbose=True):

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
        d_name_to_keyedmodule = OrderedDict()        
        for (k,m) in net.named_children():
            if verbose:
                print('[keynet.layers.KeyNet]: Keying "%s"' % k)
            assert k in layerkey, 'Key not found for layer "%s"' % k
            assert k in netshape, 'Layer name not found in net shape for layer "%s"' % k
            assert 'A' in layerkey[k] and 'Ainv' in layerkey[k], 'Keys not specified for layer "%s"' % k
            assert 'inshape' in netshape[k], 'Layer input shape not specified for layer "%s"' % k

            # Replace with keyed versions
            if isinstance(m, nn.Conv2d):
                assert m.padding[0] == m.kernel_size[0]//2 and m.padding[1] == m.kernel_size[1]//2, "Padding is assumed to be equal to (kernelsize-1)/2"
                m_keyed = KeyedConv2d(inshape=netshape[k]['inshape'],
                                      out_channels=m.out_channels,
                                      in_channels=m.in_channels,
                                      kernel_size=m.kernel_size,
                                      stride=m.stride,
                                      backend=backend)                                      
                d_name_to_keyedmodule[k] = m_keyed.key(m.weight, m.bias, layerkey[k]['A'], layerkey[k]['Ainv'])
            elif isinstance(m, nn.AvgPool2d):
                m_keyed = KeyedAvgpool2d(inshape=netshape[k]['inshape'],
                                         kernel_size=m.kernel_size,
                                         stride=m.stride,
                                         backend=backend)                                         
                d_name_to_keyedmodule[k] = m_keyed.key(layerkey[k]['A'], layerkey[k]['Ainv'])  
            elif isinstance(m, nn.ReLU):
                m_keyed = KeyedReLU(backend=backend)
                d_name_to_keyedmodule[k] = m_keyed.key(layerkey[k]['A'], layerkey[k]['Ainv'])                                
            elif isinstance(m, nn.Linear):                
                m_keyed = KeyedLinear(out_features=m.out_features,
                                      in_features=m.in_features,
                                      backend=backend)
                d_name_to_keyedmodule[k] = m_keyed.key(m.weight, m.bias, layerkey[k]['A'], layerkey[k]['Ainv'])   
            elif isinstance(m, nn.BatchNorm2d):
                assert ('_' in k) and hasattr(net, k.split('_')[0]), "Batchnorm layers must be named 'mylayername_bn' for corresponding linear layer mylayername.  (e.g. 'conv3_bn')"
                k_prev = k.split('_')[0]
                assert k_prev in d_name_to_keyedmodule, "Batchnorm layer named 'mylayer_bn' must come after 'mylayer'"
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
        self._backend = backend
        
    def forward(self, img_cipher, outkey=None):
        outkey = outkey if outkey is not None else self.embeddingkey()
        y_cipher = self._keynet.forward(img_cipher)
        return dehomogenize(self.decrypt(y_cipher, outkey) if outkey is not None else y_cipher)
    
    def decrypt(self, y_cipher, outkey=None): 
        outkey = outkey if outkey is not None else self.embeddingkey()
        return KeyedLayer(backend=self._backend, W=outkey).forward(y_cipher) if outkey is not None else y_cipher

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

        
class IdentityKeynet(KeyNet):
    """keynet.layers.IdentityKeynet class, useful for testing only"""
    def __init__(self, net, inshape, inkey, backend='scipy', verbose=True):
        f_layername_to_keypair = lambda layername, outshape: (sparse_identity_matrix(np.prod(outshape)+1),
                                                              sparse_identity_matrix(np.prod(outshape)+1))
        super(IdentityKeynet, self).__init__(net, inshape, inkey, f_layername_to_keypair, do_output_encryption=False, backend=backend, verbose=verbose)

        
class PermutationKeynet(KeyNet):
    def __init__(self, net, inshape, inkey, do_output_encryption=True, backend='scipy', verbose=True):
        f_layername_to_keypair = lambda layername, outshape: sparse_permutation_matrix_with_inverse(np.prod(outshape)+1)
        super(PermutationKeynet, self).__init__(net, inshape, inkey, f_layername_to_keypair, do_output_encryption, backend, verbose)


class StochasticKeynet(KeyNet):
    def __init__(self, net, inshape, inkey, alpha, do_output_encryption=True, backend='scipy', verbose=True):    
        f_layername_to_keypair = lambda layername, outshape: (sparse_stochastic_matrix_with_inverse(np.prod(outshape)+1, alpha) if 'relu' not in layername else
                                                              sparse_generalized_permutation_matrix_with_inverse(np.prod(outshape)+1))
        super(StochasticKeynet, self).__init__(net, inshape, inkey, f_layername_to_keypair, do_output_encryption, backend, verbose)
        

class IdentityTiledKeynet(KeyNet):
    def __init__(self, net, inshape, inkey, tilesize, do_output_encryption=True, verbose=True):    
        f_layername_to_keypair = lambda layername, outshape: (sparse_identity_tiled_matrix_with_inverse(np.prod(outshape)+1, tilesize) if 'relu' not in layername else
                                                              sparse_identity_tiled_matrix_with_inverse(np.prod(outshape)+1, tilesize))
        super(IdentityTiledKeynet, self).__init__(net, inshape, inkey, f_layername_to_keypair, do_output_encryption, backend='tiled', verbose=verbose)

        
class PermutationTiledKeynet(KeyNet):
    def __init__(self, net, inshape, inkey, tilesize, do_output_encryption=True, verbose=True):    
        f_layername_to_keypair = lambda layername, outshape: (sparse_permutation_tiled_matrix_with_inverse(np.prod(outshape)+1, tilesize) if 'relu' not in layername else
                                                              sparse_permutation_tiled_matrix_with_inverse(np.prod(outshape)+1, tilesize))
        super(PermutationTiledKeynet, self).__init__(net, inshape, inkey, f_layername_to_keypair, do_output_encryption, backend='tiled', verbose=verbose)
        
