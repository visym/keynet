from torch import nn
from numpy.linalg import multi_dot 
from keynet.torch import homogenize_matrix, sparse_toeplitz_conv2d, sparse_toeplitz_avgpool2d, scipy_coo_to_torch_sparse
from keynet.util import sparse_permutation_matrix, sparse_generalized_permutation_block_matrix_with_inverse, sparse_identity_matrix
import torch
import numpy as np
import scipy.sparse
import torch.nn.functional as F
import torch.sparse
from torch import nn, optim
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import warnings
from keynet.block import TiledMatrix
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
        assert backend in set(['torch', 'cupy', 'scipy', 'tiled']), "Unknown backend='%s' - Must be in ['torch', 'cupy', 'scipy', 'tiled']"
        self.backend = backend
        self.W = W        

    def forward(self, x_affine):
        """Keyed sparse linear transformation, using the backend specified for sparse matrix multiplication

           Input: 
             x_affine is (N x C*U*V+1) torch tensor

           Output:
             y=x*W, (N x -1) torch tensor

        """
        if self.backend == 'torch':
            return torch.sparse.mm(x_affine, self.W)   # torch.sparse, left multiply
        elif self.backend == 'cupy':
            try_import(package='cupy', pipname='cupy, cupyx')
            return from_dlpack(self.W.dot(cupy.fromDlpack(to_dlpack(x_affine))).toDlpack())
        elif self.backend == 'scipy':
            return torch.as_tensor(self.W.dot(x_affine.detach().numpy().transpose())).t()  # right multiply 
        elif self.backend == 'tiled':
            raise
        else:
            raise
        
class KeyedConv2d(KeyedLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, backend='torch_sparse'):
        super(KeyedConv2d, self).__init__(backend=backend)

        assert len(kernel_size)==1 or len(kernel_size)==2 and (kernel_size[0] == kernel_size[1]), "Kernel must be square"
        assert len(stride)==1 or len(stride)==2 and (stride[0] == stride[1]), "Strides must be isotropic"
        
        self.stride = stride[0] if len(stride)==2 else stride
        self.kernel_size = kernel_size[0] if len(kernel_size) == 2 else kernel_size        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def key(self, w, b, A, Ainv, inshape):
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
        assert self.backend == 'tiled' or (scipy.sparse.issparse(A) and scipy.sparse.issparse(Ainv)), "(A,Ainv) must be scipy.sparse matrices"
        assert Ainv.shape[0] == Ainv.shape[1], "A*Ainv must be conformal"
        assert len(inshape) == 3, "Inshape must be (C,H,W) for the shape of the tensor at the input to this layer"""
        assert np.prod(inshape)+1 == Ainv.shape[0], "Ainv does not have conformal shape"
        
        self.W = sparse_toeplitz_conv2d(inshape, w.detach().numpy(), bias=b.detach().numpy(), stride=self.stride).transpose()  # Transpose for left multiply
        if self.backend == 'torch':
            A = scipy_coo_to_torch_sparse(A, device='cuda')
            Ainv = scipy_coo_to_torch_sparse(Ainv, device='cuda')
            self.W = scipy_coo_to_torch_sparse(self.W.tocoo(), device='cuda')
            self.W = Ainv.t().dot(self.W.dot(A.t()))  # transpose for left multiply
        elif self.backend == 'cupy':
            A = cupyx.scipy.sparse.csc_matrix(A.tocsc())
            Ainv = cupyx.scipy.sparse.csc_matrix(Ainv.tocsc())            
            self.W = cupyx.scipy.sparse.csc_matrix(self.W.tocsc())  # csc for left mulipl
            self.W = Ainv.t().dot(self.W.dot(A.t()))  # transpose for left multiply                        
        elif self.backend == 'scipy':
            self.W = A.dot(self.W.transpose().dot(Ainv)).tocsr()  # right multiply in scipy
        elif self.backend == 'tiled':
            raise  # FIXME
            assert isinstance(A, TiledMatrix) and isinstance(Ainv, TiledMatrix), "backend='tiled' requires TiledMatrix inputs"
            self.What = TiledMatrix(self.What, A.blocksize())
        else:
            raise ValueError('Undefined backend "%s"' % self.backend)
        return self
    


class KeyedLinear(KeyedLayer):
    def __init__(self, in_features, out_features, backend='scipy'):
        super(KeyedLinear, self).__init__(backend=backend)
        self.in_features = in_features
        self.out_features = out_features

    def key(self, w, b, A, Ainv):
        assert w.shape[0] == self.out_features and w.shape[1] == self.in_features, "Invalid input"
        assert self.backend == 'tiled' or ((A is None or scipy.sparse.issparse(A)) and scipy.sparse.issparse(Ainv)), "(A,Ainv) must be scipy.sparse matrices"        

        if self.backend == 'torch':
            self.W = homogenize_matrix(w, b)  # transposed for left multiply                    
            self.W = torch.matmul(torch.as_tensor(Ainv.todense().transpose()), self.W)
            if A is not None:
                self.W = torch.matmul(self.W, A.todense().transpose())
            self.W = torch.as_tensor(self.W)                
        elif self.backend == 'cupy':
            pass
        elif self.backend == 'scipy':
            self.W = homogenize_matrix(w, b)  # left multiply
            self.W = Ainv.dot(self.W.detach().numpy()).transpose()  # right multiply 
            if A is not None:
                self.W = A.dot(self.W)   # right multiply
        elif self.backend == 'tiled':
            self.W = TiledMatrix(self.W, Ainv.blocksize())            
            self.W.prod(A)
            Ainv.prod(self.W)
            self.W = copy.deepcopy(Ainv)
        return self
        

class KeyedReLU(KeyedLayer):
    def __init__(self, backend='scipy'):
        super(KeyedReLU, self).__init__(backend=backend)

    def key(self, B, Ainv):
        self.W = B*Ainv  # matrix multiply
        if self.backend == 'torch':            
            self.W = scipy_coo_to_torch_sparse(self.W.tocoo(), device='cuda')
        elif self.backend == 'cupy':
            self.W = cupyx.scipy.sparse.csr_matrix(self.What.tocsr())
        elif self.backend == 'scipy':
            pass
        elif self.backend == 'tiled':
            raise
        return self
            
    def forward(self, x_affine):
        return F.relu(super(KeyedReLU, self).forward(x_affine))

    
class KeyedAvgpool2d(KeyedLayer):
    def __init__(self, kernel_size, stride, backend='scipy'):
        super(KeyedAvgpool2d, self).__init__(backend=backend)
        self.kernel_size = kernel_size
        self.stride = stride
        
    def key(self, A, Ainv, inshape):
        self.W = sparse_toeplitz_avgpool2d(inshape, (inshape[0], inshape[0], self.kernel_size, self.kernel_size), self.stride).transpose()  # transpose for left multiply
        if self.backend == 'torch':
            self.W = scipy_coo_to_torch_sparse(self.W.tocoo(), device='cuda')
            A = scipy_coo_to_torch_sparse(A.tocoo(), device='cuda')
            Ainv = scipy_coo_to_torch_sparse(Ainv.tocoo(), device='cuda')            
            self.W = Ainv.t().dot(self.W.dot(A.t()))  # transpose for left multiply                    
        elif self.backend == 'cupy':
            self.W = cupyx.scipy.sparse.csr_matrix(self.W.tocsr())
        elif self.backend == 'scipy': 
            self.W = A.dot(self.W.transpose().dot(Ainv))  # right multiply for scipy
        return self

    
class KeyedSensor(KeyedLayer):
    def __init__(self, inshape, encryptkey, decryptkey, backend='scipy'):
        super(KeyedSensor, self).__init__(backend=backend)        
        self._encryptkey = encryptkey
        self._decryptkey = decryptkey
        self._inshape = inshape
        self._tensor = None

    def __repr__(self):
        return str('<keynet.sensor: height=%d, width=%d, channels=%d>' % (self._inshape[1], self._inshape[2], self._inshape[0]))
    
    def load(self, imgfile):
        im = vipy.image.Image(imgfile).resize(self._inshape[1], self._inshape[2])
        if self._inshape[0] == 1:
            im = im.grey()
        self._tensor = im.torch().contiguous()
        return self

    def tensor(self, x=None):
        if x is None:
            return self._tensor
        else:
            self._tensor = x.clone()
            return self

    def image(self):
        img = dehomogenize(self._tensor).reshape(1, *self._inshape)  if self.isencrypted() else self._tensor  # 1x(C*H*W+1) -> 1xCxHxW
        img = np.squeeze(img.permute(2,3,1,0).numpy())  # 1xCxHxW -> HxWxC
        colorspace = 'float' if img.dtype == np.float32 else None
        colorspace = 'rgb' if img.dtype == np.uint8 and img.shape[2] == 3 else colorspace
        colorspace = 'lum' if img.dtype == np.uint8 and img.shape[2] == 1 else colorspace
        return vipy.image.Image(array=img, colorspace=colorspace)

    def isencrypted(self):
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


class KeyNet(object):
    def __init__(self, net, inshape, layerkey, backend='scipy'):
        # Get network layer shape
        net.eval()
        netshape = keynet.torch.netshape(net, inshape)
        
        # Iterate over named layers and replace with keyed versions
        d_name_to_keyedmodule = OrderedDict()        
        for (k,m) in net.named_children():
            print('[keynet.layers.KeyNet]: Keying "%s"' % k)
            assert k in layerkey, 'Key not found for layer "%s"' % k
            assert k in netshape, 'Layer name not found in net shape for layer "%s"' % k
            assert 'A' in layerkey[k] and 'Ainv' in layerkey[k], 'Keys not specified for layer "%s"' % k
            assert 'inshape' in netshape[k], 'Layer input shape not specified for layer "%s"' % k

            # Replace with keyed versions
            if isinstance(m, nn.Conv2d):
                assert m.padding[0] == m.kernel_size[0]//2 and m.padding[1] == m.kernel_size[1]//2, "Padding is assumed to be equal to (kernelsize-1)/2"
                m_keyed = KeyedConv2d(out_channels=m.out_channels,
                                      in_channels=m.in_channels,
                                      kernel_size=m.kernel_size,
                                      stride=m.stride,
                                      backend=backend)
                d_name_to_keyedmodule[k] = m_keyed.key(m.weight, m.bias, layerkey[k]['A'], layerkey[k]['Ainv'], netshape[k]['inshape'])
            elif isinstance(m, nn.AvgPool2d):
                m_keyed = KeyedAvgpool2d(kernel_size=m.kernel_size,
                                         stride=m.stride,
                                         backend=backend)
                d_name_to_keyedmodule[k] = m_keyed.key(layerkey[k]['A'], layerkey[k]['Ainv'], netshape[k]['inshape'])                
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
                d_name_to_keyedmodule[k_prev].key(bn_weight, bn_bias, layerkey[k_prev]['A'], layerkey[k_prev]['Ainv'], netshape[k_prev]['inshape'])
            elif isinstance(m, nn.Dropout):
                pass  # identity matrix at test time, ignore me
            else:
                raise ValueError('unsupported layer type "%s"' % str(type(m)))

        self._keynet = nn.Sequential(d_name_to_keyedmodule)

    def forward(self, img_cipher, outkey=None):
        y_cipher = self._keynet.forward(img_cipher)
        return dehomogenize(self.decrypt(y_cipher, outkey) if outkey is not None else y_cipher)
    
    def decrypt(self, y_cipher, key):
        return KeyedLayer(backend=self.backend, W=key).forward(y_cipher)


class IdentityKeynet(KeyNet):
    """keynet.layers.IdentityKeynet class.   Testing only"""
    def __init__(self, net, inshape, inkey):

        net.eval()
        netshape = keynet.torch.netshape(net, inshape)
        layerkey = {k:{'Ainv':sparse_identity_matrix(np.prod(v['inshape'])+1),
                       'A':sparse_identity_matrix(np.prod(v['outshape'])+1)}
                    for (k,v) in netshape.items() if 'inshape' in v and 'outshape' in v}

        (i,o) = (netshape['input'], netshape['output'])                
        assert layerkey[i]['Ainv'].shape == inkey.shape, "Invalid inkey"
        layerkey[i]['Ainv'] = inkey  # paired with sensorkey
        super(IdentityKeynet, self).__init__(net, inshape, layerkey)
    

class PermutationKeynet(KeyNet):
    def __init__(self, net, inshape, inkey, do_output_encryption=True):

        net.eval()
        netshape = keynet.torch.netshape(net, inshape)        
        layerkey = {k:{'Ainv':sparse_permutation_matrix(np.prod(v['inshape'])+1),
                       'A':sparse_permutation_matrix(np.prod(v['outshape'])+1)}
                    for (k,v) in netshape.items() if 'inshape' in v and 'outshape' in v}
        
        (i,o) = (netshape['input'], netshape['output'])        
        assert layerkey[i]['Ainv'].shape == inkey.shape, "Invalid inkey"
        layerkey[i]['Ainv'] = inkey  # paired with sensorkey
        layerkey[o]['A'] = None if not do_output_encryption else layerkey[o]['A'] 
        super(PermutationKeynet, self).__init__(net, inshape, layerkey)

        
class StochasticKeynet(KeyNet):
    # Remember that ReLU keys should be permutation 
    pass

class GeneralizedStochasticKeynet(KeyNet):
    # Remember that ReLU keys should be permutation 
    pass


class BlockPermutationKeynet(KeyNet):
    pass


class RBDGSMKeynet(KeyNet):
    """Repeated Block Diagonal Generalized Stochastic Matrix"""
    pass
    
