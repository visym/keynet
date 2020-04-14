import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import keynet.torch
import keynet.sparse
from keynet.torch import affine_to_linear, linear_to_affine
from keynet.torch import affine_to_linear_matrix
from keynet.sparse import is_scipy_sparse, sparse_toeplitz_avgpool2d, sparse_toeplitz_conv2d, SparseMatrix
import vipy
from keynet.globals import GLOBAL, verbose
import scipy.sparse
from vipy.util import Stopwatch

class KeyedLayer(nn.Module):
    def __init__(self, module, inshape, outshape, A, Ainv, tileshape=None):
        super(KeyedLayer, self).__init__()
        self._layertype = str(type(module))
        self._tileshape = tileshape
        self._inshape = inshape
        self._outshape = outshape
        self._tileshape = tileshape
        
        if isinstance(module, nn.Conv2d):
            assert len(module.kernel_size)==1 or len(module.kernel_size)==2 and (module.kernel_size[0] == module.kernel_size[1]), "Kernel must be square"
            assert len(module.stride)==1 or len(module.stride)==2 and (module.stride[0] == module.stride[1]), "Strides must be isotropic"
            assert len(inshape) == 3, "Inshape must be (C,H,W) for the shape of the tensor at the input to this layer"""
            assert module.padding[0] == module.kernel_size[0]//2 and module.padding[1] == module.kernel_size[1]//2, "Padding is assumed to be equal to (kernelsize-1)/2"            
            stride = module.stride[0] if len(module.stride)==2 else module.stride
            self._repr = 'Conv2d: in_channels=%d, out_channels=%d, kernel_size=%s, stride=%s' % (module.in_channels, module.out_channels, str(module.kernel_size), str(stride))      
            sw = Stopwatch()
            self.W = sparse_toeplitz_conv2d(inshape, module.weight.detach().numpy(), bias=module.bias.detach().numpy(), stride=module.stride[0])            
            print('[KeyedLayer]: sparse_toeplitz_conv2d=%f seconds' % sw.since())
            sw = Stopwatch()
            self.W = A.dot(self.W).dot(Ainv)  # Key!            
            print('[KeyedLayer]: conv2d dot=%f seconds' % sw.since())
            if tileshape is not None:
                self.W = keynet.sparse.Conv2dTiledMatrix(self.W, self._inshape, self._outshape, self._tileshape, bias=True)
            
        elif isinstance(module, nn.ReLU):
            raise ValueError('ReLU layer should be merged with previous layer')

        elif isinstance(module, nn.AvgPool2d):
            assert isinstance(module.kernel_size, int) or len(module.kernel_size)==2 and (module.kernel_size[0] == module.kernel_size[1]), "Kernel must be square"
            assert isinstance(module.stride, int) or len(module.stride)==2 and (module.stride[0] == module.stride[1]), "Strides must be isotropic"
            assert len(inshape) == 3, "Inshape must be (C,H,W) for the shape of the tensor at the input to this layer"""            
            stride = module.stride if isinstance(module.stride, int) else module.stride[0]
            kernel_size = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
            self._repr = 'AvgPool2d: kernel_size=%s, stride=%s' % (str(kernel_size), str(stride))
            sw = Stopwatch()
            self.W = sparse_toeplitz_avgpool2d(inshape, (inshape[0], inshape[0], kernel_size, kernel_size), stride)
            print('[KeyedLayer]: sparse_toeplitz_conv2d=%f seconds' % sw.since())
            sw = Stopwatch()
            self.W = A.dot(self.W).dot(Ainv) if A is not None else self.W.dot(Ainv)  # optional outkey
            print('[KeyedLayer]: avgpool2d dot=%f seconds' % sw.since())
            if tileshape is not None:
                self.W = keynet.sparse.Conv2dTiledMatrix(self.W, self._inshape, self._outshape, self._tileshape, bias=True)
            
        elif isinstance(module, nn.Linear):
            self._repr = 'Linear: in_features=%d, out_features=%d' % (module.in_features, module.out_features)
            self.W = scipy.sparse.coo_matrix(keynet.torch.affine_to_linear_matrix(module.weight, module.bias).detach().numpy()).transpose()  # transposed for right multiply            
            self.W = self.W.dot(Ainv) if A is None else A.dot(self.W).dot(Ainv)  # optional outkey
            
        elif isinstance(module, nn.BatchNorm2d):
            raise ValueError('batchnorm layer should be named "mylayer_bn" for batchnorm of "mylayer" and should come right before "mylayer" to merge keyed layers')
            
        elif isinstance(module, nn.Dropout):
            raise ValueError('dropout layer should be skipped during keying, and removed from final network')
            
        else:
            raise ValueError('unsupported layer type "%s"' % str(type(module)))
        
        if not isinstance(self.W, SparseMatrix) or not isinstance(self.W, keynet.sparse.SparseMatrix):
            self.W = SparseMatrix(self.W)
            
    def extra_repr(self):
        str_shape = 'backend=scipy, shape=%s, nnz=%d>' % (str(self.W.shape), self.nnz())
        return str('<%s, %s>' % (self._repr, str_shape))

    def forward(self, x_affine):
        if verbose():
            print('[keynet.layer]: forward %s' % str(self))
        y = self.W.torchdot(x_affine.t()).t()
        return y if not 'ReLU' in self._layertype else F.relu(y)
        
    def decrypt(self, Ainv, x_affine):
        """Decrypt the output of this layer (x_affine) using supplied key Ainv"""
        if scipy.sparse.issparse(Ainv):
            Ainv = SparseMatrix(Ainv)
        return Ainv.torchdot(x_affine.t()).t()
        
    def nnz(self):
        assert self.W is not None, "Layer not keyed"
        return self.W.nnz()

    def spy(self, mindim=256, showdim=1024, range=None):
        return keynet.sparse.spy(self.W.tocoo(), mindim, showdim, range=range)

