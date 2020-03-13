import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import keynet.torch
from keynet.torch import homogenize, dehomogenize
from keynet.torch import homogenize_matrix, sparse_toeplitz_conv2d, sparse_toeplitz_avgpool2d, scipy_coo_to_torch_sparse
from keynet.sparse import sparse_permutation_matrix_with_inverse, sparse_permutation_matrix, sparse_generalized_permutation_block_matrix_with_inverse, sparse_identity_matrix
from keynet.sparse import sparse_stochastic_matrix_with_inverse, sparse_generalized_stochastic_matrix_with_inverse, sparse_generalized_permutation_matrix_with_inverse, sparse_identity_matrix_like
from keynet.sparse import sparse_permutation_tiled_matrix_with_inverse, SparseTiledMatrix, sparse_identity_tiled_matrix_with_inverse, sparse_generalized_permutation_tiled_matrix_with_inverse
from keynet.sparse import sparse_generalized_stochastic_tiled_matrix_with_inverse
from keynet.sparse import SparseTiledMatrix, SparseMatrix
import vipy


class KeyedLayer(nn.Module):
    def __init__(self, W=None):
        super(KeyedLayer, self).__init__()        
        self.W = SparseMatrix(W) if W is not None else None

    def forward(self, x_affine):
        assert self.W is not None, "Invalid KeyedLayer - Must call self.key() first"
        return self.W.dot(x_affine.t()).t()  # FIXME
        
    def key(self, W, A, Ainv):
        if W is not None and A is not None and A is not None:
            B = SparseMatrix(A)
            self.W = B.matmul(W).matmul(Ainv)
        elif W is not None and A is not None:
            B = SparseMatrix(A)            
            self.W = B.matmul(W)
        elif W is not None and Ainv is not None: 
            Binv = SparseMatrix(Ainv)
            W = SparseMatrix(W)
            self.W = Binv.transpose().matmul(W.transpose()).transpose()
        elif W is None and Ainv is not None and A is not None:
            B = SparseMatrix(A)                        
            self.W = B.matmul(Ainv)
        else:
            raise ValueError('Invalid key')
        return self

    def nnz(self):
        return self.W.nnz()

    
class KeyedConv2d(KeyedLayer):
    def __init__(self, inshape, in_channels, out_channels, kernel_size, stride):
        super(KeyedConv2d, self).__init__()

        assert len(kernel_size)==1 or len(kernel_size)==2 and (kernel_size[0] == kernel_size[1]), "Kernel must be square"
        assert len(stride)==1 or len(stride)==2 and (stride[0] == stride[1]), "Strides must be isotropic"
        
        self.stride = stride[0] if len(stride)==2 else stride
        self.kernel_size = kernel_size[0] if len(kernel_size) == 2 else kernel_size        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inshape = inshape
        assert len(inshape) == 3, "Inshape must be (C,H,W) for the shape of the tensor at the input to this layer"""
        
    def extra_repr(self):
        str_shape = ', backend=%s, shape=%s>' % (str(type(self.W)), str(self.W.shape)) if self.W is not None else '>'
        return str('<KeyedConv2d: in_channels=%d, out_channels=%d, kernel_size=%s, stride=%s%s' (self.in_channels, self.out_channels, str(self.kernel_size), str(self.stride), str_shape))
        
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
        W = sparse_toeplitz_conv2d(self.inshape, w.detach().numpy(), bias=b.detach().numpy(), stride=self.stride)   # Expensive
        return super(KeyedConv2d, self).key(W, A, Ainv)

        
class KeyedLinear(KeyedLayer):
    def __init__(self, in_features, out_features):
        super(KeyedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def extra_repr(self):
        str_shape = ', backend=%s, shape=%s>' % (str(type(self.W)), str(self.W.shape)) if self.W is not None else '>'        
        return str('<KeyedLinear: in_features=%d, out_features=%d%s>' (self.in_features, self.out_features, str_shape))
        
    def key(self, w, b, A, Ainv):
        W = homogenize_matrix(w, b).t()  # transposed for right multiply
        return super(KeyedLinear, self).key(W, A, Ainv)

    
class KeyedReLU(KeyedLayer):
    def __init__(self):
        super(KeyedReLU, self).__init__()

    def extra_repr(self):
        str_shape = ': backend=%s, shape=%s>' % (str(type(self.W)), str(self.W.shape)) if self.W is not None else '>'        
        return str('<KeyedReLU%s' % (str(self.backend), str_shape))
        
    def key(self, P, Ainv):
        return super(KeyedReLU, self).key(None, P, Ainv)
        
    def forward(self, x_affine):
        return F.relu(super(KeyedReLU, self).forward(x_affine))

    
class KeyedAvgpool2d(KeyedLayer):
    def __init__(self, inshape, kernel_size, stride):
        super(KeyedAvgpool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.inshape = inshape

    def extra_repr(self):
        str_shape = ', backend=%s, shape=%s>' % (str(type(self.W)), str(self.W.shape)) if self.W is not None else '>'        
        return str('<KeyedAvgpool2d: kernel_size=%s, stride=%s%s>' (str(self.kernel_size), str(self.stride), str_shape))
    
    def key(self, A, Ainv):
        W = sparse_toeplitz_avgpool2d(self.inshape, (self.inshape[0], self.inshape[0], self.kernel_size, self.kernel_size), self.stride)  # Expensive
        return super(KeyedAvgpool2d, self).key(W, A, Ainv)


