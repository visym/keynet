import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix, coo_matrix
import scipy.signal
from sklearn.preprocessing import normalize
import torch
import PIL
import uuid
import tempfile
import os
from torch import nn
from collections import OrderedDict
import keynet.sparse
import numba


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def netshape(net, inshape):
    """Pass a dummy input into the network and retrieve the input and output shapes of all layers.  Requires named modules, and inplace layers might screw this up"""
    d_modulename_to_shape = OrderedDict()
    x_dummy = torch.rand(1,inshape[0], inshape[1], inshape[2])
    net.eval()
    
    def _layer_visitor(f_forward=None, _net=None):
        """Recursively assign forward hooks for all layers within containers"""
        """Returns list in order of forward visit with {'name'n:, 'hook':h} dictionary"""
        hooklist = []
        for name, layer in _net._modules.items():
            if isinstance(layer, nn.Sequential):
                hooklist = hooklist + _layer_visitor(f_forward, layer)
            else:
                assert not hasattr(layer, '__name')  # inplace layers?  shared layers?
                layer.__name = name
                hooklist.append(layer.register_forward_hook(f_forward))
        return hooklist

    def _get_shape(m, input, output):
        inshape = (input[0].shape[1], input[0].shape[2], input[0].shape[3]) if len(input[0].shape) == 4 else (input[0].shape[1], 1, 1)  # canonicalize to (C,H,W)
        outshape = (output.shape[1], output.shape[2], output.shape[3]) if len(output.shape) == 4 else (output.shape[1], 1, 1)  # canonicalize to (C,H,W)       

        if 'input' not in d_modulename_to_shape:
            d_modulename_to_shape['input'] = {'prevlayer':None, 'nextlayer':m.__name, 'inshape':inshape, 'outshape':outshape}  # first
        if 'output' in d_modulename_to_shape:
            del d_modulename_to_shape['output']         # delete and            

        prevlayer = [k for k in d_modulename_to_shape][-1]  # FIXME: may have more than one prevlayer
        d_modulename_to_shape[m.__name] = {'inshape':inshape,
                                           'outshape':outshape,
                                           'prevlayer':prevlayer}
        d_modulename_to_shape[prevlayer]['nextlayer'] = m.__name
        d_modulename_to_shape[m.__name]['nextlayer'] = None
        d_modulename_to_shape['output'] = {'nextlayer':None, 'prevlayer':m.__name, 'inshape':inshape, 'outshape':outshape}  # reinsert for last
        delattr(m, '__name')

    hooks = _layer_visitor(_get_shape, net)
    y_dummy = net.forward(x_dummy)
    [h.remove() for h in hooks]

    return d_modulename_to_shape


def affine_to_linear(x):
    """Convert NxCxHxW tensor to Nx(C*H*W+1) tensor where last column is one"""
    (N,C,H,W) = x.shape if len(x.shape)==4 else (1,*x.shape)
    return torch.cat( (x.view(N,C*H*W), torch.ones(N,1, dtype=x.dtype)), dim=1)


def linear_to_affine(x, outshape=None):
    """Convert Nx(K+1) tensor to NxK by removing last column (which must be one), and reshaping NxK -> NxCxHxW==outshape"""
    assert len(x.shape) == 2
    if not np.allclose(x[:,-1].detach().numpy(), 1, atol=1E-3):
        raise ValueError('invalid affine vector "%s"' % (str(x)))
    x_affine = torch.narrow(x, 1, 0, x.shape[1]-1)
    return x_affine.reshape(outshape) if outshape is not None else x_affine


def affine_to_linear_matrix(W_affine, bias=None):
    """Convert affine function (Wx+b)^T to linear function (Mx)^T such that M=[W b; 0 1] 
       For x of size (NxK), then this is equivalent to *left* multiplication
    """
    W_affine_transpose = W_affine.t()
    (R,C) = W_affine_transpose.shape
    b = torch.zeros(1,C) if bias is None else bias.reshape(1,C)
    W_linear = torch.cat( (torch.cat( (W_affine_transpose, b), dim=0), torch.zeros(R+1, 1)), dim=1)
    W_linear[-1,-1] = 1        
    return W_linear
    



def torch_sparse_to_scipy_coo(t):
    """https://stackoverflow.com/questions/50665141/converting-a-scipy-coo-matrix-to-pytorch-sparse-tensor"""
    raise
        

def fuse_conv2d_and_bn(conv2d_weight, conv2d_bias, bn_running_mean, bn_running_var, bn_eps, bn_weight, bn_bias):
    """https://discuss.pytorch.org/t/how-to-absorb-batch-norm-layer-weights-into-convolution-layer-weights/16412/4"""
    w = conv2d_weight
    mean = bn_running_mean
    var_sqrt = torch.sqrt(bn_running_var + np.float32(bn_eps))
    beta = bn_weight
    gamma = bn_bias
    out_channels = conv2d_weight.shape[0]
    if conv2d_bias is not None:
        b = conv2d_bias
    else:
        b = mean.new_zeros(mean.shape)
    w = w * (beta / var_sqrt).reshape([out_channels, 1, 1, 1])
    b = (((b - mean)/var_sqrt)*beta) + gamma
    return (w,b)


class SparseMatrix(keynet.sparse.SparseMatrix):
    def __init__(self, A=None, n_processes=1):
        self._n_processes = n_processes        
        assert self.is_torch_sparse(A) or self.is_torch_dense_float32(A) or self.is_scipy_sparse(A), "Invalid input"
        self.shape = A.shape
        self._matrix = A
        self.dtype = A.type if self.is_torch(A) else A.dtype
        self.ndim = 2

    def parallel(self, n_processes):
        self._n_processes = n_processes
        return self
    
    def from_torch_dense(self, A):
        assert self.is_torch_dense_float32(A)
        return SparseMatrix(A)

    def from_scipy_sparse(self, A):
        assert self.is_scipy_sparse(A)
        return SparseMatrix(A)  

    def matmul(self, A):
        assert isinstance(A, keynet.sparse.SparseMatrix)
        A_scipy = scipy.sparse.coo_matrix(A._matrix.detach().numpy()) if self.is_torch(A._matrix) else A._matrix
        M_scipy = scipy.sparse.coo_matrix(self._matrix.detach().numpy()) if self.is_torch(self._matrix) else self._matrix        
        self._matrix = scipy.sparse.csr_matrix.dot(M_scipy.tocsr(), A_scipy.tocsc())  # torch does not support sparse*sparse
        self.shape = self._matrix.shape        
        return self

    def dot(self, x):
        return self.torchdot(x)
    
    def torchdot(self, x):
        assert self.is_torch_dense_float32(x)
        if self.is_scipy_sparse(self._matrix):
            self._matrix = scipy_coo_to_torch_sparse(self._matrix.tocoo())  # lazy conversion
        return torch.sparse.mm(self._matrix, x)

    def nnz(self):
        return self._matrix._nnz() if self.is_torch_sparse(self._matrix) else self._matrix.size

    def transpose(self):
        self._matrix = self._matrix.t()
        self.shape = self._matrix.shape
        return self

    def tocoo(self):
        return torch_sparse_to_scipy_coo(self._matrix)

    
class TiledMatrix(keynet.sparse.TiledMatrix):
    def __init__(self, A, tileshape):
        super(TiledMatrix, self).__init__(A, tileshape)
        
    def _tiletype(self, B):
        return keynet.torch.SparseMatrix(scipy_coo_to_torch_sparse(B))
    
    @staticmethod
    @numba.jit(nopython=True, parallel=True, fastmath=True)
    def _torchdot(x_numpy, tileshape, shape, tiles, blocks):
        (h,w) = (int(tileshape[0]), int(tileshape[1]))
        (H,W) = (int(shape[0]), int(shape[1]))
        y = np.zeros( (H, x_numpy.shape[1]), dtype=np.float32)
        for (i, j, k) in blocks:
            b = tiles[k]            
            for u in range(0, b.shape[0]):
                (ii,jj,v) = (int(b[u,0]), int(b[u,1]), float(b[u,2]))
                y[i+ii, :] += v*x_numpy[j+jj, :]
        return y 
