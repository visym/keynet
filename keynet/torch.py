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
from joblib import Parallel, delayed


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
        if 'input' not in d_modulename_to_shape:
            d_modulename_to_shape['input'] = m.__name  # first
        if 'output' in d_modulename_to_shape:
            del d_modulename_to_shape['output']         # delete and            
        prevlayer = [k for k in d_modulename_to_shape][-1]
        inshape = (input[0].shape[1], input[0].shape[2], input[0].shape[3]) if len(input[0].shape) == 4 else (input[0].shape[1], 1, 1)  # canonicalize to (C,H,W)
        outshape = (output.shape[1], output.shape[2], output.shape[3]) if len(output.shape) == 4 else (output.shape[1], 1, 1)  # canonicalize to (C,H,W)       
        d_modulename_to_shape[m.__name] = {'inshape':inshape,
                                           'outshape':outshape,
                                           'prevlayer':prevlayer}
        d_modulename_to_shape['output'] = m.__name  # reinsert for last        
        delattr(m, '__name')

    hooks = _layer_visitor(_get_shape, net)
    y_dummy = net.forward(x_dummy)
    [h.remove() for h in hooks]
    return d_modulename_to_shape


def homogenize(x):
    """Convert NxCxHxW tensor to Nx(C*H*W+1) tensor where last column is one"""
    (N,C,H,W) = x.shape if len(x.shape)==4 else (1,*x.shape)
    return torch.cat( (x.view(N,C*H*W), torch.ones(N,1, dtype=x.dtype)), dim=1)


def dehomogenize(x):
    """Convert Nx(K+1) tensor to NxK by removing last column"""
    return torch.narrow(x, 1, 0, x.shape[1]-1)


def homogenize_matrix(W, bias=None):
    """Convert matrix W of size (RxC) to (R+1)x(C+1) and return [W^T 0; b^T 1].  
       For x of size (NxK), then this is equivalent to *left* multiplication
       homogenize(x).dot(homogenize_matrix(W,b)) === homogenize(x).dot(W.t()) + b.t(). 
    """
    W_transpose = W.t()
    (R,C) = W_transpose.shape
    b = torch.zeros(1,C) if bias is None else bias.reshape(1,C)
    W_affine = torch.cat( (torch.cat( (W_transpose, b), dim=0), torch.zeros(R+1, 1)), dim=1)
    W_affine[-1,-1] = 1        
    return W_affine
    

def scipy_coo_to_torch_sparse(coo, device='cpu'):
    """https://stackoverflow.com/questions/50665141/converting-a-scipy-coo-matrix-to-pytorch-sparse-tensor"""

    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    if device == 'cpu':
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))
    else:
        with torch.cuda.device(device):
            return torch.sparse.FloatTensor(i, v, torch.Size(shape)).cuda(device)


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
        assert self.is_torch_sparse(A) or self.is_torch_dense(A) or self.is_scipy_sparse(A), "Invalid input"
        self.shape = A.shape
        self._matrix = A
        self.dtype = A.type if self.is_torch(A) else A.dtype
        self.ndim = 2

    def parallel(self, n_processes):
        self._n_processes = n_processes
        return self
    
    def from_torch_dense(self, A):
        assert self.is_torch_dense(A)
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
        assert self.is_torch_dense(x)
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

    
class SparseTiledMatrix(keynet.sparse.SparseTiledMatrix):
    def __init__(self, tilesize=None, coo_matrix=None, blocktoeplitz=None, shape=None, n_processes=1):
        super(SparseTiledMatrix, self).__init__(tilesize, coo_matrix, blocktoeplitz, shape, n_processes)

    def _block(self, B):
        return keynet.torch.SparseMatrix(scipy_coo_to_torch_sparse(B))
    
