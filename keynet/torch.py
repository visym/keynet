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
                assert not hasattr(layer, '__name')  # inplace layers?
                layer.__name = name
                hooklist.append(layer.register_forward_hook(f_forward))
        return hooklist

    def _get_shape(m, input, output):
        if 'input' not in d_modulename_to_shape:
            d_modulename_to_shape['input'] = m.__name  # first
        if 'output' in d_modulename_to_shape:
            del d_modulename_to_shape['output']         # delete and            
        prevlayer = [k for k in d_modulename_to_shape][-1]
        d_modulename_to_shape[m.__name] = {'inshape':tuple(list(input[0].shape)[1:]),
                                           'outshape':tuple(list(output.shape)[1:]),
                                           'prevlayer':prevlayer}
        d_modulename_to_shape['output'] = m.__name  # reinsert for last        
        delattr(m, '__name')

    hooks = _layer_visitor(_get_shape, net)
    y_dummy = net.forward(x_dummy)
    [h.remove() for h in hooks]
    return d_modulename_to_shape


def _parallel_sparse_toeplitz_conv2d(inshape, f, bias=None, as_correlation=True, stride=1, n_processes=1):
    T = Parallel(n_jobs=n_processes)(delayed(sparse_toeplitz_conv2d)(inshape, f, bias, as_correlation, stride, n_processes=1, rowskip=i) for i in range(inshape[1]))
    R = np.sum(T).tocsr().transpose()
    R[-1] = T[0].tocsr().transpose()[-1]  # bias column
    return R.transpose().tocsr()
    
                                
def sparse_toeplitz_conv2d(inshape, f, bias=None, as_correlation=True, stride=1, rowskip=None, n_processes=1):
    """ Returns sparse toeplitz matrix (W) in coo format that is equivalent to per-channel pytorch conv2d (spatial correlation) of filter f with a given image with shape=inshape vectorized
        conv2d(img, f) == np.dot(W, img.flatten()), right multiplied
        Example usage: test_keynet.test_sparse_toeplitz_conv2d()
    """

    if n_processes > 1:
        return _parallel_sparse_toeplitz_conv2d(inshape, f, bias, as_correlation, stride, n_processes=n_processes)

    # Valid shapes
    assert(len(inshape) == 3 and len(f.shape) == 4)  # 3D tensor inshape=(inchannels, height, width), f.shape=(outchannels, kernelheight, kernelwidth, inchannels)
    assert(f.shape[1] == inshape[0])  # equal inchannels
    assert(f.shape[2]==f.shape[3] and f.shape[2]%2 == 1)  # filter is square, odd (FIXME)
    if bias is not None:
        assert(len(bias.shape) == 1 and bias.shape[0] == f.shape[0])  # filter and bias have composable shapes

    # Correlation vs. convolution?
    (C,U,V) = inshape
    (M,C,P,Q) = f.shape
    C_range = range(0,C)
    M_range = range(0,M)
    P_range = range(-((P-1)//2), ((P-1)//2) + 1) if P%2==1 else range(-((P-1)//2), ((P-1)//2) + 2)
    Q_range = range(-((Q-1)//2), ((Q-1)//2) + 1) if P%2==1 else range(-((Q-1)//2), ((Q-1)//2) + 2)
    (data, row_ind, col_ind) = ([],[],[])
    (U_div_stride, V_div_stride) = (U//stride, V//stride)

    # For every image_row
    for (ku,u) in enumerate(np.arange(0,U,stride)):
        if rowskip is not None and u != rowskip:
            continue
        # For every image_column
        for (kv,v) in enumerate(np.arange(0,V,stride)):
            # For every inchannel (transposed)
            for (k_inchannel, c_inchannel) in enumerate(C_range if as_correlation else reversed(C_range)):
                # For every kernel_row (transposed)
                for (i,p) in enumerate(P_range if as_correlation else reversed(P_range)):
                    if not ((u+p)>=0 and (u+p)<U):
                        continue  
                    # For every kernel_col (transposed)
                    for (j,q) in enumerate(Q_range if as_correlation else reversed(Q_range)):
                        # For every outchannel
                        if ((v+q)>=0 and (v+q)<V):
                            #c = np.ravel_multi_index( (c_inchannel, u+p, v+q), (C,U,V) )
                            c = c_inchannel*U*V + (u+p)*V + (v+q)
                            for (k_outchannel, c_outchannel) in enumerate(M_range if as_correlation else reversed(M_range)):
                                data.append(f[k_outchannel,k_inchannel,i,j])
                                #row_ind.append( np.ravel_multi_index( (c_outchannel,ku,kv), (M,U//stride,V//stride) ) )
                                row_ind.append( c_outchannel*(U_div_stride)*(V_div_stride) + ku*(V_div_stride) + kv )
                                col_ind.append( c )

    # Sparse matrix with optional bias using affine augmentation 
    T = scipy.sparse.coo_matrix((data, (row_ind, col_ind)), shape=(M*(U//stride)*(V//stride), C*U*V))
    if bias is not None:
        lastcol = scipy.sparse.coo_matrix(np.array([x*np.ones( (U//stride*V//stride), dtype=np.float32) for x in bias]).reshape( (M*(U//stride)*(V//stride),1) ))
    else:
        lastcol = scipy.sparse.coo_matrix(np.zeros( (T.shape[0],1), dtype=np.float32 ))
    lastrow = np.zeros(T.shape[1]+1, dtype=np.float32);  lastrow[-1]=np.float32(1.0);  
    return scipy.sparse.coo_matrix(scipy.sparse.vstack( (scipy.sparse.hstack( (T,lastcol)), scipy.sparse.coo_matrix(lastrow)) ))


def sparse_toeplitz_avgpool2d(inshape, filtershape, stride):
    (outchannel, inchannel, filtersize, filtersize) = filtershape
    (M,U,V) = (inshape)
    F = np.zeros(filtershape, dtype=np.float32)
    for k in range(0,outchannel):
        F[k,k,:,:] = 1.0 / (filtersize*filtersize)
    return sparse_toeplitz_conv2d(inshape, F, bias=None, stride=stride)
    

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
    def __init__(self, A):
        assert self.is_torch_sparse(A) or self.is_torch_dense(A) or self.is_scipy_sparse(A), "Invalid input"
        self.shape = A.shape
        self._matrix = A
        self.dtype = A.type if self.is_torch(A) else A.dtype
        self.ndim = 2
        
    def from_torch_dense(self, A):
        assert self.is_torch_dense(A)
        return SparseMatrix(A)

    def from_scipy_sparse(self, A):
        assert self.is_scipy_sparse(A)
        return SparseMatrix(A)  

    def matmul(self, A):
        assert isinstance(A, SparseMatrix)
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
    def __init__(self, tilesize=None, coo_matrix=None, blocktoeplitz=None, shape=None):
        super(SparseTiledMatrix, self).__init__(tilesize, coo_matrix, blocktoeplitz, shape)

    def _block(self, B):
        return keynet.torch.SparseMatrix(scipy_coo_to_torch_sparse(B))
    
