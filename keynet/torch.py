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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_keynet_parameters(model):
    """Hacky method to count total number of key-net sparse parameters"""
    try:
        return np.sum([getattr(model, layername).What.nnz if hasattr(getattr(model, layername).What, 'nnz') else getattr(model, layername).What.numel() for layername in dir(model) if hasattr(getattr(model, layername), 'What') and getattr(model, layername).What is not None])
    except:
        return np.sum([getattr(model, layername).What.nnz if hasattr(getattr(model, layername).What, 'nnz') else getattr(model, layername).What.size for layername in dir(model) if hasattr(getattr(model, layername), 'What') and getattr(model, layername).What is not None])


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


def sparse_toeplitz_conv2d(inshape, f, bias=None, as_correlation=True, stride=1):
    """ Returns sparse toeplitz matrix (W) in coo format that is equivalent to per-channel pytorch conv2d (spatial correlation) of filter f with a given image with shape=inshape vectorized
        conv2d(img, f) == np.dot(W, img.flatten())
        Example usage: test_keynet.test_sparse_toeplitz_conv2d()
    """

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

    # For every image_row
    for (ku,u) in enumerate(np.arange(0,U,stride)):
        # For every image_column
        for (kv,v) in enumerate(np.arange(0,V,stride)):
            # For every inchannel (transposed)
            for (k_inchannel, c_inchannel) in enumerate(C_range if as_correlation else reversed(C_range)):
                # For every kernel_row (transposed)
                for (i,p) in enumerate(P_range if as_correlation else reversed(P_range)):
                    # For every kernel_col (transposed)
                    for (j,q) in enumerate(Q_range if as_correlation else reversed(Q_range)):
                        # For every outchannel
                        if ((u+p)>=0 and (v+q)>=0 and (u+p)<U and (v+q)<V):
                            c = np.ravel_multi_index( (c_inchannel, u+p, v+q), (C,U,V) )
                            for (k_outchannel, c_outchannel) in enumerate(M_range if as_correlation else reversed(M_range)):
                                data.append(f[k_outchannel,k_inchannel,i,j])
                                row_ind.append( np.ravel_multi_index( (c_outchannel,ku,kv), (M,U//stride,V//stride) ) )
                                col_ind.append( c )

    # Sparse matrix with optional bias using affine augmentation 
    T = coo_matrix((data, (row_ind, col_ind)), shape=(M*(U//stride)*(V//stride), C*U*V))
    if bias is not None:
        lastcol = coo_matrix(np.array([x*np.ones( (U//stride*V//stride), dtype=np.float32) for x in bias]).reshape( (M*(U//stride)*(V//stride),1) ))
    else:
        lastcol = coo_matrix(np.zeros( (T.shape[0],1), dtype=np.float32 ))
    lastrow = np.zeros(T.shape[1]+1, dtype=np.float32);  lastrow[-1]=np.float32(1.0);  
    T = coo_matrix(scipy.sparse.vstack( (scipy.sparse.hstack( (T,lastcol)), coo_matrix(lastrow)) ))
    return T


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

