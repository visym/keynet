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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_keynet_parameters(model):
    try:
        return np.sum([getattr(model, layername).What.nnz if hasattr(getattr(model, layername).What, 'nnz') else getattr(model, layername).What.numel() for layername in dir(model) if hasattr(getattr(model, layername), 'What') and getattr(model, layername).What is not None])
    except:
        return np.sum([getattr(model, layername).What.nnz if hasattr(getattr(model, layername).What, 'nnz') else getattr(model, layername).What.size for layername in dir(model) if hasattr(getattr(model, layername), 'What') and getattr(model, layername).What is not None])

def conv2d_in_scipy(x,f,b,stride=1):
    """Torch equivalent conv2d operation in scipy, with input tensor x, filter weight f and bias b"""
    """x=[BATCH,INCHANNEL,HEIGHT,WIDTH], f=[OUTCHANNEL,INCHANNEL,HEIGHT,WIDTH], b=[OUTCHANNEL,1]"""

    assert(len(x.shape) == 4 and len(f.shape) == 4)
    assert(f.shape[1] == x.shape[1])  # equal inchannels
    assert(f.shape[2]==f.shape[3] and f.shape[1]%2 == 1)  # filter is square, odd
    assert(b.shape[0] == f.shape[0])  # weights and bias dimensionality match

    (N,C,U,V) = (x.shape)
    (M,K,P,Q) = (f.shape)
    x_spatialpad = np.pad(x, ( (0,0), (0,0), ((P-1)//2, (P-1)//2), ((Q-1)//2, (Q-1)//2)), mode='constant', constant_values=0)
    y = np.array([scipy.signal.correlate(x_spatialpad[n,:,:,:], f[m,:,:,:], mode='valid')[:,::stride,::stride] + b[m] for n in range(0,N) for m in range(0,M)])
    return np.reshape(y, (N,M,U//stride,V//stride) )


def sparse_toeplitz_conv2d(inshape, f, bias=None, as_correlation=True, stride=1):
    # Returns sparse toeplitz matrix (W) in coo format that is equivalent to per-channel pytorch conv2d (spatial correlation)
    # see also: test_keynet.test_sparse_toeplitz_conv2d()

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


def avgpool2d_in_scipy(x, kernelsize, stride):
    """Torch equivalent avgpool2d operation in scipy, with input tensor x"""
    """x=[BATCH,INCHANNEL,HEIGHT,WIDTH]"""
    """https://pytorch.org/docs/stable/nn.html#torch.nn.AvgPool2d"""

    assert(len(x.shape) == 4 and kernelsize%2==1)  # odd kernel size (FIXME)

    (N,C,U,V) = (x.shape)
    (P,Q) = (kernelsize,kernelsize)
    F = (1.0 / (kernelsize*kernelsize))*np.ones( (kernelsize,kernelsize))
    (rightpad, leftpad) = ((P-1)//2, (Q-1)//2)
    x_spatialpad = np.pad(x, ( (0,0), (0,0), (leftpad, rightpad), (leftpad,rightpad)), mode='constant', constant_values=0)
    y = np.array([scipy.signal.correlate(x_spatialpad[n,m,:,:], F, mode='valid')[::stride,::stride] for n in range(0,N) for m in range(0,C)])
    return np.reshape(y, (N,C,(U//stride),(V//stride)) )


def sparse_toeplitz_avgpool2d(inshape, filtershape, stride):
    (outchannel, inchannel, filtersize, filtersize) = filtershape
    (M,U,V) = (inshape)
    F = np.zeros(filtershape, dtype=np.float32)
    for k in range(0,outchannel):
        F[k,k,:,:] = 1.0 / (filtersize*filtersize)
    return sparse_toeplitz_conv2d(inshape, F, bias=None, stride=stride)
    

def affine_augmentation_tensor(x):
    if len(x.shape) == 4:
        (N,C,U,V) = x.shape
    else:
        (C,U,V) = x.shape
        N = 1
    return torch.t(torch.cat( (x.view(N,C*U*V), torch.ones(N,1)), dim=1)).contiguous()

def affine_deaugmentation_tensor(x):
    (K,N) = x.shape
    return torch.t(torch.narrow(x, 0, 0, K-1)).contiguous()

def affine_augmentation_matrix(W,bias=None):
    (M,N) = W.shape
    b = torch.zeros(M,0) if bias is None else bias.reshape(M,1)
    W_affine = torch.cat( (torch.cat( (W,b), dim=1), torch.zeros(1,N+1)), dim=0)
    W_affine[-1,-1] = 1        
    return W_affine
    

def scipy_coo_to_torch_sparse(coo, device='cpu'):
    """https://stackoverflow.com/questions/50665141/converting-a-scipy-coo-matrix-to-pytorch-sparse-tensor"""

    with torch.cuda.device(device):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape))
    

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

