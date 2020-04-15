import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import torch
import PIL
import uuid
import tempfile
import os
import time
from vipy.util import groupbyasdict
from numpy.lib.stride_tricks import as_strided
import vipy


def find_closest_positive_divisor(a, b):
    """Return non-trivial integer divisor (bh) of (a) closest to (b) in abs(b-bh) such that a % bh == 0"""
    assert a>0 and b>0
    if a<=b:
        return a
    for k in range(0, a-b+1):
        bh = b + k
        if bh>1 and a % bh == 0:
            return bh
        bh = b - k
        if bh>1 and a % bh == 0:
            return bh
    return a  # should never get here, since bh=a is always a solution

    
def matrix_blockview(W, inshape, n):
    """Reorder a sparse matrix W such that:  W*x.flatten() == matrix_blockview(W, x.shape, n)*blockview(x,n).flatten()"""
    d = {v:k for (k,v) in enumerate(blockview(np.array(range(0,np.prod(inshape))).reshape(inshape), n).flatten())}
    data = [v for (k,v) in zip(W.row, W.data) if k in d]    
    row = [d[k] for k in W.row if k in d]
    col = [d[k] for k in W.col if k in d]
    return scipy.sparse.coo_matrix( (data, (row, col)) )


def blockview(A, n):
    """View an np.array A of size (H,W) as an array of shape (H//n, W//n, n, n), such that entry blockview(A,n)[0,0] == A[0:n,0:n]"""
    assert A.ndim == 2
    shape = (A.shape[0]//n, A.shape[1]//n) + (n,n)
    strides = (n*A.strides[0], n*A.strides[1])+ A.strides
    return as_strided(A, shape=shape, strides=strides)


def torch_avgpool2d_in_scipy(x, kernelsize, stride):
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


def torch_conv2d_in_scipy(x,f,b=None,stride=1):
    """Torch equivalent conv2d operation in scipy, with input tensor x, filter weight f and bias b
    x=[BATCH,INCHANNEL,HEIGHT,WIDTH], f=[OUTCHANNEL,INCHANNEL,HEIGHT,WIDTH], b=[OUTCHANNEL,1]"""

    assert(len(x.shape) == 4 and len(f.shape) == 4)
    assert(f.shape[1] == x.shape[1])  # equal inchannels
    assert(f.shape[2]==f.shape[3] and f.shape[2]%2 == 1)  # filter is square, odd
    if b is not None:
        assert(b.shape[0] == f.shape[0])  # weights and bias dimensionality match

    (N,C,U,V) = (x.shape)
    (M,K,P,Q) = (f.shape)
    x_spatialpad = np.pad(x, ( (0,0), (0,0), ((P-1)//2, (P-1)//2), ((Q-1)//2, (Q-1)//2)), mode='constant', constant_values=0)
    y = np.array([scipy.signal.correlate(x_spatialpad[n,:,:,:], f[m,:,:,:], mode='valid')[:,::stride,::stride] + (b[m] if b is not None else 0) for n in range(0,N) for m in range(0,M)])
    return np.reshape(y, (N,M,U//stride,V//stride) )

                
def checkerboard_256x256():
    """Random uint8 rgb color 8x8 checkerboard at 256x256 resolution"""
    img = np.uint8(255*np.random.rand(8,8,3))
    img = np.array(PIL.Image.fromarray(img).resize( (256,256), PIL.Image.NEAREST))
    return img


def numpy_homogenize(x):
    return np.hstack( (x.flatten(), 1) )


def numpy_dehomogenize(x):
    return x.flatten()[0:-1]



    
