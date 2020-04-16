import numpy as np
import scipy.linalg
import PIL
import copy
import torch 
from torch import nn
import torch.nn.functional as F
import keynet.sparse
from keynet.sparse import sparse_permutation_matrix, sparse_identity_matrix, sparse_identity_matrix_like
from keynet.sparse import sparse_uniform_random_diagonal_matrix, diagonal_affine_to_linear
from keynet.sparse import sparse_toeplitz_conv2d, sparse_toeplitz_avgpool2d, sparse_channelorder_to_blockorder_matrix
from keynet.util import torch_avgpool2d_in_scipy, torch_conv2d_in_scipy
from keynet.dense import uniform_random_diagonal_matrix, random_positive_definite_matrix
from keynet.torch import affine_to_linear, linear_to_affine, affine_to_linear_matrix
import keynet.util
import keynet.mnist
import keynet.cifar10
import keynet.torch
import keynet.system
import keynet.vgg
import vipy
from vipy.util import Stopwatch


def test_diagonal_affine_to_linear():
    n = 2048
    A = sparse_uniform_random_diagonal_matrix(n, 1.0)
    (P, Pinv) = diagonal_affine_to_linear(A, withinverse=True)
    assert np.allclose(np.eye(n+1), P.dot(Pinv).todense(), atol=1E-5)
    
    (P, Pinv) = diagonal_affine_to_linear(A, bias=np.random.rand(n,1), withinverse=True)
    assert np.allclose(np.eye(n+1), P.dot(Pinv).todense(), atol=1E-5)    

    print('[test_diagonal_affine_to_linear]:  PASSED')
    

def test_torch_homogenize():
    (N,C,U,V) = (2,2,3,3)
    x = torch.tensor(np.random.rand( N,C,U,V ).astype(np.float32))    

    x_affine = keynet.torch.affine_to_linear(x)
    x_deaffine = keynet.torch.linear_to_affine(x_affine, (N,C,U,V))
    assert(np.allclose(x, x_deaffine))
    print('[test_torch_homogenize]:  Affine augmentation (round-trip)  PASSED')    
    
    W = torch.rand(C*U*V, C*U*V)
    b = torch.rand(C*U*V)
    Wh = affine_to_linear_matrix(W, b)
    assert np.allclose(keynet.torch.linear_to_affine(torch.matmul(keynet.torch.affine_to_linear(x), Wh)).numpy(), (torch.matmul(x.view(N,-1), W.t())+b).numpy())
    print('[test_torch_homogenize]:  Affine augmentation matrix   PASSED')        

    
def test_blockview():
    b = 4
    stride = 1
    (N,C,U,V) = (1,1,8,8)
    (M,C,P,Q) = (1,1,3,3)

    img = np.random.rand(N,C,U,V)
    f = np.random.randn(M,C,P,Q)
    assert(U%2==0 and V%2==0 and (stride==1 or stride==2) and P%2==1 and Q%2==1)  # ODD filters, EVEN tensors

    # Toeplitz matrix block
    A = sparse_toeplitz_conv2d( (C,U,V), f, as_correlation=True, stride=stride)
    W = keynet.util.matrix_blockview(A.tocoo(), (U,V), b)
    W_blk1 = W.todense()[0:b*b, 0:b*b]
    W_blk2 = W.todense()[-b*b:, -b*b:]    
    assert np.allclose(W_blk1, W_blk2)

    W_blk3 = sparse_toeplitz_conv2d( (1,4,4), f, as_correlation=True, stride=stride).todense()
    assert np.allclose(W_blk1, W_blk3[0:b*b, 0:b*b])
    
    # Image blocks
    imgblock = keynet.util.blockview(np.squeeze(img), b)
    blk = imgblock[0,0]
    assert blk.shape == (b,b)

    # Compare
    y = torch_conv2d_in_scipy(blk.reshape(1,1,b,b), f)
    yh = W_blk1.dot(blk.flatten())
    assert np.allclose(y.flatten(), yh.flatten())
    print('[test_blockview]:  PASSED')
    
    W = np.random.rand(8,8)
    a = np.random.rand(2,2)

    A = scipy.linalg.block_diag(*[a for k in range(0,4)])
    b = np.random.rand(2,2)
    B = scipy.linalg.block_diag(*[b for k in range(0,4)])

    C = np.dot(np.dot(A,W), B)

    C_blk = keynet.util.blockview(C,2)
    W_blk = keynet.util.blockview(W,2)    
    for i in range(0,4):
        for j in range(0,4):
            assert np.allclose( C_blk[i,j], np.dot(np.dot(a,W_blk[i,j]), b))
    print('[test_blockview_multiply]:  PASSED')


def show_sparse_blockkey(n=32):
    im = vipy.image.Image('./demo/owl.jpg').resize(256,256).grey()
    img = im.numpy()
    x = img.flatten()
    # b = scipy.sparse.coo_matrix(keynet.util.random_dense_permutation_matrix(n).astype(np.float32))
    b = scipy.sparse.coo_matrix(keynet.util.random_doubly_stochastic_matrix(n, 8).astype(np.float32))
    d = scipy.sparse.coo_matrix(keynet.util.uniform_random_diagonal_matrix(n))
    b = scipy.sparse.coo_matrix(b.dot(d).astype(np.float32))
    (rows,cols,data) = ([],[],[])
    for k in range(0, 256*256, n):
        # b = scipy.sparse.coo_matrix(keynet.util.random_dense_permutation_matrix(n).astype(np.float32))        
        for i,j,v in zip(b.row, b.col, b.data):
            rows.append(i+k)
            cols.append(j+k)
            data.append(v)
    B = scipy.sparse.coo_matrix( (data, (rows, cols)) ).tocsr()
    img_keyed = B.dot(x).reshape(256,256)
    im_keyed = vipy.image.Image(array=img_keyed, colorspace='grey')
    return im.array(np.hstack( (im.array(), im_keyed.array()) )).show()


def test_sparse_tiled_matrix():
    keynet.globals.verbose(False)

    (U,V) = (474, 78)
    W = scipy.sparse.coo_matrix(np.random.rand(U, V))
    T = keynet.sparse.TiledMatrix(W, tileshape=(14,14))
    assert np.allclose(W.todense().astype(np.float32), T.tocoo().todense(), atol=1E-5)

    (U,V) = (8,8)
    W = sparse_toeplitz_conv2d( (1,U,V), np.random.rand(1,1,3,3) )    
    T = keynet.sparse.TiledMatrix(W, tileshape=(4,4))
    assert np.allclose(W.todense().astype(np.float32), T.tocoo().todense(), atol=1E-5)

    (U,V) = (27,26)
    W = sparse_toeplitz_conv2d( (1,U,V), np.random.rand(1,1,3,3) )    
    T = keynet.sparse.Conv2dTiledMatrix(W, inshape=(1,U,V), outshape=(1,U,V), tileshape=(3,3), bias=False)
    assert np.allclose(W.todense().astype(np.float32), T.tocoo().todense(), atol=1E-5)
    
    (U,V) = (8,8)
    W = sparse_toeplitz_conv2d( (1,U,V), np.random.rand(1,1,3,3) )    
    T = keynet.sparse.Conv2dTiledMatrix(W, (1,U,V), (1,U,V), tileshape=(4,4), bias=False)
    assert np.allclose(W.todense().astype(np.float32), T.tocoo().todense(), atol=1E-5)
    
    (U,V) = (17,32)
    im = vipy.image.Image('./demo/owl.jpg').resize(U,V).grey()
    img = im.tonumpy()
    x = torch.tensor(img.reshape(1,U,V))

    x_torch = x.reshape(1, U*V)
    x_numpy = x_torch.numpy()
    W_right = sparse_toeplitz_conv2d( (1,U,V), np.random.rand(1,1,3,3), bias=None)
    W_right_dense = W_right.todense()    
    
    T_right = keynet.sparse.Conv2dTiledMatrix(W_right, inshape=(1,U,V), outshape=(1,U,V), tileshape=(4, 4), bias=False)
    yh = T_right.torchdot(x_torch.t())  # right multiply
    y = W_right_dense.dot(x_numpy.transpose())
    assert np.allclose(y.flatten(), yh.flatten(), atol=1E-5)    

    x_torch = keynet.torch.affine_to_linear(x)
    x_numpy = x_torch.numpy()
    W_right = sparse_toeplitz_conv2d( (1,U,V), np.random.rand(1,1,3,3), bias=np.random.rand(1))
    W_right_dense = W_right.todense()    
    
    T_right = keynet.sparse.Conv2dTiledMatrix(W_right, inshape=(1,U,V), outshape=(1,U,V), tileshape=(16,16), bias=True)
    assert np.allclose(W_right_dense.flatten(), T_right.tocoo().todense().flatten(), atol=1E-5)           
    yh = T_right.torchdot(x_torch.t())  # right multiply
    y = W_right_dense.dot(x_numpy.transpose())
    assert np.allclose(y.flatten(), yh.flatten(), atol=1E-5)    
    
    x1 = torch.tensor(np.random.rand(10,1).astype(np.float32))
    T1 = keynet.sparse.DiagonalTiledMatrix(np.random.rand(3,3).astype(np.float32), shape=(10,10))
    W1 = T1.tocoo().todense()
    assert np.allclose(W1.dot(x1).flatten(), T1.torchdot(x1).flatten(), atol=1E-5)
    
    T_right = keynet.sparse.Conv2dTiledMatrix(W_right, inshape=(1,U,V), outshape=(1,U,V), tileshape=(U*4, U*4), bias=True)    
    assert np.allclose(T_right.tocoo().todense().flatten(), W_right.todense().flatten(), atol=1E-5)
    
    print('[test_block_tiled]:  PASSED')
    

def show_channelorder_to_blockorder():
    im = vipy.image.Image('./demo/owl.jpg').greyscale().resize(32, 32, interp='nearest')
    img = im.array()
    img = np.expand_dims(img, 0)  # HxWxC -> 1xCxHxW
    img = np.expand_dims(img, 0)  # HxWxC -> 1xCxHxW    
    x_torch = keynet.torch.affine_to_linear(torch.as_tensor(img))
    x_numpy = np.array(x_torch).reshape(32*32+1, 1)
    
    (A,Ainv) = sparse_block_permutation_identity_tiled_matrix_with_inverse(np.prod((1,32,32))+1, 16*16)
    C = sparse_channelorder_to_blockorder_matrix((1,32,32), 16, True)

    assert np.allclose(C.dot(C.transpose()).todense(), np.eye(32*32+1))

    y_numpy = C.transpose().dot(A.dot(C.dot(x_numpy)))
    y_torch = torch.as_tensor(y_numpy).reshape(1, 32*32+1)
    y = np.array(keynet.torch.linear_to_affine(y_torch))
    
    return vipy.image.Image(array=y.reshape(32,32), colorspace='float').resize(256,256,interp='nearest').show()
    #return vipy.image.Image(array=np.array(A.tocoo().todense()), colorspace='float').show()


def test_sparse_toeplitz_conv2d():
    stride = 2
    (N,C,U,V) = (2,1,8,16)
    (M,C,P,Q) = (4,1,3,3)
    img = np.random.rand(N,C,U,V)
    f = np.random.randn(M,C,P,Q)
    b = np.random.randn(M).flatten()
    assert(U%2==0 and V%2==0 and (stride==1 or stride==2) and P%2==1 and Q%2==1)  # ODD filters, EVEN tensors

    # Toeplitz matrix:  affine augmentation
    T = sparse_toeplitz_conv2d( (C,U,V), f, b, as_correlation=True, stride=stride)
    yh = T.dot(np.hstack((img.reshape(N,C*U*V), np.ones( (N,1) ))).transpose()).transpose()[:,:-1] 
    yh = yh.reshape(N,M,U//stride,V//stride)
    T_multiproc = sparse_toeplitz_conv2d( (C,U,V), f, b, as_correlation=True, stride=stride)
    assert np.allclose(T.todense().flatten(), T_multiproc.todense().flatten(), atol=1E-5)
    
    # Spatial convolution:  torch replicated in scipy
    y_scipy = torch_conv2d_in_scipy(img, f, b, stride=stride)
    assert(np.allclose(y_scipy, yh, atol=1E-5))    
    print('[test_sparse_toeplitz_conv2d]:  Correlation (scipy vs. toeplitz): passed')    

    # Torch spatial correlation: reshape torch to be tensor sized [BATCH x CHANNEL x HEIGHT x WIDTH]
    # Padding required to allow for valid convolution to be the same size as input
    y_torch = F.conv2d(torch.tensor(img), torch.tensor(f), bias=torch.tensor(b), padding=((P-1)//2, (Q-1)//2), stride=stride)
    assert(np.allclose(y_torch, yh, atol=1E-5))
    print('[test_sparse_toeplitz_conv2d]:  Correlation (torch vs. toeplitz): passed')
    

def test_sparse_toeplitz_avgpool2d():
    np.random.seed(0)
    (N,C,U,V) = (1,1,8,10)
    (kernelsize, stride) = (3,2)
    (P,Q) = (kernelsize,kernelsize)
    img = np.random.rand(N,C,U,V)
    assert(U%2==0 and V%2==0 and kernelsize%2==1 and stride<=2)

    # Toeplitz matrix
    T = sparse_toeplitz_avgpool2d( (C,U,V), (C,C,kernelsize,kernelsize), stride=stride)
    yh = T.dot(np.hstack((img.reshape(N,C*U*V), np.ones( (N,1) ))).transpose()).transpose()[:,:-1]
    yh = yh.reshape(N,C,U//stride,V//stride)

    # Average pooling
    y_scipy = torch_avgpool2d_in_scipy(img, kernelsize, stride)
    assert(np.allclose(y_scipy, yh))
    print('[test_sparse_toeplitz_avgpool2d]:  Average pool 2D (scipy vs. toeplitz)  PASSED')    

    # Torch avgpool
    y_torch = F.avg_pool2d(torch.tensor(img), kernelsize, stride=stride, padding=((P-1)//2, (Q-1)//2))
    assert(np.allclose(y_torch,yh))
    print('[test_sparse_toeplitz_avgpool2d]: Average pool 2D (torch vs. toeplitz)  PASSED')


def _test_roundoff(m=512, n=1000):
    """Experiment with accumulated float32 rounding errors for deeper networks"""
    x = np.random.randn(m,1).astype(np.float32)
    xh = x
    for j in range(0,n):
        A = random_dense_positive_definite_matrix(m).astype(np.float32)
        xh = np.dot(A,xh)
        Ainv = np.linalg.inv(A)
        xh = np.dot(Ainv,xh)
        if j % 10 == 0:
            print('[test_roundoff]: m=%d, n=%d, j=%d, |x-xh|/|x|=%1.13f, |x-xh]=%1.13f' % (m,n,j, np.max(np.abs(x-xh)/np.abs(x)), np.max(np.abs(x-xh))))


    
def _test_semantic_security():
    """Confirm that number of non-zeros is increased in Toeplitz matrix after keying"""
    W = sparse_toeplitz_conv2d( (1,8,8), np.ones( (1,1,3,3) ))
    (B,Binv) = sparse_generalized_stochastic_block_matrix_with_inverse(65,1)
    (A,Ainv) = sparse_generalized_permutation_block_matrix_with_inverse(65,2)
    What = B*W*Ainv
    print([what.nnz - w.nnz for (what, w) in zip(What.tocsr(), W.tocsr())])
    assert(np.all([what.nnz > w.nnz for (what, w) in zip(What.tocsr(), W.tocsr())]))
    print(What.nnz)
    What = W*Ainv
    assert(What.nnz > W.nnz)
    print('[test_semantic_security]:  PASSED')



def test_sparse_matrix():

    W_numpy = np.random.rand(3,3).astype(np.float32)
    x_numpy = np.random.rand(3,1).astype(np.float32)
    y = W_numpy.dot(x_numpy)
    
    W_torch = torch.as_tensor(W_numpy)
    x_torch = torch.as_tensor(x_numpy)

    W_torch_sparse = keynet.sparse.scipy_coo_to_torch_sparse(scipy.sparse.coo_matrix(W_numpy))    
    W_scipy = scipy.sparse.coo_matrix(W_numpy)
    
    A = keynet.sparse.SparseMatrix(W_numpy)
    assert np.allclose(A.dot(x_numpy), y, atol=1E-5)
    assert np.allclose(A.torchdot(x_torch).numpy(), y, atol=1E-5)

    A = keynet.sparse.SparseMatrix(W_scipy)
    A = A.from_scipy_sparse(W_scipy)
    assert np.allclose(A.dot(x_numpy), y, atol=1E-5)
    assert np.allclose(A.torchdot(x_torch).numpy(), y, atol=1E-5)
    
    A = keynet.torch.SparseMatrix(W_torch)
    assert np.allclose(A.dot(x_torch).numpy(), y, atol=1E-5)
    assert np.allclose(A.torchdot(x_torch).numpy(), y, atol=1E-5)

    print('[test_sparse_matrix]: PASSED')


if __name__ == '__main__':
    test_diagonal_affine_to_linear()
    test_torch_homogenize()
    test_sparse_toeplitz_conv2d()
    test_sparse_toeplitz_avgpool2d()
    test_blockview()
    test_sparse_matrix()
    test_sparse_tiled_matrix()        
