from numpy.linalg import multi_dot 
import numpy as np
import scipy.linalg
import PIL
import copy
import torch 
from torch import nn
import torch.nn.functional as F
import keynet.sparse
from keynet.sparse import sparse_permutation_matrix_with_inverse, sparse_permutation_matrix, sparse_generalized_permutation_block_matrix_with_inverse
from keynet.sparse import sparse_generalized_stochastic_block_matrix_with_inverse, sparse_identity_matrix
from keynet.torch import homogenize, dehomogenize, homogenize_matrix
from keynet.torch import sparse_toeplitz_conv2d
from keynet.torch import sparse_toeplitz_avgpool2d
from keynet.util import torch_avgpool2d_in_scipy, torch_conv2d_in_scipy
from keynet.dense import uniform_random_diagonal_matrix, random_positive_definite_matrix
import keynet.util
import keynet.blockpermute
import keynet.mnist
import keynet.cifar10
import keynet.torch
import keynet.fiberbundle
import vipy
from vipy.util import Stopwatch
import keynet.system


def test_torch_homogenize():
    (N,C,U,V) = (2,2,3,3)
    x = torch.tensor(np.random.rand( N,C,U,V ).astype(np.float32))    

    x_affine = homogenize(x)
    x_deaffine = dehomogenize(x_affine).reshape( N,C,U,V )
    assert(np.allclose(x, x_deaffine))
    print('[test_torch_homogenize]:  Affine augmentation (round-trip)  PASSED')    
    
    W = torch.rand(C*U*V, C*U*V)
    b = torch.rand(C*U*V)
    Wh = homogenize_matrix(W,b)
    assert np.allclose(dehomogenize(torch.matmul(homogenize(x), Wh)).numpy(), (torch.matmul(x.view(N,-1), W.t())+b).numpy())
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
    W = keynet.util.matrix_blockview(A, (U,V), b)
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
    im = vipy.image.Image('owl.jpg').resize(256,256).grey()
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
    (U,V) = (8,8)
    W = keynet.torch.sparse_toeplitz_conv2d( (1,U,V), np.random.rand(1,1,3,3) )    
    T = keynet.sparse.SparseTiledMatrix(coo_matrix=W, tilesize=4)
    assert np.allclose(W.todense().astype(np.float32), T.tocoo().todense(), atol=1E-5)
    
    (U,V) = (17,32)
    im = vipy.image.Image('owl.jpg').resize(U,V).grey()
    img = im.tonumpy()
    x = torch.tensor(img.reshape(1,U,V))
    x_torch = homogenize(x)
    x_numpy = x_torch.numpy()
    W_right = keynet.torch.sparse_toeplitz_conv2d( (1,U,V), np.random.rand(1,1,3,3) )
    W_right_dense = W_right.todense()    

    T_right = keynet.sparse.SparseTiledMatrix(coo_matrix=W_right, tilesize=U*4)
    yh = T_right.torchdot(x_torch.t())  # right multiply
    y = W_right_dense.dot(x_numpy.transpose())
    assert np.allclose(y.flatten(), yh.flatten(), atol=1E-5)    

    x1 = torch.tensor(np.random.rand(10,1).astype(np.float32))
    T1 = keynet.sparse.SparseTiledMatrix(shape=(10,10), tile_to_blkdiag=np.random.rand(3,3))
    W1 = T1.tocoo().todense()
    assert np.allclose(W1.dot(x1).flatten(), T1.torchdot(x1).flatten(), atol=1E-5)
    
    T_right = keynet.sparse.SparseTiledMatrix(coo_matrix=W_right, tilesize=U*4)    
    assert np.allclose(T_right.tocoo().todense().flatten(), W_right.todense().flatten(), atol=1E-5)
    
    T1 = keynet.sparse.SparseTiledMatrix(tilesize=3, coo_matrix=scipy.sparse.coo_matrix(np.random.rand(9,10)))
    T2 = keynet.sparse.SparseTiledMatrix(tilesize=3, coo_matrix=scipy.sparse.coo_matrix(np.random.rand(10,11)))    
    W1 = T1.tocoo().todense()
    W2 = T2.tocoo().todense()
    assert np.allclose(W1.dot(W2).flatten(), T1.matmul(T2).tocoo().todense().flatten(), atol=1E-5)
        
    W2_right = keynet.torch.sparse_toeplitz_conv2d( (1,U,V), np.random.rand(1,1,3,3) )
    T2_right = keynet.sparse.SparseTiledMatrix(coo_matrix=W2_right, tilesize=T_right.tilesize())    
    T_tight = T_right.matmul(T2_right)
    y = W_right_dense.dot(W2_right.todense()).dot(x_numpy.transpose())
    yh = T_right.torchdot(x_torch.t()).numpy()
    assert np.allclose(y.flatten(), yh.flatten(), atol=1E-5)
    
    T1 = keynet.sparse.SparseTiledMatrix(shape=(10,10), tile_to_blkdiag=np.random.rand(3,3))
    T2 = keynet.sparse.SparseTiledMatrix(shape=(10,10), tile_to_blkdiag=np.random.rand(3,3))    
    W1 = T1.tocoo().todense()
    W2 = T2.tocoo().todense()
    assert np.allclose(W1.dot(W2).flatten(), T1.matmul(T2).tocoo().todense().flatten(), atol=1E-5)

    T3 = keynet.sparse.SparseTiledMatrix(coo_matrix=T1.tocoo(), tilesize=3)
    assert len(T3.tiles()) == 2  

    T2 = keynet.sparse.SparseTiledMatrix(coo_matrix=W2_right.astype(np.float32), tilesize=3)
    T1 = keynet.sparse.SparseTiledMatrix(shape=(T2.shape), tile_to_blkdiag=np.random.rand(3,3).astype(np.float32))
    T3 = keynet.sparse.SparseTiledMatrix(shape=(T2.shape), tile_to_blkdiag=np.random.rand(3,3).astype(np.float32))
    assert T1.shape == T1.tocoo().shape
    assert T2.shape == T2.tocoo().shape        
    assert T3.shape == T3.tocoo().shape

    T4 = T1.clone().matmul(T2).matmul(T3)
    assert np.allclose(T4.tocoo().todense().flatten(), T1.tocoo().todense().dot(T2.tocoo().todense().dot(T3.tocoo().todense())).flatten(), atol=1E-5)
    
    print('[test_block_tiled]:  PASSED')
    

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


def show_fiberbundle_simulation():
    """Save a temp image containing the fiber bundle simulation for the image 'owl.jpg'"""
    img_color = np.array(PIL.Image.open('owl.jpg').resize( (512,512) ))
    img_sim = keynet.fiberbundle.simulation(img_color, h_xtalk=0.05, v_xtalk=0.05, fiber_core_x=16, fiber_core_y=16, do_camera_noise=True)
    return vipy.image.Image(array=np.uint8(img_sim), colorspace='rgb').show().savetmp()


def show_fiberbundle_simulation_32x32():
    """Save a 32x32 CIFAR-10 like temp image containing the fiber bundle simulation for the image 'owl.jpg'"""
    img_color_large = np.array(PIL.Image.open('owl.jpg').resize( (512,512), PIL.Image.BICUBIC ))  
    img_sim_large = keynet.fiberbundle.simulation(img_color_large, h_xtalk=0.05, v_xtalk=0.05, fiber_core_x=16, fiber_core_y=16, do_camera_noise=False)
    img_sim = np.array(PIL.Image.fromarray(np.uint8(img_sim_large)).resize( (32,32), PIL.Image.BICUBIC ).resize( (512,512), PIL.Image.NEAREST ))
    return vipy.image.Image(array=np.uint8(img_sim), colorspace='rgb').show().savetmp()

    
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


def test_keynet_constructor():
    inshape = (1,28,28)
    x = torch.randn(1, *inshape)
    net = keynet.mnist.LeNet_AvgPool()
    net.load_state_dict(torch.load('./models/mnist_lenet_avgpool.pth'));

    (sensor, knet) = keynet.system.IdentityKeynet(inshape, net)
    assert np.allclose(knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet_constructor]:  IdentityKeynet (scipy) PASSED')

    (sensor, knet) = keynet.system.Keynet(inshape, net, 'identity', 'torch')
    assert np.allclose(knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet_constructor]:  IdentityKeynet (torch) PASSED')
    
    (sensor, knet) = keynet.system.PermutationKeynet(inshape, net, do_output_encryption=False)
    assert np.allclose(knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet_constructor]:  PermutationKeynet PASSED')    

    (sensor, knet) = keynet.system.StochasticKeynet(inshape, net, alpha=1, do_output_encryption=False)
    assert np.allclose(knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet_constructor]:  StochasticKeynet (alpha=1) PASSED')    

    (sensor, knet) = keynet.system.StochasticKeynet(inshape, net, alpha=2, do_output_encryption=False)
    assert np.allclose(knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet_constructor]:  StochasticKeynet (alpha=2) PASSED')    
    
    (sensor, knet) = keynet.system.IdentityTiledKeynet(inshape, net, 27)
    assert np.allclose(knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    knet.num_parameters()
    print('[test_keynet_constructor]:  IdentityTiledKeynet PASSED')
        
    (sensor, knet) = keynet.system.PermutationTiledKeynet(inshape, net, 27, do_output_encryption=False)
    yh = knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten()
    y = net.forward(x).detach().numpy().flatten()
    assert np.allclose(knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    knet.num_parameters()    
    print('[test_keynet_constructor]:  PermutationTiledKeynet PASSED')
    
    
    inshape = (3,32,32)
    net = keynet.cifar10.AllConvNet()    
    net.load_state_dict(torch.load('./models/cifar10_allconv.pth', map_location=torch.device('cpu')));
    x = torch.randn(1, *inshape)
    (sensor, knet) = keynet.system.IdentityTiledKeynet(inshape, net, 2048)    
    yh = knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten()
    y = net.forward(x).detach().numpy().flatten()
    assert np.allclose(yh, y, atol=1E-5)    
    print('[test_keynet_constructor]:  PASSED')    
    
    
def test_keynet_mnist():
    torch.set_grad_enabled(False)
    np.random.seed(0)
    X = [torch.tensor(np.random.rand(1,1,28,28).astype(np.float32)) for j in range(0,16)]

    # LeNet
    net = keynet.mnist.LeNet()
    net.load_state_dict(torch.load('./models/mnist_lenet.pth'))
    net.eval()
    with Stopwatch() as sw:
        y = [net(x) for x in X]        
        y = [net(x) for x in X]
    print('[test_keynet_mnist]: Elapsed: %f sec' % (sw.elapsed/(2*len(X))))
    print('[test_keynet_mnist]: LeNet parameters: %d' % keynet.torch.count_parameters(net))

    # LeNet-AvgPool
    net = keynet.mnist.LeNet_AvgPool()
    net.load_state_dict(torch.load('./models/mnist_lenet_avgpool.pth'))
    net.eval()
    with Stopwatch() as sw:
        y = [net(x) for x in X]
    print('[test_keynet_mnist]: Elapsed: %f sec' % (sw.elapsed/len(X)))
    print('[test_keynet_mnist]: LeNet_AvgPool parameters: %d' % keynet.torch.count_parameters(net))

    # Identity KeyNet
    inshape = (1,28,28)
    (sensor, knet) = keynet.system.IdentityKeynet(inshape, net)    
    yh_identity = knet.forward(sensor.encrypt(X[0]).tensor()).detach().numpy().flatten()
    assert(np.allclose(np.array(y[0]), yh_identity))
    print('[test_keynet_mnist]:  IdentityKeynet: passed')
    print('[test_keynet_mnist]:  IdentityKey parameters: %d' % knet.num_parameters())

    # IdentityTiled keynet
    inshape = (1,28,28)
    (sensor, knet) = keynet.system.IdentityTiledKeynet(inshape, net, tilesize=32)    
    yh_identity = knet.forward(sensor.encrypt(X[0]).tensor()).detach().numpy().flatten()
    assert(np.allclose(np.array(y[0]), yh_identity))
    print('[test_keynet_mnist]:  IdentityTiledKeynet: passed')
    print('[test_keynet_mnist]:  IdentityTiledKey parameters: %d' % knet.num_parameters())
    
    # Permutation KeyLeNet
    (sensor, knet) = keynet.system.PermutationTiledKeynet(inshape, net, tilesize=32)    
    yh = knet.forward(sensor.encrypt(X[0]).tensor()).detach().numpy().flatten()
    assert(np.allclose(np.array(y[0]), yh))
    print('[test_keynet_mnist]:  PermutationTiledKeynet: passed')
    print('[test_keynet_mnist]:  PermutationTiledKey parameters: %d' % knet.num_parameters())

    # Stochastic
    (sensor, knet) = keynet.system.StochasticKeynet(inshape, net, alpha=2, beta=2)    
    yh = knet.forward(sensor.encrypt(X[0]).tensor()).detach().numpy().flatten()
    assert(np.allclose(np.array(y[0]), yh))
    print('[test_keynet_mnist]:  StochasticKeynet: passed')
    print('[test_keynet_mnist]:  StochasticKeynet parameters: %d' % knet.num_parameters())

    # Stochastic Tiled
    (sensor, knet) = keynet.system.StochasticTiledKeynet(inshape, net, alpha=2, beta=2, tilesize=32)    
    yh = knet.forward(sensor.encrypt(X[0]).tensor()).detach().numpy().flatten()
    assert(np.allclose(np.array(y[0]), yh))
    print('[test_keynet_mnist]:  StochasticTiledKeynet: passed')
    print('[test_keynet_mnist]:  StochasticTiledKeynet parameters: %d' % knet.num_parameters())    
    return 

    
def test_keynet_cifar10():
    raise ValueError('FIXME')

    from keynet.cifar10 import AllConvNet, StochasticKeyNet

    torch.set_grad_enabled(False)
    np.random.seed(0)
    X = [torch.tensor(np.random.rand(1,3,32,32).astype(np.float32)) for j in range(0,16)]

    # AllConvNet
    net = AllConvNet()
    net.eval()
    net.load_state_dict(torch.load('./models/cifar10_allconv.pth'))

    with Stopwatch() as sw:
        y = [net(x) for x in X]
    with Stopwatch() as sw:
        y = [net(x) for x in X]
    print('AllConvNet Elapsed: %f sec' % (sw.elapsed/len(X)))

    # StochasticKeyNet
    for alpha in [1,2,4]:
        A0 = sparse_permutation_matrix(3*32*32 + 1)
        A0inv = A0.transpose()
        knet = StochasticKeyNet(alpha=alpha, use_cupy_sparse=False, use_torch_sparse=False)
        knet.eval()    
        knet.load_state_dict_keyed(torch.load('./models/cifar10_allconv.pth'), A0inv=A0inv)
        Xh = [knet.encrypt(A0, x) for x in X]
        with Stopwatch() as sw:
            yh = [knet.decrypt(knet(xh)) for xh in Xh]
        with Stopwatch() as sw:
            yh = [knet.decrypt(knet(xh)) for xh in Xh]
        print('Keyed-AllConvNet (alpha=%d) Elapsed: %f sec' % (alpha, sw.elapsed/len(Xh)))

        print(y[0], yh[0])
        assert (np.allclose(np.array(y[0]).flatten(), np.array(yh[0]).flatten(), atol=1E-1))
        print('CIFAR-10 StochasticKeyNet (alpha=%d): passed' % alpha)
        
        print('AllConvNet parameters: %d' % keynet.torch.count_parameters(net))
        print('StochasticKeyNet (alpha=%d) parameters: %d' % (alpha, keynet.torch.count_keynet_parameters(knet)))
    

def test_sparse_matrix():

    W_numpy = np.random.rand(3,3).astype(np.float32)
    x_numpy = np.random.rand(3,1).astype(np.float32)
    y = W_numpy.dot(x_numpy)
    
    W_torch = torch.as_tensor(W_numpy)
    x_torch = torch.as_tensor(x_numpy)

    W_torch_sparse = keynet.torch.scipy_coo_to_torch_sparse(scipy.sparse.coo_matrix(W_numpy))    
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
    test_torch_homogenize()
    test_sparse_toeplitz_conv2d()
    test_sparse_toeplitz_avgpool2d()
    #test_keynet_cifar10()
    test_blockview()
    test_sparse_matrix()
    test_sparse_tiled_matrix()        
    test_keynet_constructor()
    test_keynet_mnist()    
