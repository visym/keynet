from numpy.linalg import multi_dot 
import numpy as np
import scipy.linalg
import PIL
from PIL import Image
import copy
import torch.nn.functional as F
import torch 
import vipy.image  # bash setup
import vipy.visualize  # bash setup
from vipy.util import Stopwatch
from keynet.util import sparse_permutation_matrix, sparse_identity_matrix, sparse_generalized_permutation_block_matrix, sparse_generalized_stochastic_block_matrix
from keynet.torch import affine_augmentation_tensor, affine_deaugmentation_tensor, affine_deaugmentation
from keynet.torch import sparse_toeplitz_conv2d, conv2d_in_scipy
from keynet.torch import sparse_toeplitz_avgpool2d, avgpool2d_in_scipy, conv2d_in_scipy
from keynet.util import sparse_diagonal_matrix, sparse_inverse_diagonal_matrix, random_dense_positive_definite_matrix
import keynet.util
import keynet.blockpermute
import keynet.mnist
import keynet.cifar10
import keynet.torch
import keynet.fiberbundle
from torch import nn
import cupyx
import cupy

def example_2x2():
    """Reproduce figure 2 in paper"""
    np.random.seed(0)

    img = np.array([[11,12],[21,22]]).astype(np.float32)
    x = img.flatten().reshape(4,1)
    D1 = keynet.util.uniform_random_dense_diagonal_matrix(4)
    P1 = keynet.util.random_dense_doubly_stochastic_matrix(4,2)
    A1 = np.dot(D1,P1)
    A1inv = np.linalg.inv(A1)

    P2 = keynet.util.random_dense_permutation_matrix(4)
    D2 = keynet.util.uniform_random_dense_diagonal_matrix(4)
    A2 = np.dot(D2,P2)
    A2inv = np.linalg.inv(A2)

    W1 = np.array([[-1,1,0,0],[0,-1,0,0],[0,0,-1,1],[0,0,0,-1]]).astype(np.float32)
    W1hat = multi_dot( (A2,W1,A1inv) )

    print('img')
    print(img)

    print('x')
    print(x)

    print('D1')
    print(D1)
    print('P1')
    print(P1)
    print('A1')
    print(A1)

    print('W1')
    print(W1)
    print('W1hat')
    print(W1hat)

    print('A1inv*A1')
    print(np.dot(A1inv, A1))

    print('A2')
    print(A2)

    x1h = multi_dot( (W1hat, A1, x) )

    x2h = copy.deepcopy(x1h)
    x2h[x2h<=0] = 0  # ReLU

    x1 = multi_dot( (W1, x) )
    x2 = copy.deepcopy(x1)
    x2[x2<=0] = 0  # ReLU

    print('A1*x, x1, x1h, x2, x2h, A2inv * x2h')
    print(multi_dot( (A1, x) ))
    print(x1)
    print(x1h)

    print(x2)
    print(x2h)
    print(multi_dot( (A2inv, x2h) ))



def optical_transformation_montage():
    (m,n) = (256,256)
    img = np.array(PIL.Image.open('owl.jpg').resize( (256,256) ))

    D = [np.maximum(1E-6, 1.0 + (s*np.random.rand( m,n,3 )-(s/2.0))) for s in [0.1, 1.0, 10000.0]]
    B = [255*np.maximum(1E-6, s*np.random.rand( m,n,3 )) for s in [0.1, 1.0, 10000.0]]
    P = [keynet.blockpermute.identity_permutation_mask(256),
         keynet.blockpermute.local_permutation_mask(256, 2, minscale=2, identityscale=3), 
         keynet.blockpermute.local_permutation_mask(256, 2, minscale=3, identityscale=4), 
         keynet.blockpermute.local_permutation_mask(256, 2, minscale=4, identityscale=5), 
         keynet.blockpermute.local_permutation_mask(256, 2, minscale=4, identityscale=6), 
         keynet.blockpermute.local_permutation_mask(256, 2, minscale=4, identityscale=7), 
         keynet.blockpermute.local_permutation_mask(256, 2, minscale=4, identityscale=8)]
         
    imlist = []
    for (d,b) in zip(D,B):
        for p in P:
            img_permuted = keynet.blockpermute.block_permute(img, p)
            img_scaled = np.multiply(d, img_permuted) + b
            img_scaled = np.uint8(255*((img_scaled-np.min(img_scaled))/ (np.max(img_scaled)-np.min(img_scaled))))
            imlist.append(vipy.image.Image(array=img_scaled))

    img_montage = vipy.visualize.montage(imlist, 256,256, rows=3, cols=7, grayscale=False, skip=False, border=1)

    print(keynet.util.savetemp(img_montage))
    return img_montage



def test_affine_augmentation():
    (N,C,U,V) = (2,2,3,3)
    x = torch.tensor(np.random.rand( N,C,U,V ).astype(np.float32))    

    x_affine = affine_augmentation_tensor(x)
    x_deaffine = affine_deaugmentation_tensor(x_affine).reshape( N,C,U,V )
    assert(np.allclose(x, x_deaffine))
    print('Affine augmentation (round-trip): passed')    
    
    x_affine_numpy = np.hstack((np.array(x).reshape(N, C*U*V), np.ones( (N,1) ))).transpose()
    assert(np.allclose(x_affine, x_affine_numpy))
    print('Affine augmentation (torch vs. numpy): passed')    
    


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
    y_scipy = conv2d_in_scipy(img, f, b, stride=stride)
    assert(np.allclose(y_scipy, yh))
    print('Correlation (scipy vs. toeplitz): passed')    

    # Torch spatial correlation: reshape torch to be tensor sized [BATCH x CHANNEL x HEIGHT x WIDTH]
    # Padding required to allow for valid convolution to be the same size as input
    y_torch = F.conv2d(torch.tensor(img), torch.tensor(f), bias=torch.tensor(b), padding=((P-1)//2, (Q-1)//2), stride=stride)
    assert(np.allclose(y_torch,yh))
    print('Correlation (torch vs. toeplitz): passed')

    return(T)


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
    y_scipy = avgpool2d_in_scipy(img, kernelsize, stride)
    assert(np.allclose(y_scipy, yh))
    print('Average pool 2D (scipy vs. toeplitz): passed')    

    # Torch avgpool
    y_torch = F.avg_pool2d(torch.tensor(img), kernelsize, stride=stride, padding=((P-1)//2, (Q-1)//2))
    assert(np.allclose(y_torch,yh))
    print('Average pool 2D (torch vs. toeplitz): passed')
    return T


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
    with Stopwatch() as sw:
        y = [net(x) for x in X]
    print('Elapsed: %f sec' % (sw.elapsed/len(X)))
    print('LeNet parameters: %d' % keynet.torch.count_parameters(net))

    # LeNet-AvgPool
    net = keynet.mnist.LeNet_AvgPool()
    net.load_state_dict(torch.load('./models/mnist_lenet_avgpool.pth'))
    net.eval()
    with Stopwatch() as sw:
        y = [net(x) for x in X]
    print('Elapsed: %f sec' % (sw.elapsed/len(X)))
    print('LeNet_AvgPool parameters: %d' % keynet.torch.count_parameters(net))

    # Identity KeyNet
    A0 = sparse_identity_matrix(28*28*1 + 1)
    A0inv = A0
    knet = keynet.mnist.KeyNet()
    knet.load_state_dict_keyed(torch.load('./models/mnist_lenet_avgpool.pth'), A0inv=A0inv)
    knet.eval()    
    yh_identity = knet.decrypt(knet(knet.encrypt(A0, X[0])))
    assert(np.allclose(np.array(y[0]), np.array(yh_identity)))
    print('MNIST IdentityKeyNet: passed')
    print('IdentityKeyNet parameters: %d' % keynet.torch.count_keynet_parameters(knet))

    # Permutation KeyNet
    A0 = sparse_permutation_matrix(28*28*1 + 1)
    A0inv = A0.transpose()
    knet = keynet.mnist.PermutationKeyNet()
    knet.load_state_dict_keyed(torch.load('./models/mnist_lenet_avgpool.pth'), A0inv=A0inv)
    knet.eval()    
    yh_permutation = knet.decrypt(knet(knet.encrypt(A0, X[0])))
    fc3_permutation = knet.fc3.What
    assert(np.allclose(np.array(y[0]), np.array(yh_permutation)))
    print('MNIST PermutationKeyNet: passed')
    print('PermutationKeyNet parameters: %d' % keynet.torch.count_keynet_parameters(knet))

    # Diagonal KeyNet
    A0 = sparse_diagonal_matrix(28*28*1 + 1)
    A0inv = sparse_inverse_diagonal_matrix(A0)
    knet = keynet.mnist.DiagonalKeyNet()
    knet.load_state_dict_keyed(torch.load('./models/mnist_lenet_avgpool.pth'), A0inv=A0inv)
    knet.eval()    
    yh_diagonal = knet.decrypt(knet(knet.encrypt(A0, X[0])))
    fc3_diagonal = knet.fc3.What
    assert(np.allclose(np.array(y[0]), np.array(yh_diagonal)))
    print('MNIST DiagonalKeyNet: passed')
    print('DiagonalKeyNet parameters: %d' % keynet.torch.count_keynet_parameters(knet))

    # Diagonal vs. Permutation KeyNet (fc3 What comparison)
    assert(not np.allclose(np.array(fc3_permutation), np.array(fc3_diagonal)))
    print('MNIST PermutationKeyNet vs. DiagonalKeyNet: passed')

    # Stochastic KeyNet
    A0 = sparse_diagonal_matrix(28*28*1 + 1)
    A0inv = sparse_inverse_diagonal_matrix(A0)
    knet = keynet.mnist.StochasticKeyNet()
    knet.load_state_dict_keyed(torch.load('./models/mnist_lenet_avgpool.pth'), A0inv=A0inv)
    knet.eval()    
    yh_stochastic = knet.decrypt(knet(knet.encrypt(A0, X[0])))
    assert(np.allclose(np.array(y[0]), np.array(yh_stochastic)))
    print('MNIST StochasticKeyNet: passed')
    print('StochasticKeyNet parameters: %d' % keynet.torch.count_keynet_parameters(knet))

    # Block KeyNet
    for alpha in [1,2,4]:
        A0 = sparse_diagonal_matrix(28*28*1 + 1)
        A0inv = sparse_inverse_diagonal_matrix(A0)
        knet = keynet.mnist.BlockKeyNet(alpha=alpha)
        knet.load_state_dict_keyed(torch.load('./models/mnist_lenet_avgpool.pth'), A0inv=A0inv)
        knet.eval()    
        Xh = [knet.encrypt(A0, x) for x in X]
        with Stopwatch() as sw:
            yh_stochastic = [knet.decrypt(knet(xh)) for xh in Xh]
        with Stopwatch() as sw:
            yh_stochastic = [knet.decrypt(knet(xh)) for xh in Xh]
        print('Elapsed: %f sec' % (sw.elapsed/len(X)))
        print(y[0])
        print(yh_stochastic[0])
        assert(np.allclose(np.array(y[0]), np.array(yh_stochastic[0]), atol=1E-1))
        print('MNIST BlockKeyNet (alpha=%d): passed' % alpha)
        print('BlockKeyNet (alpha=%d) parameters: %d' % (alpha, keynet.torch.count_keynet_parameters(knet)))

    # Generalized Stochastic KeyNet
    # FIXME: the number of parameters isn't quite right here due to a boundary condition on keynet.util.sparse_random_diagonally_dominant_doubly_stochastic_matrix
    for alpha in [1,2,4]:
        A0 = sparse_diagonal_matrix(28*28*1 + 1)
        A0inv = sparse_inverse_diagonal_matrix(A0)
        knet = keynet.mnist.GeneralizedStochasticKeyNet(alpha=alpha)
        knet.load_state_dict_keyed(torch.load('./models/mnist_lenet_avgpool.pth'), A0inv=A0inv)
        knet.eval()    
        Xh = [knet.encrypt(A0, x) for x in X]
        with Stopwatch() as sw:
            yh_stochastic = [knet.decrypt(knet(xh)) for xh in Xh]
        with Stopwatch() as sw:
            yh_stochastic = [knet.decrypt(knet(xh)) for xh in Xh]
        print('Elapsed: %f sec' % (sw.elapsed/len(X)))
        assert(np.allclose(np.array(y[0]), np.array(yh_stochastic[0]), atol=1E-4))
        print('MNIST GeneralizedStochasticKeyNet (alpha=%d): passed' % alpha)
        print('GeneralizedStochasticKeyNet (alpha=%d) parameters: %d' % (alpha, keynet.torch.count_keynet_parameters(knet)))


    
def test_keynet_cifar10():
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


def test_roundoff(m=512, n=1000):
    x = np.random.randn(m,1).astype(np.float32)
    xh = x
    for j in range(0,n):
        A = random_dense_positive_definite_matrix(m).astype(np.float32)
        xh = np.dot(A,xh)
        Ainv = np.linalg.inv(A)
        xh = np.dot(Ainv,xh)
        if j % 10 == 0:
            print('[test_roundoff]: m=%d, n=%d, j=%d, |x-xh|/|x|=%1.13f, |x-xh]=%1.13f' % (m,n,j, np.max(np.abs(x-xh)/np.abs(x)), np.max(np.abs(x-xh))))


def test_fiberbundle_simulation():
    img_color = np.array(PIL.Image.open('owl.jpg').resize( (512,512) ))
    img_sim = keynet.fiberbundle.simulation(img_color, h_xtalk=0.05, v_xtalk=0.05, fiber_core_x=16, fiber_core_y=16, do_camera_noise=True)
    keynet.util.savetemp(np.uint8(img_sim))


def test_fiberbundle_simulation_32x32():
    img_color_large = np.array(PIL.Image.open('owl.jpg').resize( (512,512), PIL.Image.BICUBIC ))  
    img_sim_large = keynet.fiberbundle.simulation(img_color_large, h_xtalk=0.05, v_xtalk=0.05, fiber_core_x=16, fiber_core_y=16, do_camera_noise=False)
    img_sim = np.array(PIL.Image.fromarray(np.uint8(img_sim_large)).resize( (32,32), PIL.Image.BICUBIC ).resize( (512,512), PIL.Image.NEAREST ))
    #keynet.util.savetemp(np.uint8(img_color_large))
    keynet.util.savetemp(np.uint8(img_sim))

    
def test_semantic_security():
    W = sparse_toeplitz_conv2d( (1,8,8), np.ones( (1,1,3,3) ))
    (B,Binv) = sparse_generalized_stochastic_block_matrix(65,1)
    (A,Ainv) = sparse_generalized_permutation_block_matrix(65,2)
    What = B*W*Ainv
    print([what.nnz - w.nnz for (what, w) in zip(What.tocsr(), W.tocsr())])
    assert(np.all([what.nnz > w.nnz for (what, w) in zip(What.tocsr(), W.tocsr())]))
    print(What.nnz)
    What = W*Ainv
    assert(What.nnz > W.nnz)

    print('test_semantic_security: passed')
    # FIXME: A,Ainv are negative in general for permutation, Ainv is nevative for stochastic
    # If Ainv is nevative, then AWAinv -> A(WAinv_+ - WAinv_-) and same as before to recover A, which can be used to recover W and Ainv


def test_sparse_multiply():
    (n_outchannel, n_inchannel, width, height, kernelsize) = (32,32,128,128,3)
    #(n_outchannel, n_inchannel, width, height, kernelsize) = (6,4,8,8,3)  # Quicktest parameters

    f = nn.Conv2d(n_inchannel, n_outchannel, kernelsize, padding=1)     
    f_numpy = lambda x: conv2d_in_scipy(x, f=f.weight.data.detach().numpy(), b=f.bias.data.detach().numpy(), stride=1)

    W = sparse_toeplitz_conv2d( (n_inchannel,width,height), f.weight.data.detach().numpy(), bias=f.bias.data.detach().numpy())
    W_torch = keynet.torch.scipy_coo_to_torch_sparse(W)
    W = W.tocsr()
    (A,Ainv) = sparse_generalized_permutation_block_matrix(n_inchannel*width*height+1, 1) 
    (B,Binv) = sparse_generalized_permutation_block_matrix(n_outchannel*width*height+1, 1) 
    W_keynet = B*W*Ainv  
    W_keynet_torch = keynet.torch.scipy_coo_to_torch_sparse(W_keynet.tocoo())

    X_numpy = [np.random.rand( 1,n_inchannel,width,height ).astype(np.float32) for j in range(0,10)]
    X_torch = [torch.tensor(x) for x in X_numpy]
    X_torch_keynet = [affine_augmentation_tensor(x) for x in X_torch]
    X_numpy_keynet = [np.array(x) for x in X_torch_keynet]

    # Sanity check
    assert(np.allclose(f(X_torch[0]).detach().numpy(), f_numpy(X_numpy[0]), atol=1E-4))
    assert(np.allclose(f_numpy(X_numpy[0]).flatten(), affine_deaugmentation(W.dot(X_numpy_keynet[0])), atol=1E-4))

    # Timing
    with Stopwatch() as sw:
        y = [f(x) for x in X_torch]
    print('pytorch conv2d (cpu): %f ms' % (1000*sw.elapsed/len(X_torch)))

    with Stopwatch() as sw:
        y = [torch.sparse.mm(W_torch, x) for x in X_torch_keynet]
    print('pytorch sparse conv2d (cpu): %f ms' % (1000*sw.elapsed/len(X_torch_keynet)))

    with Stopwatch() as sw:
        y = [torch.sparse.mm(W_keynet_torch, x) for x in X_torch_keynet]
    print('pytorch sparse conv2d permuted (cpu): %f ms' % (1000*sw.elapsed/len(X_torch_keynet)))

    with Stopwatch() as sw:
        y = [f_numpy(x) for x in X_numpy]
    print('numpy conv2d (cpu): %f ms' % (1000*sw.elapsed/len(X_numpy)))

    with Stopwatch() as sw:
        y = [W.dot(x) for x in X_numpy_keynet]
    print('numpy sparse conv2d (cpu): %f ms' % (1000*sw.elapsed/len(X_numpy_keynet)))

    with Stopwatch() as sw:
        y = [torch.as_tensor(W.dot(x.numpy())) for x in X_torch_keynet]
    print('numpy sparse conv2d with torch conversion (cpu): %f ms' % (1000*sw.elapsed/len(X_torch_keynet)))

    with Stopwatch() as sw:
        y = [W_keynet.dot(x) for x in X_numpy_keynet]
    print('numpy sparse conv2d permuted (cpu): %f ms' % (1000*sw.elapsed/len(X_numpy_keynet)))

    with Stopwatch() as sw:
        y = [torch.as_tensor(W_keynet.dot(x.numpy())) for x in X_torch_keynet]
    print('numpy sparse conv2d permuted with torch conversion (cpu): %f ms' % (1000*sw.elapsed/len(X_torch_keynet)))

    f_gpu = f.cuda(0)
    X_torch_gpu = [x.cuda(0) for x in X_torch]    
    with Stopwatch() as sw:
        for j in range(0,len(X_torch)):
            y = f_gpu(X_torch_gpu[j])
    print('pytorch conv2d (gpu): %f ms' % (1000*sw.elapsed/len(X_torch_gpu)))

    W_torch_gpu = W_torch.cuda(1)
    X_torch_keynet_gpu = [x.cuda(1) for x in X_torch_keynet]    
    with Stopwatch() as sw:
        for j in range(0,len(X_torch)):
            y = torch.sparse.mm(W_torch_gpu, X_torch_keynet_gpu[j])
    print('pytorch sparse conv2d (gpu): %f ms' % (1000*sw.elapsed/len(X_torch_keynet_gpu)))

    W_keynet_torch_gpu = W_keynet_torch.cuda(2)
    X_torch_keynet_gpu = [x.cuda(2) for x in X_torch_keynet]    
    with Stopwatch() as sw:
        for j in range(0,len(X_torch)):
            y = torch.sparse.mm(W_keynet_torch_gpu, X_torch_keynet_gpu[j])
    print('pytorch sparse conv2d permuted (gpu): %f ms' % (1000*sw.elapsed/len(X_torch_keynet_gpu)))
    
    with cupy.cuda.Device(3):
        W_keynet_cupy = cupyx.scipy.sparse.csr_matrix(W_keynet.tocsr())
        X_cupy_keynet_gpu = [cupy.array(x) for x in X_numpy_keynet]    
        with Stopwatch() as sw:
            for j in range(0,len(X_cupy_keynet_gpu)):
                y = W_keynet_cupy.dot(X_cupy_keynet_gpu[j])
        print('cupy sparse conv2d permuted (gpu): %f ms' % (1000*sw.elapsed/len(X_cupy_keynet_gpu)))
