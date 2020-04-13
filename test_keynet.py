import sys
import numpy as np
import scipy.linalg
import PIL
import copy
import torch 
from torch import nn
import torch.nn.functional as F
import keynet.sparse
from keynet.sparse import sparse_permutation_matrix, sparse_identity_matrix, sparse_identity_matrix_like
from keynet.torch import affine_to_linear, linear_to_affine, affine_to_linear_matrix
from keynet.sparse import sparse_toeplitz_conv2d, sparse_toeplitz_avgpool2d
from keynet.util import torch_avgpool2d_in_scipy, torch_conv2d_in_scipy
from keynet.dense import uniform_random_diagonal_matrix, random_positive_definite_matrix
import keynet.util
import keynet.mnist
import keynet.cifar10
import keynet.torch
import keynet.system
import keynet.vgg
import vipy
from vipy.util import Stopwatch
from keynet.globals import GLOBAL


def test_identity_keynet():
    inshape = (1,28,28)
    x = torch.randn(1, *inshape)
    net = keynet.mnist.LeNet_AvgPool()
    net.load_state_dict(torch.load('./models/mnist_lenet_avgpool.pth'))

    (sensor, knet) = keynet.system.IdentityKeynet(inshape, net)
    assert np.allclose(knet.forward(sensor.encrypt(x).astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet]:  IdentityKeynet PASSED')

    
def test_tiled_keynet():
    inshape = (1,28,28)
    x = torch.randn(1, *inshape)
    net = keynet.mnist.LeNet_AvgPool()

    (sensor, knet) = keynet.system.Keynet(inshape, net, backend='scipy', tileshape=(28,28))
    assert np.allclose(knet.forward(sensor.encrypt(x).astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet]:  tiled IdentityKeynet PASSED')    
    
    #(sensor, knet) = keynet.system.TiledPermutationKeynet(inshape, net, tileshape=(14,14))
    #assert np.allclose(knet.forward(sensor.encrypt(x).astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)    
    #print('[test_keynet]:  tiled PermutationKeynet PASSED')
    

def test_permutation_keynet():
    inshape = (1,28,28)
    x = torch.randn(1, *inshape)
    net = keynet.mnist.LeNet_AvgPool()
    net.load_state_dict(torch.load('./models/mnist_lenet_avgpool.pth'))

    (sensor, knet) = keynet.system.PermutationKeynet(inshape, net)
    assert np.allclose(knet.forward(sensor.encrypt(x).astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet]:  global PermutationKeynet  -  PASSED')    
    
    (sensor, knet) = keynet.system.Keynet(inshape, net, global_geometric='permutation', memoryorder='block', blocksize=14)
    assert np.allclose(knet.forward(sensor.encrypt(x).astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5) 
    print('[test_keynet]:  global PermutationKeynet  -  PASSED')


def test_photometric_keynet():
    inshape = (1,28,28)
    x = torch.randn(1, *inshape)
    net = keynet.mnist.LeNet_AvgPool()

    (sensor, knet) = keynet.system.Keynet(inshape, net, global_photometric='uniform_random_gain', beta=1.0)
    assert np.allclose(knet.forward(sensor.encrypt(x).astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet]:  Analog Gain Keynet  -  PASSED')

    (sensor, knet) = keynet.system.Keynet(inshape, net, global_photometric='uniform_random_bias', gamma=1.0)
    assert np.allclose(knet.forward(sensor.encrypt(x).astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet]:  Analog Bias Keynet  -  PASSED')

    (sensor, knet) = keynet.system.Keynet(inshape, net, global_photometric='uniform_random_affine', beta=1.0, gamma=1.0)
    assert np.allclose(knet.forward(sensor.encrypt(x).astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-4)
    print('[test_keynet]:  Analog Affine Keynet  -  PASSED')
        

def test_keynet_scipy():
    inshape = (1,28,28)
    x = torch.randn(1, *inshape)
    net = keynet.mnist.LeNet_AvgPool()
    net.load_state_dict(torch.load('./models/mnist_lenet_avgpool.pth'));

    (sensor, knet) = keynet.system.IdentityKeynet(inshape, net)
    assert np.allclose(knet.forward(sensor.encrypt(x).astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet_constructor]:  IdentityKeynet (scipy) PASSED')
    vipy.util.save((sensor, knet), 'keynet_lenet_identity.pkl')
        
    (sensor, knet) = keynet.system.PermutationKeynet(inshape, net, do_output_encryption=False)
    assert np.allclose(knet.forward(sensor.encrypt(x).astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet_constructor]:  PermutationKeynet PASSED')    

    (sensor, knet) = keynet.system.StochasticKeynet(inshape, net, alpha=1, do_output_encryption=False)
    assert np.allclose(knet.forward(sensor.encrypt(x).astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    vipy.util.save((sensor, knet), 'keynet_lenet_stochastic.pkl')        
    print('[test_keynet_constructor]:  StochasticKeynet (alpha=1) PASSED')    

    (sensor, knet) = keynet.system.StochasticKeynet(inshape, net, alpha=2, do_output_encryption=False)
    assert np.allclose(knet.forward(sensor.encrypt(x).astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet_constructor]:  StochasticKeynet (alpha=2) PASSED')    
    
    (sensor, knet) = keynet.system.TiledIdentityKeynet(inshape, net, 27)
    assert np.allclose(knet.forward(sensor.encrypt(x).astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    knet.num_parameters()
    print('[test_keynet_constructor]:  TiledIdentityKeynet PASSED')
        
    (sensor, knet) = keynet.system.TiledPermutationKeynet(inshape, net, 27, do_output_encryption=False)
    yh = knet.forward(sensor.encrypt(x).astensor()).detach().numpy().flatten()
    y = net.forward(x).detach().numpy().flatten()
    assert np.allclose(knet.forward(sensor.encrypt(x).astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    knet.num_parameters()    
    print('[test_keynet_constructor]:  TiledPermutationKeynet PASSED')
        
    inshape = (3,32,32)
    net = keynet.cifar10.AllConvNet()    
    net.load_state_dict(torch.load('./models/cifar10_allconv.pth', map_location=torch.device('cpu')));
    x = torch.randn(1, *inshape)
    (sensor, knet) = keynet.system.IdentityKeynet(inshape, net)    
    yh = knet.forward(sensor.encrypt(x).astensor()).detach().numpy().flatten()
    y = net.forward(x).detach().numpy().flatten()
    assert np.allclose(yh, y, atol=1E-5)
    vipy.util.save((sensor, knet), 'keynet_allconv.pkl')
    print('[test_keynet_constructor]:  IdentityKeynet (allconvnet) parameters=%d' % (knet.num_parameters()))
    print('[test_keynet_constructor]:  IdentityKeynet (allconvnet) PASSED')

    GLOBAL['PROCESSES'] = 8
    (sensor, knet) = keynet.system.TiledIdentityKeynet(inshape, net, 32)    
    yh = knet.forward(sensor.encrypt(x).astensor()).detach().numpy().flatten()
    y = net.forward(x).detach().numpy().flatten()
    assert np.allclose(yh, y, atol=1E-5)
    vipy.util.save((sensor, knet), 'keynet_allconv_tiled.pkl')
    print('[test_keynet_constructor]:  TiledIdentityKeynet (allconvnet) parameters=%d' % (knet.num_parameters()))    
    print('[test_keynet_constructor]:  TiledIdentityKeynet (allconvnet) PASSED')

    print('[test_keynet_constructor]:  PASSED')    
    

def _test_keynet_torch():
    # FIXME
    (sensor, knet) = keynet.system.Keynet(inshape, net, 'identity', 'torch')
    assert np.allclose(knet.forward(sensor.encrypt(x).astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet_constructor]:  IdentityKeynet (torch) PASSED')

    
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
    yh_identity = knet.forward(sensor.encrypt(X[0]).astensor()).detach().numpy().flatten()
    assert(np.allclose(np.array(y[0]), yh_identity))
    print('[test_keynet_mnist]:  IdentityKeynet: passed')
    print('[test_keynet_mnist]:  IdentityKey parameters: %d' % knet.num_parameters())

    # IdentityTiled keynet
    inshape = (1,28,28)
    (sensor, knet) = keynet.system.TiledIdentityKeynet(inshape, net, tilesize=32)    
    yh_identity = knet.forward(sensor.encrypt(X[0]).astensor()).detach().numpy().flatten()
    assert(np.allclose(np.array(y[0]), yh_identity))
    print('[test_keynet_mnist]:  TiledIdentityKeynet: passed')
    print('[test_keynet_mnist]:  TiledIdentityKey parameters: %d' % knet.num_parameters())
    
    # Permutation KeyLeNet
    (sensor, knet) = keynet.system.TiledPermutationKeynet(inshape, net, tilesize=32)    
    yh = knet.forward(sensor.encrypt(X[0]).astensor()).detach().numpy().flatten()
    assert(np.allclose(np.array(y[0]), yh))
    print('[test_keynet_mnist]:  TiledPermutationKeynet: passed')
    print('[test_keynet_mnist]:  TiledPermutationKeynet parameters: %d' % knet.num_parameters())

    # Stochastic
    (sensor, knet) = keynet.system.StochasticKeynet(inshape, net, alpha=2, beta=2)    
    yh = knet.forward(sensor.encrypt(X[0]).astensor()).detach().numpy().flatten()
    assert(np.allclose(np.array(y[0]), yh))
    print('[test_keynet_mnist]:  StochasticKeynet: passed')
    print('[test_keynet_mnist]:  StochasticKeynet parameters: %d' % knet.num_parameters())

    # Stochastic Tiled
    (sensor, knet) = keynet.system.TiledStochasticKeynet(inshape, net, alpha=2, beta=2, tilesize=32)    
    yh = knet.forward(sensor.encrypt(X[0]).astensor()).detach().numpy().flatten()
    assert(np.allclose(np.array(y[0]), yh))
    print('[test_keynet_mnist]:  TiledStochasticKeynet: passed')
    print('[test_keynet_mnist]:  TiledStochasticKeynet parameters: %d' % knet.num_parameters())    
    return 

    
def test_vgg16():

    inshape = (3,224,224)
    x = torch.randn(1, *inshape)
    net = keynet.vgg.VGG16()
    print('vgg16: num parameters=%d' % keynet.torch.count_parameters(net))

    keynet.globals.num_processes(48, backend='joblib')
    (sensor, knet) = keynet.system.TiledIdentityKeynet(inshape, net, 224//4)
    print(vipy.util.save((sensor, knet), 'test_vgg16.pkl'))
    #(sensor, knet) = vipy.util.load('test_vgg16.pkl')

    #keynet.globals.num_processes(24, backend='dask')
    assert np.allclose(knet.forward(sensor.encrypt(x).astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('vgg16: keynet-56 num parameters=%d' % knet.num_parameters())


def test_vgg16_stochastic():
    inshape = (3,224,224)
    x = torch.randn(1, *inshape)
    net = keynet.vgg.VGG16()
    print('vgg16: num parameters=%d' % keynet.torch.count_parameters(net))

    keynet.globals.num_processes(48)
    (sensor, knet) = keynet.system.Keynet(inshape, net, tileshape=(224//4, 224//4), 
                                          global_geometric='hierarchical_permutation', hierarchical_blockshape=(2,2), hierarchical_permute_at_level=(0,1,2),
                                          local_geometric='doubly_stochastic', alpha=2.0, blocksize=224//4,
                                          local_photometric='uniform_random_affine', beta=1.0, gamma=1.0,
                                          memoryorder='channel')
                                          
    assert np.allclose(knet.forward(sensor.encrypt(x).astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('vgg16: keynet-orthogonal-56 num parameters=%d' % knet.num_parameters())


def test_vgg16_orthogonal():
    inshape = (3,224,224)
    x = torch.randn(1, *inshape)
    net = keynet.vgg.VGG16()
    print('vgg16: num parameters=%d' % keynet.torch.count_parameters(net))


    keynet.globals.num_processes(48, backend='joblib')
    (sensor, knet) = keynet.system.Keynet(inshape, net, tileshape=(224//4, 224//4), 
                                          global_geometric='hierarchical_permutation', hierarchical_blockshape=(2,2), hierarchical_permute_at_level=(0,1,2),
                                          local_geometric='givens_orthogonal', alpha=2.0, blocksize=224//4,
                                          local_photometric='uniform_random_affine', beta=1.0, gamma=1.0,
                                          memoryorder='block')
    print(vipy.util.save((sensor, knet), 'test_vgg16_orthogonal.pkl'))
                                          
    keynet.globals.num_processes(48, backend='dask')
    assert np.allclose(knet.forward(sensor.encrypt(x).astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('vgg16: keynet-orthogonal-56 num parameters=%d' % knet.num_parameters())


if __name__ == '__main__':
    test_identity_keynet()
    test_tiled_keynet()
    test_permutation_keynet()
    test_photometric_keynet()
    
    #test_keynet_scipy()    
    
    #test_vgg16_stochastic()
    #test_memory_order()
    #test_keynet_mnist()

    if sys.argv[1] == 'vgg16':
        test_vgg16()
    elif sys.argv[1] == 'vgg16-orthogonal':
        #test_vgg16_stochastic()
        test_vgg16_orthogonal()
