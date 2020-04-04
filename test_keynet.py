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
    assert np.allclose(knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet]:  IdentityKeynet PASSED')


def test_permutation_keynet():
    inshape = (1,28,28)
    x = torch.randn(1, *inshape)
    net = keynet.mnist.LeNet_AvgPool()
    net.load_state_dict(torch.load('./models/mnist_lenet_avgpool.pth'))

    (sensor, knet) = keynet.system.PermutationKeynet(inshape, net)
    assert np.allclose(knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet]:  PermutationKeynet  -  PASSED')


def test_gain_keynet():
    inshape = (1,28,28)
    x = torch.randn(1, *inshape)
    net = keynet.mnist.LeNet_AvgPool()

    (sensor, knet) = keynet.system.Keynet(inshape, net, global_photometric='uniform_gain', beta=1.0)
    assert np.allclose(knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet]:  Analog Gain Keynet  -  PASSED')


def test_bias_keynet():
    inshape = (1,28,28)
    x = torch.randn(1, *inshape)
    net = keynet.mnist.LeNet_AvgPool()

    (sensor, knet) = keynet.system.Keynet(inshape, net, global_photometric='uniform_bias', beta=1.0)
    assert np.allclose(knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet]:  Analog Bias Keynet  -  PASSED')
    
    

def test_keynet_scipy():
    inshape = (1,28,28)
    x = torch.randn(1, *inshape)
    net = keynet.mnist.LeNet_AvgPool()
    net.load_state_dict(torch.load('./models/mnist_lenet_avgpool.pth'));

    (sensor, knet) = keynet.system.IdentityKeynet(inshape, net)
    assert np.allclose(knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet_constructor]:  IdentityKeynet (scipy) PASSED')
    vipy.util.save((sensor, knet), 'keynet_lenet_identity.pkl')
        
    (sensor, knet) = keynet.system.PermutationKeynet(inshape, net, do_output_encryption=False)
    assert np.allclose(knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet_constructor]:  PermutationKeynet PASSED')    

    (sensor, knet) = keynet.system.StochasticKeynet(inshape, net, alpha=1, do_output_encryption=False)
    assert np.allclose(knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    vipy.util.save((sensor, knet), 'keynet_lenet_stochastic.pkl')        
    print('[test_keynet_constructor]:  StochasticKeynet (alpha=1) PASSED')    

    (sensor, knet) = keynet.system.StochasticKeynet(inshape, net, alpha=2, do_output_encryption=False)
    assert np.allclose(knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet_constructor]:  StochasticKeynet (alpha=2) PASSED')    
    
    (sensor, knet) = keynet.system.TiledIdentityKeynet(inshape, net, 27)
    assert np.allclose(knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    knet.num_parameters()
    print('[test_keynet_constructor]:  TiledIdentityKeynet PASSED')
        
    (sensor, knet) = keynet.system.TiledPermutationKeynet(inshape, net, 27, do_output_encryption=False)
    yh = knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten()
    y = net.forward(x).detach().numpy().flatten()
    assert np.allclose(knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    knet.num_parameters()    
    print('[test_keynet_constructor]:  TiledPermutationKeynet PASSED')
        
    inshape = (3,32,32)
    net = keynet.cifar10.AllConvNet()    
    net.load_state_dict(torch.load('./models/cifar10_allconv.pth', map_location=torch.device('cpu')));
    x = torch.randn(1, *inshape)
    (sensor, knet) = keynet.system.IdentityKeynet(inshape, net)    
    yh = knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten()
    y = net.forward(x).detach().numpy().flatten()
    assert np.allclose(yh, y, atol=1E-5)
    vipy.util.save((sensor, knet), 'keynet_allconv.pkl')
    print('[test_keynet_constructor]:  IdentityKeynet (allconvnet) parameters=%d' % (knet.num_parameters()))
    print('[test_keynet_constructor]:  IdentityKeynet (allconvnet) PASSED')

    GLOBAL['PROCESSES'] = 8
    (sensor, knet) = keynet.system.TiledIdentityKeynet(inshape, net, 32)    
    yh = knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten()
    y = net.forward(x).detach().numpy().flatten()
    assert np.allclose(yh, y, atol=1E-5)
    vipy.util.save((sensor, knet), 'keynet_allconv_tiled.pkl')
    print('[test_keynet_constructor]:  TiledIdentityKeynet (allconvnet) parameters=%d' % (knet.num_parameters()))    
    print('[test_keynet_constructor]:  TiledIdentityKeynet (allconvnet) PASSED')

    print('[test_keynet_constructor]:  PASSED')    
    

def _test_keynet_torch():
    # FIXME
    (sensor, knet) = keynet.system.Keynet(inshape, net, 'identity', 'torch')
    assert np.allclose(knet.forward(sensor.encrypt(x).tensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
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
    yh_identity = knet.forward(sensor.encrypt(X[0]).tensor()).detach().numpy().flatten()
    assert(np.allclose(np.array(y[0]), yh_identity))
    print('[test_keynet_mnist]:  IdentityKeynet: passed')
    print('[test_keynet_mnist]:  IdentityKey parameters: %d' % knet.num_parameters())

    # IdentityTiled keynet
    inshape = (1,28,28)
    (sensor, knet) = keynet.system.TiledIdentityKeynet(inshape, net, tilesize=32)    
    yh_identity = knet.forward(sensor.encrypt(X[0]).tensor()).detach().numpy().flatten()
    assert(np.allclose(np.array(y[0]), yh_identity))
    print('[test_keynet_mnist]:  TiledIdentityKeynet: passed')
    print('[test_keynet_mnist]:  TiledIdentityKey parameters: %d' % knet.num_parameters())
    
    # Permutation KeyLeNet
    (sensor, knet) = keynet.system.TiledPermutationKeynet(inshape, net, tilesize=32)    
    yh = knet.forward(sensor.encrypt(X[0]).tensor()).detach().numpy().flatten()
    assert(np.allclose(np.array(y[0]), yh))
    print('[test_keynet_mnist]:  TiledPermutationKeynet: passed')
    print('[test_keynet_mnist]:  TiledPermutationKeynet parameters: %d' % knet.num_parameters())

    # Stochastic
    (sensor, knet) = keynet.system.StochasticKeynet(inshape, net, alpha=2, beta=2)    
    yh = knet.forward(sensor.encrypt(X[0]).tensor()).detach().numpy().flatten()
    assert(np.allclose(np.array(y[0]), yh))
    print('[test_keynet_mnist]:  StochasticKeynet: passed')
    print('[test_keynet_mnist]:  StochasticKeynet parameters: %d' % knet.num_parameters())

    # Stochastic Tiled
    (sensor, knet) = keynet.system.TiledStochasticKeynet(inshape, net, alpha=2, beta=2, tilesize=32)    
    yh = knet.forward(sensor.encrypt(X[0]).tensor()).detach().numpy().flatten()
    assert(np.allclose(np.array(y[0]), yh))
    print('[test_keynet_mnist]:  TiledStochasticKeynet: passed')
    print('[test_keynet_mnist]:  TiledStochasticKeynet parameters: %d' % knet.num_parameters())    
    return 

    


def test_vgg16():
    keynet.globals.num_processes(48)
    net = keynet.vgg.VGG16()
    print('vgg16: num parameters=%d' % keynet.torch.count_parameters(net))
    (sensor, model) = keynet.system.TiledIdentityKeynet( (3, 224, 224), net, 32)
    print('vgg16: keynet num parameters=%d' % model.num_parameters())
    return 


def test_vgg16_permutation():
    keynet.globals.num_processes(48)    
    net = keynet.vgg.VGG16()
    print('vgg16: num parameters=%d' % keynet.torch.count_parameters(net))
    (sensor, model) = keynet.system.TiledPermutationKeynet( (3, 224, 224), net, 32)
    print('vgg16: keynet num parameters=%d' % model.num_parameters())
    print(vipy.util.save((sensor, model), 'keynet_vgg16_tiled_permutation.pkl'))


def test_vgg16_stochastic():
    keynet.globals.num_processes(48)    
    net = keynet.vgg.VGG16()
    print('vgg16: num parameters=%d' % keynet.torch.count_parameters(net))
    (sensor, model) = keynet.system.TiledStochasticKeynet( (3, 224, 224), net, 32, alpha=2, beta=1.0)
    print('vgg16: keynet num parameters=%d' % model.num_parameters())
    print(vipy.util.save((sensor, model), 'keynet_vgg16_tiled_stochastic_alpha%d_tile%d.pkl' % (2, 32)))


if __name__ == '__main__':
    test_identity_keynet()
    test_permutation_keynet()
    test_gain_keynet()    
    test_bias_keynet()
    
    #test_keynet_scipy()    
    
    #test_vgg16_stochastic()
    #test_memory_order()
    #test_keynet_mnist()
    #test_vgg16_permutation()
    #test_vgg16()
