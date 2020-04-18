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
    assert np.allclose(knet.forward(sensor.fromtensor(x).encrypt().astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet]:  IdentityKeynet PASSED')

    
def test_tiled_keynet():
    inshape = (1,28,28)
    x = torch.randn(1, *inshape)
    net = keynet.mnist.LeNet_AvgPool()
    (sensor, knet) = keynet.system.Keynet(inshape, net, backend='scipy', tileshape=None)
    (sensor, knet_tiled) = keynet.system.Keynet(inshape, net, backend='scipy', tileshape=(28,28))

    yh = knet.forward(sensor.fromtensor(x).encrypt().astensor()).detach().numpy().flatten()
    y = net.forward(x).detach().numpy().flatten()
    print(yh,y)
    assert np.allclose(knet.forward(sensor.fromtensor(x).encrypt().astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet]:  tiled IdentityKeynet PASSED')    
    
def test_permutation_keynet():
    inshape = (1,28,28)
    x = torch.randn(1, *inshape)
    net = keynet.mnist.LeNet_AvgPool()
    net.load_state_dict(torch.load('./models/mnist_lenet_avgpool.pth'))

    (sensor, knet) = keynet.system.PermutationKeynet(inshape, net)
    assert np.allclose(knet.forward(sensor.fromtensor(x).encrypt().astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet]:  global PermutationKeynet  -  PASSED')    
    
    (sensor, knet) = keynet.system.Keynet(inshape, net, global_geometric='permutation', memoryorder='block', blocksize=14)
    assert np.allclose(knet.forward(sensor.fromtensor(x).encrypt().astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5) 
    print('[test_keynet]:  global PermutationKeynet  -  PASSED')


def test_photometric_keynet():
    inshape = (1,28,28)
    x = torch.randn(1, *inshape)
    net = keynet.mnist.LeNet_AvgPool()

    (sensor, knet) = keynet.system.Keynet(inshape, net, global_photometric='uniform_random_gain', beta=1.0)
    assert np.allclose(knet.forward(sensor.fromtensor(x).encrypt().astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet]:  Analog Gain Keynet  -  PASSED')

    (sensor, knet) = keynet.system.Keynet(inshape, net, global_photometric='uniform_random_bias', gamma=1.0)
    assert np.allclose(knet.forward(sensor.fromtensor(x).encrypt().astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('[test_keynet]:  Analog Bias Keynet  -  PASSED')

    (sensor, knet) = keynet.system.Keynet(inshape, net, global_photometric='uniform_random_affine', beta=1.0, gamma=1.0)
    assert np.allclose(knet.forward(sensor.fromtensor(x).encrypt().astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-4)
    print('[test_keynet]:  Analog Affine Keynet  -  PASSED')
        

def test_vgg16_identity():

    inshape = (3,224,224)
    x = torch.randn(1, *inshape)
    net = keynet.vgg.VGG16()

    print('vgg16: num parameters=%d' % keynet.torch.count_parameters(net))
    (sensor, knet) = keynet.system.IdentityKeynet(inshape, net)

    yh = knet.forward(sensor.fromtensor(x).encrypt().astensor()).detach().numpy().flatten()
    y = net.forward(x).detach().numpy().flatten()
    assert np.allclose(yh, y, atol=1E-3)
    print('vgg16: keynet-56 num parameters=%d' % knet.num_parameters())


def test_vgg16_identity_tiled():

    inshape = (3,224,224)
    x = torch.randn(1, *inshape)
    net = keynet.vgg.VGG16()

    print('vgg16: num parameters=%d' % keynet.torch.count_parameters(net))
    (sensor, knet) = keynet.system.TiledIdentityKeynet(inshape, net, 224//4)
    print(vipy.util.save((sensor, knet), 'test_vgg16.pkl'))
    #(sensor, knet) = vipy.util.load('test_vgg16.pkl')

    yh = knet.forward(sensor.fromtensor(x).encrypt().astensor()).detach().numpy().flatten()
    y = net.forward(x).detach().numpy().flatten()
    
    assert np.allclose(yh, y, atol=1E-3)
    print('vgg16: keynet-56 num parameters=%d' % knet.num_parameters())


def test_vgg16_stochastic():
    inshape = (3,224,224)
    x = torch.randn(1, *inshape)
    net = keynet.vgg.VGG16()
    print('vgg16: num parameters=%d' % keynet.torch.count_parameters(net))

    keynet.globals.num_processes(48)
    (sensor, knet) = keynet.system.Keynet(inshape, net, tileshape=(224//16, 224//16), 
                                          global_geometric='hierarchical_permutation', hierarchical_blockshape=(2,2), hierarchical_permute_at_level=(0,1,2),
                                          local_geometric='doubly_stochastic', alpha=2.0, blocksize=224//16,
                                          local_photometric='uniform_random_affine', beta=1.0, gamma=1.0,
                                          memoryorder='channel')
                                          
    assert np.allclose(knet.forward(sensor.fromtensor(x).encrypt().astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('vgg16: keynet-orthogonal-56 num parameters=%d' % knet.num_parameters())


def test_vgg16_orthogonal():
    inshape = (3,224,224)
    x = torch.randn(1, *inshape)
    net = keynet.vgg.VGG16()
    print('vgg16: num parameters=%d' % keynet.torch.count_parameters(net))

    (sensor, knet) = keynet.system.Keynet(inshape, net, tileshape=(224//16, 224//16), 
                                          global_geometric='identity', hierarchical_blockshape=(2,2), hierarchical_permute_at_level=(0,1,2),
                                          local_geometric='givens_orthogonal', alpha=2.0, blocksize=224//16,
                                          local_photometric='uniform_random_affine', beta=1.0, gamma=1.0,
                                          memoryorder='channel')
    print(vipy.util.save((sensor, knet, net), 'test_vgg16_orthogonal.pkl'))
                                          
    #(sensor, knet) = vipy.util.load('test_vgg16_orthogonal.pkl')
    yh = knet.forward(sensor.fromtensor(x).encrypt().astensor()).detach().numpy().flatten()
    y = net.forward(x).detach().numpy().flatten()    
    print(y)
    print(yh)
    assert np.allclose(yh, y, atol=1E-3)
    #assert np.allclose(knet.forward(sensor.fromtensor(x).encrypt().astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('vgg16: keynet-orthogonal-56 num parameters=%d' % knet.num_parameters())

def test_vgg16_orthogonal_8():
    inshape = (3,224,224)
    x = torch.randn(1, *inshape)
    net = keynet.vgg.VGG16()
    print('vgg16: num parameters=%d' % keynet.torch.count_parameters(net))

    (sensor, knet) = keynet.system.Keynet(inshape, net, tileshape=(224//8, 224//8), 
                                          global_geometric='identity', hierarchical_blockshape=(2,2), hierarchical_permute_at_level=(0,1,2),
                                          local_geometric='givens_orthogonal', alpha=2.0, blocksize=224//8,
                                          local_photometric='uniform_random_affine', beta=1.0, gamma=1.0,
                                          memoryorder='channel')
    print(vipy.util.save((sensor, knet, net), 'test_vgg16_orthogonal_8.pkl'))
                                          
    #(sensor, knet) = vipy.util.load('test_vgg16_orthogonal_4.pkl')
    yh = knet.forward(sensor.fromtensor(x).encrypt().astensor()).detach().numpy().flatten()
    y = net.forward(x).detach().numpy().flatten()    
    print(y)
    print(yh)
    assert np.allclose(yh, y, atol=1E-3)
    #assert np.allclose(knet.forward(sensor.fromtensor(x).encrypt().astensor()).detach().numpy().flatten(), net.forward(x).detach().numpy().flatten(), atol=1E-5)
    print('vgg16: keynet-orthogonal-56 num parameters=%d' % knet.num_parameters())


def test_lenet_orthogonal():
    inshape = (1,28,28)
    x = torch.randn(1, *inshape)
    net = keynet.mnist.LeNet_AvgPool()
    print('lenet: num parameters=%d' % keynet.torch.count_parameters(net))

    (sensor, knet) = keynet.system.Keynet(inshape, net, tileshape=None, 
                                          global_geometric='hierarchical_rotation', hierarchical_blockshape=(2,2), hierarchical_permute_at_level=(0),
                                          global_photometric='uniform_random_bias', 
                                          local_geometric='givens_orthogonal', alpha=2.0, blocksize=8,
                                          local_photometric='uniform_random_affine', beta=1.0, gamma=1.0,
                                          memoryorder='block')

    print(vipy.util.save((sensor, knet), 'test_lenet_orthogonal.pkl'))
    yh = knet.forward(sensor.fromtensor(x).encrypt().astensor()).detach().numpy().flatten()
    y = net.forward(x).detach().numpy().flatten()    
    print(y)
    print(yh)
    assert np.allclose(y, yh, atol=1E-5)
    print('lenet: keynet-orthogonal-8 num parameters=%d' % knet.num_parameters())


def test_lenet_orthogonal_tiled():
    inshape = (1,28,28)
    x = torch.randn(1, *inshape)
    net = keynet.mnist.LeNet_AvgPool()
    print('lenet: num parameters=%d' % keynet.torch.count_parameters(net))

    (sensor, knet) = keynet.system.Keynet(inshape, net, tileshape=(4,4), 
                                          global_geometric='hierarchical_permutation', hierarchical_blockshape=(2,2), hierarchical_permute_at_level=(0,1),
                                          global_photometric='identity',
                                          local_geometric='givens_orthogonal', alpha=2.0, blocksize=4,
                                          local_photometric='uniform_random_affine', beta=1.0, gamma=1.0,
                                          memoryorder='block')

    yh = knet.forward(sensor.fromtensor(x).encrypt().astensor()).detach().numpy().flatten()
    y = net.forward(x).detach().numpy().flatten()
    
    print(y)
    print(yh)
    assert np.allclose(y, yh, atol=1E-5)
    print('lenet-keyed: orthogonal-tiled-4 num parameters=%d' % knet.num_parameters())


def test_allconvnet_orthogonal_tiled():
    inshape = (3,32,32)
    x = torch.randn(1, *inshape)
    net = keynet.cifar10.AllConvNet(batchnorm=False)
    net.eval()  
    print('allconvnet: num parameters=%d' % keynet.torch.count_parameters(net))

    (sensor, knet) = keynet.system.Keynet(inshape, net, tileshape=(8,8), 
                                          global_geometric='hierarchical_permutation', hierarchical_blockshape=(2,2), hierarchical_permute_at_level=(0,1),
                                          global_photometric='identity',
                                          local_geometric='givens_orthogonal', alpha=8, blocksize=8,
                                          local_photometric='uniform_random_affine', beta=1.0, gamma=1.0,
                                          memoryorder='block')

    yh = knet.forward(sensor.fromtensor(x).encrypt().astensor()).detach().numpy().flatten()
    y = net.forward(x).detach().numpy().flatten()    
    print(y, yh)
    assert np.allclose(y, yh, atol=1E-5)
    print('allconvnet-keyed:  orthogonal-tiled-8 num parameters=%d' % knet.num_parameters())


def test_allconvnet_identity(tiled=False):
    inshape = (3,32,32)
    x = torch.randn(1, *inshape)
    net = keynet.cifar10.AllConvNet(batchnorm=True)
    net.eval()  
    print('allconvnet: num parameters=%d' % keynet.torch.count_parameters(net))

    (sensor, knet) = keynet.system.Keynet(inshape, net, tileshape=None if not tiled else (8,8), 
                                          global_geometric='identity', hierarchical_blockshape=(2,2), hierarchical_permute_at_level=(0,1),
                                          global_photometric='identity',
                                          local_geometric='identity', alpha=2.0, blocksize=8,
                                          local_photometric='identity', beta=1.0, gamma=1.0,
                                          memoryorder='channel')

    yh = knet.forward(sensor.fromtensor(x).encrypt().astensor()).detach().numpy().flatten()
    y = net.forward(x).detach().numpy().flatten()    
    print(y, yh)
    assert np.allclose(y, yh, atol=1E-5)
    print('allconvnet-keyed:  identity%s num parameters=%d' % ('-tiled' if tiled else '', knet.num_parameters()))


if __name__ == '__main__':
    
    if len(sys.argv) == 1:
        test_tiled_keynet()
        test_identity_keynet()
        test_permutation_keynet()
        test_photometric_keynet()

        test_lenet_orthogonal()
        test_lenet_orthogonal_tiled()

        test_allconvnet_identity()
        test_allconvnet_identity(tiled=True)
        test_allconvnet_orthogonal_tiled()

    elif sys.argv[1] == 'vgg16-identity-tiled':
        test_vgg16_identity_tiled()
    elif sys.argv[1] == 'vgg16-identity':
        test_vgg16_identity()
    elif sys.argv[1] == 'vgg16-orthogonal-8':
        test_vgg16_orthogonal_8()
    elif sys.argv[1] == 'lenet-orthogonal-tiled':
        test_lenet_orthogonal_tiled()
    elif sys.argv[1] == 'lenet-orthogonal':
        test_lenet_orthogonal()
    elif sys.argv[1] == 'allconvnet-orthogonal-tiled':
        test_allconvnet_orthogonal_tiled()
    elif sys.argv[1] == 'allconvnet-identity':
        test_allconvnet_identity()
    elif sys.argv[1] == 'allconvnet-identity-tiled':
        test_allconvnet_identity(tiled=True)
    else:
        raise ValueError('unknown option "%s"' % sys.argv[1])

