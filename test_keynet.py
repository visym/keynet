from numpy.linalg import multi_dot 
import numpy as np
import scipy.linalg
import math
import PIL
from PIL import Image
import os
import uuid
import sys
import copy
import scipy.signal
import scipy.ndimage
import torch.nn.functional as F
import torch 
import vipy.image  # bash setup
import vipy.visualize  # bash setup
from keynet.util import sparse_permutation_matrix, sparse_identity_matrix

def test_generalized_permutation():
    np.set_printoptions(precision=2)

    # Symmetric block diagonal convolutional toeplitz matrix
    #W = scipy.linalg.block_diag(*[np.random.rand(2,2) for k in range(0,3)])
    #W = W + W.transpose() + 1E-2
    W = scipy.linalg.block_diag(*[np.ones( (2,2) )-1E-2 for k in range(0,3)]) + 1E-2
    
    # Permutation matrix
    P = np.eye(6)
    #P = np.random.permutation(P)
    
    # Gain matrix
    #G = np.diag(10000*np.random.rand(6))
    G = np.diag([1, 100, 1, 100, 1, 100])

    # A=G^{-1} * P^{-1} * W * P * G
    print(P)
    print(W)
    print(G)
    A = multi_dot( (np.linalg.inv(G), P.transpose(), W, P, G) ) 

    print(A)
    


def _block_permute(img, blocksize, permsize):
    """For every non-overlapping block in img of size (blocksize x blocksize), randomly permute the (permsize x permsize) subblocks within this block, preserving order of elements within subblock"""
    assert(blocksize % permsize == 0)
    for i in np.arange(0, img.shape[0], blocksize):
        for j in np.arange(0, img.shape[1], blocksize):
            subimg = np.copy(img[i:i+blocksize, j:j+blocksize])
            subblocksize = blocksize // permsize
            (U,V) = (np.arange(0, blocksize, subblocksize), np.arange(0, blocksize, subblocksize))
            for (u, u_perm) in zip(U, np.random.permutation(U)):
                for (v, v_perm) in zip(V, np.random.permutation(V)):
                    img[i+u:i+u+subblocksize, j+v:j+v+subblocksize] = subimg[u_perm:u_perm+subblocksize, v_perm:v_perm+subblocksize]
    return img


def block_permutation_mask(n, m, minscale=3):
    """generate an nxn top-down, hierarchical block permutation mask of size mxm down to level index minscale"""
    assert(n % m == 0)
    mask = np.arange(0,n*n).reshape( (n,n) ).astype(np.uint32)
    maxscale = int(np.ceil(math.log(n,m)))
    for k in reversed(np.arange(minscale, maxscale+1)):
        mask = _block_permute(mask, np.power(m,k), m)
    return mask


def local_permutation_mask(n, m, minscale, identityscale):
    """generate an nxn top-down, hierarchical block permutation mask of size mxm down to level index minscale that preserves global block structure for scales above identityscale"""
    assert(n % m == 0)
    mask = np.arange(0,n*n).reshape( (n,n) ).astype(np.uint32)
    maxscale = int(np.ceil(math.log(n,m)))
    assert(minscale<=maxscale and identityscale<=maxscale)    
    for k in reversed(np.arange(minscale, maxscale+1)):
        if k < identityscale:
            mask = _block_permute(mask, np.power(m,k), m)
    return mask


def identity_permutation_mask(n):
    return np.arange(0,n*n).reshape( (n,n) ).astype(np.uint32)


def global_permutation_mask(n, m, minscale, identityscale):
    """generate an nxn top-down, hierarchical block permutation mask of size mxm down to level index minscale that preserves global block structure for scales above identityscale"""
    assert(n % m == 0)
    mask = np.arange(0,n*n).reshape( (n,n) ).astype(np.uint32)
    maxscale = int(np.ceil(math.log(n,m)))
    assert(minscale<=maxscale and identityscale<=maxscale)
    for k in reversed(np.arange(minscale, maxscale+1)):
        if k >= identityscale:
            mask = _block_permute(mask, np.power(m,k), m)
    return mask


def block_permute(img, mask):
    """Apply permutation mask to color image"""
    assert(len(img.shape) == 3 and len(mask.shape) == 2 and img.shape[0:2] == mask.shape[0:2])
    img_permuted = np.copy(img)
    for c in range(img.shape[2]):
        img_channel = img[:,:,c]
        img_permuted[:,:,c] = img_channel.ravel()[np.argsort(mask.ravel())].reshape( img_channel.shape )
    return img_permuted


def imshow(img):
    f = '/tmp/%s.png' % uuid.uuid1().hex
    im = PIL.Image.fromarray(img.astype(np.uint8)).save(f)
    os.system('open %s' % f)


def savetemp(img):
    f = '/tmp/%s.png' % uuid.uuid1().hex
    PIL.Image.fromarray(img.astype(np.uint8)).save(f)
    return f

def test_checkerboard_permute():
    img = np.uint8(255*np.random.rand(8,8,3))
    img = np.array(PIL.Image.fromarray(img).resize( (256,256), PIL.Image.NEAREST))

    mask = block_permutation_mask(256,2, minscale=8)
    img_permuted = block_permute(img, mask)
    imshow(img)
    imshow(img_permuted)

def test_corner_permute():
    img = np.zeros( (243,243,3) ).astype(np.uint8)
    img[0:3,0:3] = 64
    img[0:3,-3:] = 128
    img[-3:,0:3] = 196
    img[-3:,-3:] = 255    

    mask = block_permutation_mask(243,3)
    img_permuted = block_permute(img, mask)
    imshow(img)
    imshow(img_permuted)
    
    
def test_image_permute(imgfile="/Users/jeffrey.byrne/Desktop/visym_logo_whitebg_notext.png"):
    img = np.array(PIL.Image.open(imgfile).resize( (256,256) ))
        
    mask = block_permutation_mask(256, 4, minscale=4)
    img_permuted = block_permute(img, mask)
    imshow(img)
    imshow(img_permuted)

def test_vgg16_permute(imgfile="/Users/jeffrey.byrne/Desktop/visym_logo_whitebg_notext.png"):
    # The input image must be 243x243, you will need to center crop your 256x256 images to 243x243
    img = np.array(PIL.Image.open(imgfile).resize( (243,243) ))

    # The mask should have a subblock size of three to match the VGG-16 conv layer kernel size
    # This mask should be generated *once* and reused for all training
    mask = block_permutation_mask(243, 3, minscale=3)

    # The permuted image should be used for training
    img_permuted = block_permute(img, mask)

    # You will need to center crop 224x224 from the resulting permuted 243x243 image.  This does introduce boundary artifacts.
    # The boundary artifacts will need to be addressed in a custom block permutation so that we do not crop away important patches
    img_cropped = img_permuted[9:243-10, 9:243-10, :]
    assert(img_cropped.shape[0:2] == (224,224) )
    return img_cropped
    

def test_256x256_local_block_permutation(imgfile="/Users/jeffrey.byrne/Desktop/jebyrne.jpg"):
    # The mask should have a subblock size of 2x2, with six levels.  The top levels are not permuted, the bottom level is permuted
    img = np.array(PIL.Image.open(imgfile).resize( (256,256) ))
    mask = local_permutation_mask(256, 2, minscale=4, identityscale=5)
    img_permuted = block_permute(img, mask)
    imshow(img_permuted)

def test_256x256_global_block_permutation(imgfile="/Users/jeffrey.byrne/Desktop/jebyrne.jpg"):
    # The mask should have a subblock size of 2x2, with two levels.  The top level is permuted, the second level is not permuted
    # This is equivalent to the block_permutation with one level
    img = np.array(PIL.Image.open(imgfile).resize( (256,256) ))
    mask = global_permutation_mask(256, 2, minscale=7, identityscale=8)
    img_permuted = block_permute(img, mask)
    imshow(img_permuted)
    
def test_vgg16_local_block_permutation(imgfile="/Users/jeffrey.byrne/Desktop/jebyrne.jpg"):
    img = np.array(PIL.Image.open(imgfile).resize( (243,243) ))
    mask = local_permutation_mask(243, 3, minscale=3, identityscale=4)

    # The permuted image should be used for training
    img_permuted = block_permute(img, mask)

    # You will need to center crop 224x224 from the resulting permuted 243x243 image.  This does introduce boundary artifacts.
    # The boundary artifacts will need to be addressed in a custom block permutation so that we do not crop away important patches
    img_cropped = img_permuted[9:243-10, 9:243-10, :]
    assert(img_cropped.shape[0:2] == (224,224) )
    #imshow(img_permuted)
    #return img_cropped
    

def random_permutation_matrix(n):
    A = np.eye(n)
    A = np.random.permutation(A)
    return A

def random_doubly_stochastic_matrix(n,k):
    A = np.random.rand()*random_permutation_matrix(n)
    for k in range(0,k):
        A = A + np.random.rand()*random_permutation_matrix(n)
    for k in range(0,100):
        A = A / np.sum(A,axis=0)
        A = A / np.sum(A,axis=1)        
    return A

def gaussian_random_diagonal_matrix(n,sigma=1):
    d = sigma*np.random.randn(n)
    return (np.diag(d))

def uniform_random_diagonal_matrix(n,scale=1,eps=1E-6):
    d = scale*np.random.rand(n) + eps
    return (np.diag(d))

def relu(x):
    xh = copy.deepcopy(x)
    xh[xh<=0] = 0
    return xh

def example_2x2():

    np.random.seed(0)

    img = np.array([[11,12],[21,22]]).astype(np.float32)
    x = img.flatten().reshape(4,1)
    D1 = uniform_random_diagonal_matrix(4)
    P1 = random_doubly_stochastic_matrix(4,2)
    A1 = np.dot(D1,P1)
    A1inv = np.linalg.inv(A1)

    P2 = random_permutation_matrix(4)
    D2 = uniform_random_diagonal_matrix(4)
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
    x2h = relu(x1h)

    x1 = multi_dot( (W1, x) )
    x2 = relu(x1)

    print('A1*x, x1, x1h, x2, x2h, A2inv * x2h')
    print(multi_dot( (A1, x) ))
    print(x1)
    print(x1h)

    print(x2)
    print(x2h)
    print(multi_dot( (A2inv, x2h) ))


def checkerboard_256x256():
    img = np.uint8(255*np.random.rand(8,8,3))
    img = np.array(PIL.Image.fromarray(img).resize( (256,256), PIL.Image.NEAREST))
    return img


def optical_transformation_montage():
    (m,n) = (256,256)
    img = np.array(PIL.Image.open('/proj/enigma/jebyrne/jebyrne_cropped.jpg').resize( (256,256) ))

    D = [np.maximum(1E-6, 1.0 + (s*np.random.rand( m,n,3 )-(s/2.0))) for s in [0.1, 1.0, 10000.0]]
    B = [255*np.maximum(1E-6, s*np.random.rand( m,n,3 )) for s in [0.1, 1.0, 10000.0]]
    P = [identity_permutation_mask(256),
         local_permutation_mask(256, 2, minscale=2, identityscale=3), 
         local_permutation_mask(256, 2, minscale=3, identityscale=4), 
         local_permutation_mask(256, 2, minscale=4, identityscale=5), 
         local_permutation_mask(256, 2, minscale=4, identityscale=6), 
         local_permutation_mask(256, 2, minscale=4, identityscale=7), 
         local_permutation_mask(256, 2, minscale=4, identityscale=8)]
         #local_permutation_mask(256, 2, minscale=3, identityscale=8),
         #local_permutation_mask(256, 2, minscale=2, identityscale=8)]
         #global_permutation_mask(256, 2, minscale=7, identityscale=8)]
         
    imlist = []
    for (d,b) in zip(D,B):
        for p in P:
            img_permuted = block_permute(img, p)
            img_scaled = np.multiply(d, img_permuted) + b
            img_scaled = np.uint8(255*((img_scaled-np.min(img_scaled))/ (np.max(img_scaled)-np.min(img_scaled))))
            imlist.append(vipy.image.Image(array=img_scaled))

    img_montage = vipy.visualize.montage(imlist, 256,256, rows=3, cols=7, grayscale=False, skip=False, border=1)

    print(savetemp(img_montage))
    return img_montage



def test_affine_augmentation():
    import torch
    from keynet.util import torch_affine_augmentation_tensor, torch_affine_deaugmentation_tensor
    (N,C,U,V) = (2,2,3,3)
    x = torch.tensor(np.random.rand( N,C,U,V ).astype(np.float32))    
    x_affine = torch_affine_augmentation_tensor(x)
    x_deaffine = torch_affine_deaugmentation_tensor(x_affine).reshape( N,C,U,V )
    assert(np.allclose(x, x_deaffine))
    print('Affine augmentation (round-trip): passed')    
    
    x_affine_numpy = np.hstack((np.array(x).reshape(N, C*U*V), np.ones( (N,1) )))
    assert(np.allclose(x_affine, x_affine_numpy))
    print('Affine augmentation (torch vs. numpy): passed')    
    


def test_sparse_toeplitz_conv2d():
    from keynet.util import sparse_toeplitz_conv2d, torch_conv2d_in_scipy, torch_affine_augmentation_tensor

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
    assert(np.allclose(y_scipy, yh))
    print('Correlation (scipy vs. toeplitz): passed')    

    # Torch spatial correlation: reshape torch to be tensor sized [BATCH x CHANNEL x HEIGHT x WIDTH]
    # Padding required to allow for valid convolution to be the same size as input
    y_torch = F.conv2d(torch.tensor(img), torch.tensor(f), bias=torch.tensor(b), padding=((P-1)//2, (Q-1)//2), stride=stride)
    assert(np.allclose(y_torch,yh))
    print('Correlation (torch vs. toeplitz): passed')

    return(T)


def test_sparse_toeplitz_avgpool2d():
    from keynet.util import sparse_toeplitz_avgpool2d, torch_avgpool2d_in_scipy

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
    print('Average pool 2D (scipy vs. toeplitz): passed')    

    # Torch avgpool
    y_torch = F.avg_pool2d(torch.tensor(img), kernelsize, stride=stride, padding=((P-1)//2, (Q-1)//2))
    assert(np.allclose(y_torch,yh))
    print('Average pool 2D (torch vs. toeplitz): passed')
    return T


def test_keynet_mnist():
    import mnist

    keynet = mnist.KeyNet()
    keynet.load_state_dict_keyed(torch.load('/proj/enigma/jebyrne/mnist_avgpool.pth'), A0inv=sparse_identity_matrix(28*28*1 + 1))
    keynet.eval()    

    net = mnist.LeNet_AvgPool()
    net.load_state_dict(torch.load('/proj/enigma/jebyrne/mnist_avgpool.pth'))
    net.eval()

    torch.set_grad_enabled(False)
    x = torch.tensor(np.random.rand(1,1,28,28).astype(np.float32))
    y = net(x)
    yh = keynet(x)
    
    assert(np.allclose(np.array(y), np.array(yh)))
    print('MNIST keynet: passed')
    








