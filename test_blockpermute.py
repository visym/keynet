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
from keynet.blockpermute import block_permutation_mask, block_permute, local_permutation_mask, global_permutation_mask
from keynet.util import savetemp, imshow


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
    
    
def test_image_permute(imgfile):
    img = np.array(PIL.Image.open(imgfile).resize( (256,256) ))        
    mask = block_permutation_mask(256, 4, minscale=4)
    img_permuted = block_permute(img, mask)
    imshow(img)
    imshow(img_permuted)


def test_vgg16_permute(imgfile):
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
    

def test_256x256_local_block_permutation(imgfile='owl.jpg'):
    # The mask should have a subblock size of 2x2, with six levels.  The top levels are not permuted, the bottom level is permuted
    img = np.array(PIL.Image.open(imgfile).resize( (256,256) ))
    mask = local_permutation_mask(256, 2, minscale=4, identityscale=5)
    img_permuted = block_permute(img, mask)
    imshow(img_permuted)

def test_256x256_global_block_permutation(imgfile='owl.jpg'):
    # The mask should have a subblock size of 2x2, with two levels.  The top level is permuted, the second level is not permuted
    # This is equivalent to the block_permutation with one level
    img = np.array(PIL.Image.open(imgfile).resize( (256,256) ))
    mask = global_permutation_mask(256, 2, minscale=7, identityscale=8)
    img_permuted = block_permute(img, mask)
    imshow(img_permuted)
    
def test_vgg16_local_block_permutation(imgfile='owl.jpg'):
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
    

