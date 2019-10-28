from numpy.linalg import multi_dot 
import numpy as np
import math
import PIL
from PIL import Image


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


