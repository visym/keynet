import scipy.sparse
import numpy as np


def block_permute(img, blockshape, seed=None):
    """For every non-overlapping block in img of size (H,W)=blockshape, randomly permute the blocks, preserving the order within the block"""
    assert img.shape[0] % blockshape[0] == 0 and img.shape[1] % blockshape[1] == 0, "Blocksize must be evenly divisible with image shape"
    U = np.random.RandomState(seed=seed).permutation(np.arange(0, img.shape[0], blockshape[0]))
    V = np.random.RandomState(seed=seed).permutation(np.arange(0, img.shape[1], blockshape[1]))
    img_permuted = np.copy(img)
    for (i,i_perm) in zip(np.arange(0, img.shape[0], blockshape[0]), U):
        for (j,j_perm) in zip(np.arange(0, img.shape[1], blockshape[1]), V):
            img_permuted[i_perm:i_perm+blockshape[0], j_perm:j_perm+blockshape[1]] = img[i:i+blockshape[0], j:j+blockshape[1]]
    return img_permuted


def hierarchical_block_permute(img, num_blocks, permute_at_level, min_blockshape=8, seed=None):
    """Generate a top-down, hierarchical block permutation
    
       input:
         -img:  The HxWxC image to permute
         -num_blocks:  A tuple (N,M) such that each level is decomposed into NxM=(rows,cols) blocks, this must be equally divisible with image size at all levels of hierarchy
         -permute_at_level: array of length log2(min(img.shape))) with entries permute_at_level[k]=[true|false] if the image at level k is permuted.  k=0 is the full size image

    """

    assert img.shape[0] % num_blocks[0] == 0 and img.shape[1] % num_blocks[1] == 0, "Recursive image size %s and block layout %s must be divisible" % (str(img.shape[0:2]), str(num_blocks))
    imgsize = (img.shape[0], img.shape[1])
    blockshape = (img.shape[0] // num_blocks[0], img.shape[1] // num_blocks[1])
    levels = int(np.log2(min(imgsize)))
    img_permuted = np.copy(img)
    if 0 in permute_at_level:
        img_permuted = block_permute(img_permuted, blockshape, seed=seed)        
    for i in range(0, img.shape[0], blockshape[0]):
        for j in range(0, img.shape[1], blockshape[1]):
            subimg = img_permuted[i:i+blockshape[0], j:j+blockshape[1]]
            if min(blockshape) > min_blockshape:
                subimg_permuted = hierarchical_block_permute(subimg, num_blocks, permute_at_level=np.array(permute_at_level)-1, seed=seed, min_blockshape=min_blockshape)
                img_permuted[i:i+blockshape[0], j:j+blockshape[1]] = subimg_permuted
            elif max(permute_at_level) > 0:
                raise ValueError('Recusrive blockshape=%s <= minimum blockshape=%d' % (subimg.shape[0:2], min_blockshape)) 
    return img_permuted


def hierarchical_block_permutation_matrix(imgshape, num_blocks, permute_at_level, min_blockshape=8, seed=None):
    """Given an image with shape=(HxWxC) return the permutation matrix P such that P.dot(img.flatten()).reshape(imgshape) == hierarchical_block_permute(img, ...)"""
    img = np.array(range(0, np.prod(imgshape))).reshape(imgshape)
    img_permute = hierarchical_block_permute(img, num_blocks, permute_at_level, min_blockshape, seed=seed)
    cols = img_permute.flatten()    
    rows = range(0, len(cols))
    vals = np.ones_like(rows)
    return scipy.sparse.coo_matrix( (vals, (rows, cols)), shape=(np.prod(img.shape), np.prod(img.shape)), dtype=np.float32)

