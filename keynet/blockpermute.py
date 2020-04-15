import scipy.sparse
import numpy as np
from keynet.util import find_closest_positive_divisor


def block_permute(img, cropshape, seed=None):
    """For every non-overlapping subimg in img of size cropshape=(H,W), randomly permute the blocks, preserving the order and channels within the block.
       This assumes that img.shape == (H,W,C)
    """
    assert img.shape[0] % cropshape[0] == 0 and img.shape[1] % cropshape[1] == 0, "Blocksize must be evenly divisible with image shape"
    if seed is not None:
        np.random.seed(seed)
    U = np.random.permutation(np.arange(0, img.shape[0], cropshape[0]))
    V = np.random.permutation(np.arange(0, img.shape[1], cropshape[1]))
    img_permuted = np.copy(img)
    for (i,i_perm) in zip(np.arange(0, img.shape[0], cropshape[0]), U):
        for (j,j_perm) in zip(np.arange(0, img.shape[1], cropshape[1]), V):
            img_permuted[i_perm:i_perm+cropshape[0], j_perm:j_perm+cropshape[1]] = img[i:i+cropshape[0], j:j+cropshape[1]]
    return img_permuted


def hierarchical_block_permute(img, blockshape, permute_at_level, min_blocksize=8, seed=None, twist=False, strict=True):
    """Generate a top-down, hierarchical block permutation
    
       input:
         -img:  The HxWxC image to permute
         -blockshape:  A tuple (N,M) such that each level is decomposed into NxM=(rows,cols) blocks, this must be equally divisible with image size at all levels of hierarchy
         -permute_at_level: array of maximum length log2(min(img.shape))) with entries permute_at_level=[0,2] if the image at level k is permuted.  k=0 is the full size image
         -twist:  restrict the permutation at each level to a rotation only
         -min_blocksize:  the smallest dimension of any block
    """

    if len(permute_at_level) == 0 or blockshape == img.shape:
        return np.copy(img)

    if (img.shape[0] % blockshape[0] != 0 and img.shape[1] % blockshape[1] != 0):
        if strict:
            raise ValueError("Recursive image size %s and block layout %s must be divisible" % (str(img.shape[0:2]), str(blockshape)))
        else:
            # Try to set the closest blockshape that is evenly divisible with img.shape
            new_blockshape = (find_closest_positive_divisor(img.shape[0], blockshape[0]), find_closest_positive_divisor(img.shape[1], blockshape[1]))
            print('[keynet.blockpermute]: Ragged blockshape %s for image size %s, since strict=false, setting to %s' % (str(blockshape), str(img.shape[0:2]), str(new_blockshape)))        
            blockshape = new_blockshape

    imgsize = (img.shape[0], img.shape[1])
    cropshape = (img.shape[0] // blockshape[0], img.shape[1] // blockshape[1])
    levels = int(np.log2(min(imgsize)))
    img_permuted = np.copy(img)
    if seed is not None:
        np.random.seed(seed)
    if 0 in permute_at_level:
        if twist:
            twist_direction = 1 if np.random.rand()>0.5 else 3
            img_permuted = np.rot90(img_permuted, k=twist_direction)
        else:
            img_permuted = block_permute(img_permuted, cropshape, seed=None)  # seed must be None

    if len(permute_at_level)==1 and permute_at_level[0] == 0:
        return img_permuted
    for i in range(0, img.shape[0], cropshape[0]):
        for j in range(0, img.shape[1], cropshape[1]):
            subimg = img_permuted[i:i+cropshape[0], j:j+cropshape[1]]
            if min(cropshape) >= min_blocksize and max(permute_at_level) > 0:
                subimg_permuted = hierarchical_block_permute(subimg, blockshape, permute_at_level=np.array(permute_at_level)-1, seed=None, min_blocksize=min_blocksize, twist=twist)  # seed must be None
                img_permuted[i:i+cropshape[0], j:j+cropshape[1]] = subimg_permuted
            elif max(permute_at_level) > 0:
                raise ValueError('Recursive blockshape=%s < minimum blockshape=%d' % (subimg.shape[0:2], min_blocksize)) 
    return img_permuted


def hierarchical_block_permutation_matrix(imgshape, blockshape, permute_at_level, min_blocksize=8, seed=None, twist=False, withinverse=False, strict=True):
    """Given an image with shape=(HxWxC) return the permutation matrix P such that P.dot(img.flatten()).reshape(imgshape) == hierarchical_block_permute(img, ...)"""
    img = np.array(range(0, np.prod(imgshape))).reshape(imgshape)
    img_permute = hierarchical_block_permute(img, blockshape, permute_at_level, min_blocksize, seed=seed, twist=twist, strict=strict)
    cols = img_permute.flatten()    
    rows = range(0, len(cols))
    vals = np.ones_like(rows)
    P = scipy.sparse.coo_matrix( (vals, (rows, cols)), shape=(np.prod(img.shape), np.prod(img.shape)), dtype=np.float32)
    return P if not withinverse else (P, P.transpose())

