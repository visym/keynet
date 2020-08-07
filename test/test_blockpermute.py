import numpy as np
from keynet.blockpermute import block_permute, hierarchical_block_permute, hierarchical_block_permutation_matrix
import vipy.image
import vipy



def show_corner_block_permutation():
    img = np.zeros( (256,256,3) ).astype(np.uint8)
    img[0:16,0:16,:] = [255,0,0]
    img[0:16,-16:] = [0,255,0]
    img[-16:,0:16] = [0,0,255]
    img[-16:,-16:] = [255,255,255]

    img_permuted = block_permute(img, (64,64) )
    vipy.image.Image(array=img).close().show()
    vipy.image.Image(array=img_permuted).show()
    
    
def show_image_block_permutation(imgfile='./demo/owl.jpg'):
    im = vipy.image.Image(imgfile).resize(512, 512, interp='nearest')
    img_permuted = block_permute(im.array(), (128,128))
    im.close().show()
    vipy.image.Image(array=img_permuted).show()


def show_global_hierarchical_block_permutation(imgfile='./demo/owl.jpg'):
    im = vipy.image.Image(imgfile).resize(512, 512, interp='nearest')
    img_permuted = hierarchical_block_permute(im.array(), blockshape=(2,2), permute_at_level=[0])
    im.close().show()
    vipy.image.Image(array=img_permuted).show()

    
def show_local_hierarchical_block_permutation(imgfile='./demo/owl.jpg'):
    im = vipy.image.Image(imgfile).resize(512, 512, interp='nearest')
    img_permuted = hierarchical_block_permute(im.array(), blockshape=(2,2), permute_at_level=[4,5])
    im.close().show()
    vipy.image.Image(array=img_permuted).show()

    
def show_global_hierarchical_block_permutation_2x2(imgfile='./demo/owl.jpg'):
    im = vipy.image.Image(imgfile).resize(512, 512, interp='nearest')
    img_permuted = hierarchical_block_permute(im.array(), blockshape=(2,2), permute_at_level=[0,1,2,3,4,5])
    im.close().show()
    vipy.image.Image(array=img_permuted).show()
    
    
def show_global_hierarchical_block_permutation_3x3(imgfile='./demo/owl.jpg'):
    im = vipy.image.Image(imgfile).resize(486, 486, interp='nearest')
    img_permuted = hierarchical_block_permute(im.array(), blockshape=(3,3), permute_at_level=[0,1,2,3])
    im.close().show()
    vipy.image.Image(array=img_permuted).show()


def show_global_hierarchical_block_permutation_grey_2x3(imgfile='./demo/owl.jpg'):
    im = vipy.image.Image(imgfile).resize(rows=512, cols=486, interp='nearest').greyscale()
    img_permuted = hierarchical_block_permute(im.array(), blockshape=(2,3), permute_at_level=[0,1,2,3])
    im.close().show()
    vipy.image.Image(array=img_permuted, colorspace='float').show()
    

def test_hierarchical_block_permutation(imgfile='./demo/owl.jpg', show=False, seed=42):
    im = vipy.image.Image(imgfile).resize(256, 256, interp='nearest')
    img_permuted = hierarchical_block_permute(im.array(), blockshape=(2,2), permute_at_level=[0,1], seed=seed)
    P = hierarchical_block_permutation_matrix(im.array().shape, blockshape=(2,2), permute_at_level=[0,1], seed=seed)
    img_permuted_matrix = np.array(P.dot(im.array().flatten()).reshape(img_permuted.shape))
    
    if show:
        vipy.image.Image(array=img_permuted).show()
        vipy.image.Image(array=img_permuted_matrix, colorspace='float').show()
        
    assert np.allclose(img_permuted_matrix.flatten(), img_permuted.flatten(), atol=1E-5)
    print('[test_blockpermute]: hierarchical_block_permutation  PASSED')
    
