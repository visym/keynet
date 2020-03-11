import numpy as np
import vipy
import warnings
import torch
from keynet.sparse import sparse_permutation_matrix_with_inverse, sparse_permutation_matrix, sparse_generalized_permutation_block_matrix_with_inverse, sparse_identity_matrix
from keynet.sparse import sparse_stochastic_matrix_with_inverse, sparse_permutation_tiled_matrix_with_inverse, sparse_identity_tiled_matrix_with_inverse
from keynet.sparse import SparseTiledMatrix
from keynet.torch import homogenize, dehomogenize
from keynet.layer import KeyedLayer
import keynet.fiberbundle
import PIL

class Keysensor(KeyedLayer):
    def __init__(self, inshape, encryptkey, decryptkey, backend):
        super(Keysensor, self).__init__(backend=backend)        
        self._encryptkey = encryptkey
        self._decryptkey = decryptkey
        self._inshape = inshape
        self._tensor = None

    def __repr__(self):
        return str('<Keysensor: height=%d, width=%d, channels=%d>' % (self._inshape[1], self._inshape[2], self._inshape[0]))
    
    def load(self, imgfile):
        im = vipy.image.Image(imgfile).resize(height=self._inshape[1], width=self._inshape[2])
        if self._inshape[0] == 1:
            im = im.grey()
        self._tensor = im.float().torch().contiguous()
        return self

    def tensor(self, x=None):
        if x is None:
            return self._tensor
        else:
            self._tensor = x.clone().type(torch.FloatTensor)
            return self

    def image(self):
        img = dehomogenize(self._tensor).reshape(1, *self._inshape)  if self.isencrypted() else self._tensor  # 1x(C*H*W+1) -> 1xCxHxW
        img = np.squeeze(img.permute(2,3,1,0).numpy())  # 1xCxHxW -> HxWxC
        colorspace = 'float' if img.dtype == np.float32 else None
        colorspace = 'rgb' if img.dtype == np.uint8 and img.shape[2] == 3 else colorspace
        colorspace = 'lum' if img.dtype == np.uint8 and img.shape[2] == 1 else colorspace
        return vipy.image.Image(array=img, colorspace=colorspace)

    def keypair(self):
        return (self._encryptkey, self._decryptkey)

    def key(self):
        return self._decryptkey
    
    def isencrypted(self):
        return self.isloaded() and self._tensor.ndim == 2

    def isloaded(self):
        return self._tensor is not None
    
    def encrypt(self, x_raw=None):
        """img_tensor is NxCxHxW, return Nx(C*H*W+1) homogenized and encrypted"""
        self.tensor(x_raw) 
        assert self.isloaded(), "Load image first"
        self.W = self._encryptkey    # Used in super().forward()       
        self._tensor = super(Keysensor, self).forward(homogenize(self._tensor)) if not self.isencrypted() else self._tensor
        return self
        
    def decrypt(self, x_cipher=None):
        """x_cipher is Nx(C*H*W+1) homogenized, convert to NxCxHxW decrypted"""
        self.tensor(x_cipher)
        assert self.isloaded(), "Load image first"        
        self.W = self._decryptkey   # Used in super().forward()               
        self._tensor = dehomogenize(super(Keysensor, self).forward(self._tensor)).reshape(self._tensor.shape[0], *self._inshape) if self.isencrypted() else self._tensor
        return self


class IdentityKeysensor(Keysensor):
    def __init__(self, inshape, backend='scipy'):
        (encryptkey, decryptkey) = (sparse_identity_matrix(np.prod(inshape)+1), sparse_identity_matrix(np.prod(inshape)+1))        
        super(IdentityKeysensor, self).__init__(inshape, encryptkey, decryptkey, backend)

        
class PermutationKeysensor(Keysensor):
    def __init__(self, inshape, backend='scipy'):
        (encryptkey, decryptkey) = sparse_permutation_matrix_with_inverse(np.prod(inshape)+1)        
        super(PermutationKeysensor, self).__init__(inshape, encryptkey, decryptkey, backend)

        
class StochasticKeysensor(Keysensor):
    def __init__(self, inshape, alpha, backend='scipy'):
        (encryptkey, decryptkey) = sparse_stochastic_matrix_with_inverse(np.prod(inshape)+1, alpha)
        super(StochasticKeysensor, self).__init__(inshape, encryptkey, decryptkey, backend)


class IdentityTiledKeysensor(Keysensor):
    def __init__(self, inshape, tilesize):
        (encryptkey, decryptkey) = sparse_identity_tiled_matrix_with_inverse(np.prod(inshape)+1, tilesize)
        super(IdentityTiledKeysensor, self).__init__(inshape, encryptkey, decryptkey, backend='tiled')

                                    
class PermutationTiledKeysensor(Keysensor):
    def __init__(self, inshape, tilesize):
        (encryptkey, decryptkey) = sparse_permutation_tiled_matrix_with_inverse(np.prod(inshape)+1, tilesize)
        super(PermutationTiledKeysensor, self).__init__(inshape, encryptkey, decryptkey, backend='tiled')
        
        
class OpticalFiberBundle(Keysensor):
    def __init__(self, inshape):
        (encryptkey, decryptkey) = (sparse_identity_matrix(np.prod(inshape)+1), sparse_identity_matrix(np.prod(inshape)+1))
        super(OpticalFiberBundle, self).__init__(inshape, encryptkey, decryptkey, backend='scipy')
    
    def load(self, imgfile)
        img_color = vipy.image.Image(imgfile).maxdim(max(self._inshape)).centercrop(height=self._inshape[1], width=self._inshape[2]).numpy()
        img_sim = keynet.fiberbundle.simulation(img_color, h_xtalk=0.05, v_xtalk=0.05, fiber_core_x=16, fiber_core_y=16, do_camera_noise=True)
        return vipy.image.Image(array=np.uint8(img_sim), colorspace='rgb')
        
