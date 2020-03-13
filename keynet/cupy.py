import cupyx
import cupy
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

import keynet.sparse

class SparseMatrix(keynet.sparse.SparseMatrix):
    def __init__(self, A):
        assert self.is_cupy_sparse(A), "Invalid input"
        
        super(SparseMatrixTorch, self).__init__()
        self.shape = A.shape
        self._matrix = A
        self.dtype = A.type
        self.ndim = 2
        
    def from_torch_dense(self, A):
        raise
        
    def from_scipy_sparse(self, A):
        raise
        
    def matmul(self, A):
        raise
        
    def dot(self, x):
        raise
        
    def torchdot(self, x):
        raise
        
    def nnz(self):
        raise
        
    def transpose(self):
        raise
        
    def tocoo(self):
        raise

class SparseTiledMatrix(keynet.sparse.SparseTiledMatrix):
    def __init__(self, tilesize=None, coo_matrix=None, blocktoeplitz=None, shape=None):
        super(SparseTiledMatrix, self).__init__(tilesize, coo_matrix, blocktoeplitz, shape)

    def _block(self, B):
        return keynet.cupy.SparseMatrix(scipy_coo_to_torch_sparse(B))
