import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import torch
import PIL
import uuid
import tempfile
import os
import time
from vipy.util import groupbyasdict
from keynet.util import random_positive_definite_matrix
import copy
from itertools import groupby, product
import tempfile
from vipy.util import Stopwatch
from keynet.torch import scipy_coo_to_torch_sparse
try:
    import cupyx
    import cupy
except:    
    pass  # Exception on init if cupy backend used
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import torch.sparse
import scipy.sparse
from numpy.linalg import multi_dot 


def sparse_block_diag(mats, format='coo'):
    """Create a sparse matrix with elements in mats as blocks on the diagonal"""
    n = len(mats)    
    (rows,cols,data) = ([],[],[])
    (U,V) = (0,0)
    for (k,b) in enumerate(mats):
        b = scipy.sparse.coo_matrix(b)
        for i,j,v in zip(b.row, b.col, b.data):
            rows.append(i+U)
            cols.append(j+V)
            data.append(v)
        (U, V) = (U+b.shape[0], V+b.shape[1])            
    return scipy.sparse.coo_matrix( (data, (rows, cols)), shape=(U, V)).asformat(format)


def sparse_permutation_matrix(n, dtype=np.float32):
    data = np.ones(n).astype(dtype)
    row_ind = list(range(0,n))
    col_ind = np.random.permutation(list(range(0,n)))
    return csr_matrix((data, (row_ind, col_ind)), shape=(n,n))


def sparse_permutation_matrix_with_inverse(n, dtype=np.float32):
    data = np.ones(n).astype(dtype)
    row_ind = list(range(0,n))
    col_ind = np.random.permutation(list(range(0,n)))
    P = csr_matrix((data, (row_ind, col_ind)), shape=(n,n))
    return (P, P.transpose())


def sparse_identity_matrix(n, dtype=np.float32):
    return scipy.sparse.eye(n, dtype=dtype)


def sparse_identity_matrix_like(A):
    return scipy.sparse.eye(A.shape[0], dtype=A.dtype)


def sparse_gaussian_random_diagonal_matrix(n,mu=1,sigma=1,eps=1E-6):
    return scipy.sparse.diags(np.maximum(eps, np.array(sigma*np.random.randn(n)+mu).astype(np.float32)), dtype=np.float32)


def sparse_gaussian_random_diagonal_matrix_with_inverse(n,mu=1,sigma=1,eps=1E-6):
    D = scipy.sparse.diags(np.maximum(eps, np.array(sigma*np.random.randn(n)+mu).astype(np.float32)))
    Dinv = scipy.sparse.diags(1.0 / D.diagonal())
    return (D,Dinv)


def sparse_uniform_random_diagonal_matrix(n, scale=1, eps=1E-6, dtype=np.float32):
    return scipy.sparse.diags(np.array(scale*np.random.rand(n) + eps).astype(dtype))


def sparse_random_doubly_stochastic_matrix(n, k, n_iter=100):
    A = np.random.rand()*sparse_permutation_matrix(n)
    for k in range(0,k):
        A = A + np.random.rand()*sparse_permutation_matrix(n)
    for k in range(0,n_iter):
        A = normalize(A, norm='l1', axis=0)
        A = normalize(A, norm='l1', axis=1)
    return A


def sparse_random_diagonally_dominant_doubly_stochastic_matrix(n, k=None, n_iter=100):
    """Return sparse matrix (nxn) such that at nost k elements per row are non-zero and matrix is positive definite"""
    k = n if k is None else k
    n_iter = 10 if k<=3 else n_iter  # speedup
    d = np.random.rand(k,n)
    d[0,:] = np.maximum(d[0,:], np.sum(d[1:,:], axis=0) + 0.1)  # first column is greater than sum of other columns 
    d = d / np.sum(d,axis=0).reshape(1,n)  # sum over cols
    k_range = list(range(-((k-1)//2), 1+((k-1)//2)) if k%2==1 else list(range(-(k//2), k//2)))
    k_range.remove(0)
    k_range = [0] + k_range  # first row is main diagonal
    A = scipy.sparse.spdiags(d, k_range, n, n, format='csr')
    for k in range(0,n_iter):
        A = normalize(A, norm='l1', axis=0)
        A = normalize(A, norm='l1', axis=1)
    return A
    

def sparse_stochastic_matrix_with_inverse(n, m): 
    """Returns (A,Ainv) for (nxn) sparse matrix of the form P*S*I, where S is block diagonal of size m, and P is permutation"""
    """Setting m=1 results in a permutation matrix"""
    #assert(k<=m and n>=m) 
    m = np.minimum(n,m)

    P = sparse_permutation_matrix(n)
    B = [sparse_random_diagonally_dominant_doubly_stochastic_matrix(m) for j in np.arange(0,n-m,m)]
    B = B + [sparse_random_diagonally_dominant_doubly_stochastic_matrix(n-len(B)*m)]
    S = sparse_block_diag(B, format='csr')
    A = P*S

    Binv = [np.linalg.inv(b.todense()) for b in B]
    Sinv = sparse_block_diag(Binv, format='csr')
    Pinv = P.transpose()
    Ainv = Sinv * Pinv

    return(A.astype(np.float32), Ainv.astype(np.float32)) 


def sparse_generalized_permutation_matrix_with_inverse(n, beta): 
    (A,Ainv) = sparse_permutation_matrix_with_inverse(n)
    (D,Dinv) = sparse_gaussian_random_diagonal_matrix_with_inverse(n, sigma=beta)
    return (D.dot(A), Ainv.dot(Dinv))  # (AB)^{-1} = B^{-1}A^{-1}

def sparse_generalized_stochastic_matrix_with_inverse(n, alpha, beta):
    (A,Ainv) = sparse_stochastic_matrix_with_inverse(n, alpha)
    (D,Dinv) = sparse_gaussian_random_diagonal_matrix_with_inverse(n, mu=1, sigma=beta)
    return (D.dot(A), Ainv.dot(Dinv))  # (AB)^{-1} = B^{-1}A^{-1}
    

def sparse_positive_definite_block_diagonal_with_inverse(n, m, dtype=np.float32):
    m = np.minimum(n,m)
    B = [random_positive_definite_matrix(m,dtype) for j in np.arange(0,n-m,m)]
    B = B + [random_positive_definite_matrix(n-len(B)*m, dtype)]
    A = sparse_block_diag(B, format='csr')
    Binv = [np.linalg.inv(b) for b in B]
    Ainv = sparse_block_diag(Binv, format='csr')
    return(A,Ainv)


def sparse_generalized_stochastic_block_matrix_with_inverse(n, m, dtype=np.float32):
    """Returns (A,Ainv) for (nxn) sparse matrix of the form P*S*D, where D is uniform random diagonal, S is stochastic block matrix of size m, and P is permutation"""
    """Setting m=1 results in scaled permutation matrix"""
    #assert(k<=m and n>=m) 
    m = np.minimum(n, m)

    (M, Minv) = sparse_stochastic_matrix_with_inverse(n,m)
    D = sparse_uniform_random_diagonal_matrix(n)
    A = M*D
    Dinv = sparse_inverse_diagonal_matrix(D)
    Ainv = Dinv * Minv

    return(A.astype(np.float32), Ainv.astype(np.float32))


def sparse_generalized_permutation_block_matrix_with_inverse(n, m, dtype=np.float32):
    """Returns (A,Ainv) for (nxn) sparse matrix of the form B*P*D, where D is uniform random diagonal, B is block diagonal of size m, and P is permutation"""
    """Setting m=k=1 results in scaled permutation matrix"""
    m = np.minimum(n,m)

    (B,Binv) = sparse_positive_definite_block_diagonal_with_inverse(n,m,dtype=np.float64)
    D = sparse_uniform_random_diagonal_matrix(n,dtype=np.float64)
    P = sparse_permutation_matrix(n,dtype=np.float64)
    A = B*P*D
    Dinv = sparse_inverse_diagonal_matrix(D)
    Pinv = P.transpose()
    Ainv = Dinv * Pinv * Binv

    return(A.astype(dtype), Ainv.astype(dtype))


class SparseMatrix(object):
    def __init__(self):
        self._matrix = None
        self.ndim = 2
        self.shape = (0,0)  
        self.dtype = None
        
    def __repr__(self):
        return str('<keynet.SparseMatrix: H=%d, W=%d, backend=%s>' % (self.shape[0], self.shape[1], str(type(self._matrix))))

    def __getitem__(self, k):
        if not (isinstance(k, slice) or isinstance(k, tuple)):
            k = slice(k,k+1)  # force result to be 2D
        return SparseMatrix(self._matrix.__getitem__(k))  # no copy

    def __add__(self, other):
        assert isinstance(other, SparseMatrix), "Invalid input"
        assert self.shape == other.shape, "Invalid shape"
        self._matrix += other._matrix
        return self
        
    def is_torch_sparse(self, x):
        return isinstance(x, torch.sparse.FloatTensor)

    def is_torch_dense(self, x):
        return isinstance(x, torch.FloatTensor)

    def is_scipy_sparse(self, x):
        return scipy.sparse.issparse(x)

    def is_torch(self, x):
        return self.is_torch_sparse(x) or self.is_torch_dense(x)
    
    def is_numpy_dense(self, x):
        return isinstance(x, np.ndarray)

    def is_sparse(self, x):
        return isinstance(x, SparseMatrix)

    def clone(self):
        return copy.deepcopy(self)

    # Must be overloaded
    def matmul(self, A):
        raise  
    def torchdot(self, x):
        raise  
    def from_torch_dense(self, A):
        raise
    def from_scipy_sparse(self, A):
        raise
    def nnz(self):
        raise
    def transpose(self):
        raise
    def tocoo(self):
        raise

    
class SparseMatrixScipy(SparseMatrix):
    def __init__(self, A):
        assert self.is_scipy_sparse(A) or self.is_numpy_dense(A), "Invalid input - %s" % (str(type(A)))
        
        super(SparseMatrixScipy, self).__init__()
        self.shape = A.shape  # shape=(H,W)
        self._matrix = A.tocsr() if self.is_scipy_sparse(A) else A
        self.dtype = A.dtype
        self.ndim = 2
        
    def from_torch_dense(self, A):
        assert self.is_torch_dense(A)                
        return SparseMatrixScipy(A.detach().numpy())

    def from_scipy_sparse(self, A):
        assert self.is_scipy_sparse(A)        
        return SparseMatrixScipy(A)
    
    def matmul(self, A):
        assert isinstance(A, SparseMatrixScipy)
        self._matrix = scipy.sparse.csr_matrix.dot(self._matrix, A._matrix)
        self.shape = self._matrix.shape
        return self

    def dot(self, x_numpy):
        assert self.is_numpy_dense(x_numpy)
        return scipy.sparse.csr_matrix.dot(self._matrix, np.matrix(x_numpy))
        
    def torchdot(self, x_torch):
        assert self.is_torch_dense(x_torch)
        return torch.as_tensor(scipy.sparse.csr_matrix.dot(self._matrix, np.matrix(x_torch.detach().numpy())))

    def nnz(self):
        return self._matrix.nnz

    def transpose(self):
        self._matrix = self._matrix.transpose()
        self.shape = self._matrix.shape
        return self

    def tocoo(self):
        return self._matrix.tocoo()
    
    
class SparseMatrixTorch(SparseMatrix):
    def __init__(self, A):
        assert self.is_torch_sparse(A) or self.is_torch_dense(A), "Invalid input"
        
        super(SparseMatrixTorch, self).__init__()
        self.shape = A.shape
        self._matrix = A
        self.dtype = A.type
        self.ndim = 2
        
    def from_torch_dense(self, A):
        assert self.is_torch_dense(x)
        return SparseMatrixTorch(x)

    def from_scipy_sparse(self, A):
        assert self.is_scipy_sparse(A)
        return SparseMatrixTorch(scipy_coo_to_torch_sparse(A))

    def matmul(self, A):
        assert isinstance(A, SparseMatrixTorch)        
        self._matrix = torch.matmul(self._matrix, A._matrix)
        self.shape = self._matrix.shape        
        return self

    def dot(self, x):
        return self.torchdot(x)
    
    def torchdot(self, x):
        assert self.is_torch_dense(x)
        return torch.matmul(self._matrix, x)

    def nnz(self):
        return self._matrix._nnz() if self.is_torch_sparse(self._matrix) else self._matrix.size

    def transpose(self):
        self._matrix = self._matrix.t()
        self.shape = self._matrix.shape
        return self

    def tocoo(self):
        return torch_sparse_to_scipy_coo(self._matrix)

    
class SparseTiledMatrix(SparseMatrix):
    def __init__(self, tilesize=None, coo_matrix=None, blocktoeplitz=None, shape=None):
        self.dtype = None
        self.shape = None
        self.ndim = None
        self._tilesize = None        
        self._d_blockhash_to_tile = {}
        self._blocklist = []
        
        if coo_matrix is not None and tilesize is not None:
            self._from_coomatrix(coo_matrix, tilesize)
        elif blocktoeplitz is not None and shape is not None:
            self._from_blocktoeplitz(shape, blocktoeplitz)
        else:
            raise ValueError('Must provide a constructor')

    def __repr__(self):
        return str('<keynet.SparseTiledMatrix: H=%d, W=%d, tilesize=%d, tiles=%d>' % (*self.shape, self.tilesize(), len(self.tiles())))

    def tilesize(self):
        return self._tilesize

    def tiles(self):
        return list(self._d_blockhash_to_tile.values())
    
    def _block(self, B):
        return SparseMatrixScipy(B)
    
    def is_tiled_sparse(self, x):
        return isinstance(x, SparseTiledMatrix)

    def from_torch_dense(self, A):
        assert self.is_torch_dense(A)
        return SparseTiledMatrix(coo_matrix=scipy.sparse.coo_matrix(A.detach().numpy()), tilesize=self.tilesize())

    def from_scipy_sparse(self, A):
        assert self.is_scipy_sparse(A)
        return SparseTiledMatrix(coo_matrix=A.tocoo(), tilesize=self.tilesize())
    
    def _from_coomatrix(self, T, tilesize, verbose=False):
        """Given a sparse matrix T, split into non-overlapping nxn blocks or 'tiles' of size self._tilesize x self.Blocksize, and return an indexed representation for unique submatrices which provides memory efficient matrix vector multiplication when T is self-similar
        
        Representation
            B = [(i,j,k),...] for block index (i,j) with submatrix key k
            M = {k:np.array(), ...} a submatrix dictionary, such that the submatrix for block (i,j)=(B[u][0], B[u][1]) is M[B[u][2]]
        
        """
        assert self.is_scipy_sparse(T), "COO sparse matrix must be scipy.sparse.coo_matrix()"
        
        self._tilesize = tilesize
        self.dtype = T.dtype
        self.shape = (T.shape[0], T.shape[1])
        self.ndim = 2

        T = T.tocoo()        
        n = tilesize
        (H,W) = self.shape        
        ijv = [(i,j,v, (i//n,j//n)) for (i,j,v) in zip(T.row, T.col, T.data)]  # preallocate
        ijv.sort(key=lambda x: x[3])  # in-place sort for groupby, sort only indexes
        d_blockidx_to_rows_cols_vals = {k:(tuple(zip(*tuple(v)))[0:3]) for (k,v) in groupby(ijv, key=lambda x: x[3])}
        
        # Single process hashing and tiling
        d = d_blockidx_to_rows_cols_vals
        (B, M, n) = ([], {}, tilesize)              
        for (bi, i) in enumerate(range(0, T.shape[0], n)):
            for (bj, j) in enumerate(range(0, T.shape[1], n)):
                if (bi, bj) in d and len(d[bi, bj])>0:
                    (rows, cols, vals) = d[bi, bj]
                    (blockrows, blockcols) = (np.array(rows)-i, np.array(cols)-j)
                    trimshape = (min(H-i, n), min(W-j, n))
                    if trimshape[0] < n or trimshape[1] < n:
                        (blockrows, blockcols, vals) = zip(*[(ii,jj,vv) for (ii,jj,vv) in zip(blockrows, blockcols, vals) if ii < trimshape[0] and jj < trimshape[1]])
                    if len(blockrows) > 0:
                        k = hash(tuple(list(trimshape) + sorted([tuple(r) for r in np.vstack( (W*np.array(blockrows)+np.array(blockcols), np.array(vals)) ).tolist()], key=lambda x: x[0])))
                        # k = np.random.randint(100000000)  # TESTING
                        if k not in M:
                            M[k] = self._block(scipy.sparse.coo_matrix( (vals, (blockrows, blockcols)), shape=trimshape, dtype=np.float32))
                        B.append( (bi, bj, k) )
        self._blocklist = B
        self._d_blockhash_to_tile = M
        return self

    def _from_blocktoeplitz(self, shape, B):
        """A block Toeplitz matrix has blocks repeated down the main diagonal"""
        assert B.shape[0] == B.shape[1] and B.ndim == 2, "Invalid block, must be square"
        
        self._tilesize = B.shape[0]
        self.shape = shape
        self.dtype = B.dtype
        self.ndim = 2

        (H,W) = shape                    
        n = self._tilesize
        self._blocklist = [(i//n, i//n, 0) for i in range(0, min(self.shape), n)]
        self._d_blockhash_to_tile = {0: self._block(scipy.sparse.coo_matrix(B, dtype=np.float32))}
        
        if min(shape) % n != 0:
            (i,j,k) = self._blocklist[-1]
            self._blocklist[-1] = (i,j,1)
            self._d_blockhash_to_tile[1] = self._block(scipy.sparse.coo_matrix(np.eye(n)[0:H-i*n, 0:W-i*n]).astype(np.float32))
        return self    

    def dot(self, x):
        assert self.is_numpy_dense(x)
        return self.torchdot(torch.as_tensor(x)).numpy()
    
    def torchdot(self, x):
        """Input is (C*H*W+1)xN tensor, compute right matrix multiplication T*x, return (-1)xN"""
                
        n = self._tilesize
        (H,W) = self.shape

        y = torch.zeros((H, x.shape[1])).type(torch.FloatTensor)  # device?        
        for (i,j,k) in self._blocklist:
            if k is not None:
                (H_clip, W_clip) = (min(H, i*n+n), min(W, j*n+n))
                y[i*n:H_clip, :] += self._d_blockhash_to_tile[k].torchdot(x[j*n:W_clip, :])
        return y
                
    def matmul(self, other):
        """For two Tiled() object T1, T2, compute T1.dot(T2) and save in T1"""
        assert isinstance(other, SparseTiledMatrix)
        assert other._tilesize == self._tilesize, "Non-conformal tilesize"
        assert other.shape[0] == self.shape[1], "Non-conformal shape"
        
        n = self.tilesize()
        (H,W) = self.shape

        # Accumulate
        M_accum = {}
        M_hash = {}
        d_product = {}        
        for (i, jj, v) in self._blocklist:
            for (ii, j, vo) in other._blocklist:
                if jj == ii and v is not None and vo is not None:
                    if (v,vo) not in d_product:
                        d_product[(v,vo)] = self._d_blockhash_to_tile[v].clone().matmul(other._d_blockhash_to_tile[vo])   # cache
                    if (i,j) not in M_accum:
                        M_accum[(i,j)] = d_product[(v,vo)]
                        M_hash[(i,j)] = [(v,vo)]
                    else:
                        M_accum[(i,j)] += d_product[(v,vo)]
                        M_hash[(i,j)].append( (v,vo) )

        (B, M) = ([], {})
        for ((i,j), m) in M_accum.items():
            k = hash(tuple(sorted(M_hash[(i,j)], key=lambda x: x[0])))
            if k not in M:
                M[k] = m
            B.append( (i,j,k) )                        
        
        self._blocklist = B
        self._d_blockhash_to_tile = M
        self.shape = (self.shape[0], other.shape[1])
        return self

    def transpose(self):
        self._blocklist = [(j,i,k) for (i,j,k) in self._blocklist]
        self._d_blockhash_to_tile = {k:v.transpose() for (k,v) in self._d_blockhash_to_tile.items()}
        self.shape = (self.shape[1], self.shape[0])
        return self
    
    def tocoo(self):
        """Convert to Scipy COOrdinate sparse matrix, this is an expensive operation that should be used for small matrices only and for testing purposes"""
        ((H,W), n) = (self.shape, self._tilesize)
        d = {(i*n, j*n):self._d_blockhash_to_tile[k].tocoo() for (i,j,k) in self._blocklist}
        B = [ [d[(i,j)] if (i,j) in d else None for j in range(0,W,n)] for i in range(0,H,n)]            
        return scipy.sparse.bmat([ [d[(i,j)] if (i,j) in d else None for j in range(0,W,n)] for i in range(0,H,n)], format='coo')

    def nnz(self):
        return sum([m.nnz() for m in self._d_blockhash_to_tile.values()])


class SparseTiledMatrixTorch(SparseTiledMatrix):
    def __init__(self, tilesize=None, coo_matrix=None, blocktoeplitz=None, shape=None):
        super(SparseTiledMatrixTorch, self).__init__(tilesize, coo_matrix, blocktoeplitz, shape)

    def _block(self, B):
        return SparseMatrixTorch(scipy_coo_to_torch_sparse(B))

    
def sparse_identity_tiled_matrix_with_inverse(N, tilesize):
    (B, Binv) = (sparse_identity_matrix(tilesize), sparse_identity_matrix(tilesize))
    return (SparseTiledMatrix(shape=(N,N), blocktoeplitz=B.todense(), tilesize=tilesize),
            SparseTiledMatrix(shape=(N,N), blocktoeplitz=Binv.todense(), tilesize=tilesize))


def sparse_permutation_tiled_matrix_with_inverse(N, tilesize):
    (B, Binv) = sparse_permutation_matrix_with_inverse(tilesize)
    return (SparseTiledMatrix(shape=(N,N), blocktoeplitz=B.todense(), tilesize=tilesize),
            SparseTiledMatrix(shape=(N,N), blocktoeplitz=Binv.todense(), tilesize=tilesize))


def sparse_generalized_permutation_tiled_matrix_with_inverse(N, tilesize, beta):
    (B, Binv) = sparse_generalized_permutation_matrix_with_inverse(tilesize, beta)
    return (SparseTiledMatrix(shape=(N,N), blocktoeplitz=B.todense(), tilesize=tilesize),
            SparseTiledMatrix(shape=(N,N), blocktoeplitz=Binv.todense(), tilesize=tilesize))


def sparse_generalized_stochastic_tiled_matrix_with_inverse(N, tilesize, alpha, beta=0):
    (B, Binv) = sparse_generalized_stochastic_matrix_with_inverse(tilesize, alpha, beta)
    return (SparseTiledMatrix(shape=(N,N), blocktoeplitz=B.todense(), tilesize=tilesize),
            SparseTiledMatrix(shape=(N,N), blocktoeplitz=Binv.todense(), tilesize=tilesize))



