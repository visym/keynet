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
from itertools import groupby

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
    return scipy.sparse.diags(np.maximum(eps, np.array(sigma*np.random.randn(n)+mu).astype(np.float32)))


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

    return(A,Ainv)


def sparse_generalized_permutation_matrix_with_inverse(n): 
    (A,Ainv) = sparse_permutation_matrix_with_inverse(n)
    (D,Dinv) = sparse_gaussian_random_diagonal_matrix_with_inverse(n)
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


class SparseTiledMatrix(object):
    def __init__(self, tilesize=None, coo_matrix=None, blocktoeplitz=None, shape=None):
        if coo_matrix is not None:
            self.from_coomatrix(coo_matrix, tilesize)
        elif blocktoeplitz is not None:
            self.from_blocktoeplitz(shape, blocktoeplitz)
        else:
            raise ValueError('Must provide a constructor')

    def from_coomatrix(self, T, tilesize):
        """Given a sparse matrix T, split into non-overlapping nxn blocks or 'tiles' of size self._tilesize x self.Blocksize, and return an indexed representation for unique submatrices which provides memory efficient matrix vector multiplication when T is self-similar
        
        Representation
            B = [(i,j,k),...] for block index (i,j) with submatrix key k
            M = {k:np.array(), ...} a submatrix dictionary, such that the submatrix for block (i,j)=(B[u][0], B[u][1]) is M[B[u][2]]
        
        """
        T = T.tocoo()
        (B, M, n) = ([], {}, tilesize)
        ijv = [(i,j,v) for (i,j,v) in zip(T.row, T.col, T.data)]  # preallocate
        ijv.sort(key=lambda x: (x[0]//n, x[1]//n))  # in-place sort for groupby, sort only indexes
        d_blockidx_to_entries = {k:sorted(v, key=lambda y: (y[0], y[1])) for (k,v) in groupby(ijv, key=lambda x: (x[0]//n, x[1]//n))}   # sorted for hash
        for i in range(0, T.shape[0], n):
            for j in range(0, T.shape[1], n):
                if (i//n,j//n) in d_blockidx_to_entries:
                    (rows, cols, vals) = zip(*[(ii-i,jj-j,v) for (ii,jj,v) in d_blockidx_to_entries[(i//n,j//n)]])
                    k = hash((vals, (rows, cols)))
                    if k not in M:
                        m = scipy.sparse.coo_matrix( (vals, (rows,cols)), shape=(n, n)).todense().astype(np.float32)  # submatrix                                            
                        M[k] = torch.as_tensor(m)
                else:
                    k = None
                B.append( (i//n, j//n, k) )
        self._B = list(set(B))
        self._M = M
        self._tilesize = tilesize
        self.dtype = T.dtype
        self.shape = (T.shape[0], T.shape[1])
        self.ndim = 2
        return self

    def from_blocktoeplitz(self, shape, B):
        """A block Toeplitz matrix has blocks repeated down the main diagonal"""
        assert B.shape[0] == B.shape[1] and B.ndim == 2, "Invalid block, must be square"
        self._tilesize = B.shape[0]
        self.shape = shape
        n = self._tilesize
        self._B = [(i//n, i//n, 0) for i in range(0, min(self.shape), n)]
        self._M = {0: torch.as_tensor(B)}
        if min(self.shape) % n != 0:
            (H,W) = shape            
            (i,j,k) = self._B[-1]
            self._B[-1] = (i,j,1)
            #Z = np.zeros_like(B)
            Z = np.identity(B.shape[0], dtype=B.dtype)  # boundary must be invertible
            Z[H-i*n:, W-j*n:] = 0
            self._M[1] = torch.as_tensor(Z)
        self.dtype = B.dtype
        self.ndim = 2
        return self    
                            
    def __repr__(self):
        return str('<keynet.SparseTiledMatrix: H=%d, W=%d, tilesize=%d, tiles=%d>' % (*self.shape, self.tilesize(), len(self.tiles())))

    def tilesize(self):
        return self._tilesize

    def tiles(self):
        return list(self._M.values())
    
    def leftdot(self, x):
        """Input is NxCxHxW tensor viewed as Nx(C*H*W) tensor, compute left matrix multiplication (x * T) return (NxC*H*W)"""
        n = self._tilesize
        (H,W) = self.shape        
        y = torch.zeros((x.shape[0], W), dtype=x.dtype, device=x.device)
        for (i,j,k) in self._B:
            if k is not None:
                (H_clip, W_clip) = (min(H, i*n+n), min(W, j*n+n))
                y[:, j*n:W_clip] += torch.matmul(x[:, i*n:H_clip], self._M[k][0:(H_clip-i*n), 0:(W_clip-j*n)])
        return y

    def dot(self, x):
        """Input is (C*H*W+1)xN tensor, compute right matrix multiplication T*x, return (-1)xN"""
        n = self._tilesize
        (H,W) = self.shape        
        y = torch.zeros((H, x.shape[1]), dtype=x.dtype, device=x.device)
        for (i,j,k) in self._B:
            if k is not None:
                (H_clip, W_clip) = (min(H, i*n+n), min(W, j*n+n))
                y[i*n:H_clip, :] += torch.matmul(self._M[k][0:(H_clip-i*n), 0:(W_clip-j*n)], x[j*n:W_clip, :])
        return y
                
    def transpose(self):
        self._B = [(j,i,k) for (i,j,k) in self._B]
        self._M = {k:v.t() for (k,v) in self._M.items()}
        self.shape = (self.shape[1], self.shape[0])
        return self

    def prod(self, other):
        """For two Tiled() object T1, T2, compute T1.dot(T2) and save in T1"""
        assert isinstance(other, SparseTiledMatrix)
        assert other._tilesize == self._tilesize

        n = self.tilesize()
        (H,W) = self.shape

        # Pre-multiply
        d_product = {}
        for (v, m) in self._M.items():
            for (vo, mo) in other._M.items():
                d_product[(v,vo)] = torch.matmul(m, mo)

        # Accumulate
        M_accum = {}
        M_hash = {}
        for (i, jj, v) in self._B:
            for (ii, j, vo) in other._B:
                if jj == ii and v is not None and vo is not None:
                    if (i,j) not in M_accum:
                        M_accum[(i,j)] = d_product[(v,vo)]
                        M_hash[(i,j)] = v+vo
                    else:
                        M_accum[(i,j)] += d_product[(v,vo)]
                        M_hash[(i,j)] += v+vo                        

        (B, M) = ([], {})
        for ((i,j), m) in M_accum.items():
            k = M_hash[(i,j)]
            if k not in M:
                M[k] = m
            B.append( (i,j,k) )                        
        
        self._B = B
        self._M = M
        self.shape = (self.shape[0], other.shape[1])
        return self
    
    def tocoo(self):
        """Convert to COOrdinate sparse matrix, this is an expensive operation that should be used for small matrices only and for testing purposes"""
        ((H,W), n) = (self.shape, self._tilesize)
        d = {(i*n, j*n):k for (i,j,k) in self._B}
        return scipy.sparse.bmat([ [scipy.sparse.coo_matrix(self._M[d[(i,j)]][0:min(n, H-i), 0: min(n, W-j)]) if ((i,j) in d and d[(i,j)] is not None) else scipy.sparse.coo_matrix( (min(n, H-i), min(n, W-j)) ) for j in range(0,W,n)]
                                for i in range(0,H,n)])

    def clone(self):
        return copy.deepcopy(self)

    
def sparse_identity_tiled_matrix_with_inverse(N, tilesize):
    (B, Binv) = (sparse_identity_matrix(tilesize), sparse_identity_matrix(tilesize))
    return (SparseTiledMatrix(shape=(N,N), blocktoeplitz=B.todense(), tilesize=tilesize),
            SparseTiledMatrix(shape=(N,N), blocktoeplitz=Binv.todense(), tilesize=tilesize))


def sparse_permutation_tiled_matrix_with_inverse(N, tilesize):
    (B, Binv) = sparse_permutation_matrix_with_inverse(tilesize)
    return (SparseTiledMatrix(shape=(N,N), blocktoeplitz=B.todense(), tilesize=tilesize),
            SparseTiledMatrix(shape=(N,N), blocktoeplitz=Binv.todense(), tilesize=tilesize))

