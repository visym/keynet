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
    def __init__(self, A):
        self._matrix = A
        assert len(A.shape) == 2, "SparseMatrix must be 2D"
        self.ndim = 2
        self.shape = A.shape

    def __repr__(self):
        return str('<keynet.SparseMatrix: H=%d, W=%d, backend=%s>' % (self.shape[0], self.shape[1], self.backend()))

    def __getitem__(self, k):
        if self.is_numpy_dense() or self.is_scipy_sparse() or self.is_torch():
            if not (isinstance(k, slice) or isinstance(k, tuple)):
                k = slice(k,k+1)  # force result to be 2D
            return SparseMatrix(self._matrix.__getitem__(k))  # no copy
        else:
            raise

    def backend(self):
        return str(type(self._matrix))
    
    def __add__(self, other):
        assert isinstance(other, SparseMatrix), "Invalid input"
        assert other.backend() == self.backend(), "Backend mismatch"
        assert self.shape == other.shape
        self._matrix += other._matrix
        return self
        
    def is_torch_sparse(self, x=None):
        return isinstance(x, torch.sparse.FloatTensor) if x is not None else isinstance(self._matrix, torch.sparse.FloatTensor)

    def is_torch_dense(self, x=None):
        return isinstance(x, torch.FloatTensor) if x is not None else isinstance(self._matrix, torch.FloatTensor)

    def is_scipy_sparse(self, x=None):
        return scipy.sparse.issparse(x) if x is not None else scipy.sparse.issparse(self._matrix)	

    def is_torch(self, x=None):
        return self.is_torch_sparse(x) or self.is_torch_dense(x)
    
    def is_cupy_sparse(self, x=None):
        return False

    def is_numpy_dense(self, x=None):
        return isinstance(x, np.ndarray) if x is not None else isinstance(self._matrix, np.ndarray)

    def is_tiled_sparse(self, x=None):
        return isinstance(x, SparseTiledMatrix) if x is not None else isinstance(self._matrix, SparseTiledMatrix)

    def is_keynet_sparse(self, x=None):
        return isinstance(x, SparseMatrix) if x is not None else isinstance(self._matrix, SparseMatrix)

    def _backend_like(self, x):
        if self.is_torch_dense(x):
            return 'torch'
        elif self.is_numpy_dense(x):
            return 'numpy'
        elif self.is_scipy_sparse(x):
            return 'scipy'
        elif self.is_torch_sparse(x):
            return 'torch.sparse'
        else:
            raise ValueError('Unknown backend for type "%s"' % str(type(x)))

    def _convert_matrix(self, to):
        self._matrix = self._convert_tensor(self._matrix, to)
        return self
        
    def _convert_tensor(self, x, to):
        if self.is_torch_dense(to) or to is 'torch':
            if self.is_numpy_dense(x):
                return torch.as_tensor(x)
            elif self.is_scipy_sparse(x):
                return torch.as_tensor(x.todense())
            elif self.is_torch_dense(x):
                return x
        elif self.is_numpy_dense(to) or to is 'numpy':
            if self.is_torch_dense(x):
                return x.detach().numpy()
            elif self.is_torch_sparse(x):
                return x.to_dense().detach().numpy()
            elif self.is_numpy_dense(x):
                return x
        elif self.is_scipy_sparse(to) or to is 'scipy':
            if self.is_torch_dense(x):
                return scipy.sparse.coo_matrix(x.detach().numpy(), dtype=np.float32)
            elif self.is_torch_sparse(x):
                # return torch_sparse_to_scipy_coo(x) 
                return scipy.sparse.coo_matrix(x.to_dense().detach().numpy(), dtype=np.float32)
            elif self.is_scipy_sparse(x):
                return x
        elif self.is_torch_sparse(to) or to is 'torch.sparse':
            if self.is_scipy_sparse(x):
                return scipy_coo_to_torch_sparse(x)
            elif self.is_torch_sparse(x):
                return x.to_dense()  # right multiply requires dense
        elif self.is_cupy_sparse(to) or to is 'cupy':
            if self.is_torch_dense(x):
                try_import(package='cupy', pipname='cupy, cupyx')
                return from_dlpack(self._matrix.dot(cupy.fromDlpack(to_dlpack(x_affine.t()))).toDlpack()).t()
        raise ValueError('Invalid backend "%s" for tensor format "%s"' % (self.backend(), str(type(x))))
        

    def matmul(self, x):
        if self.is_keynet_sparse(x):
            assert self.shape[1] == x.shape[0], "dimension mismatch"
            #assert self.backend() == x.backend(), "backend mismatch"
            return SparseMatrix(self.matmul(x._matrix))  # SparseMatrix() input returns SparseMatrix()
        elif self.is_tiled_sparse():
            if self.is_scipy_sparse(x):
                xh = SparseTiledMatrix(coo_matrix=x, tilesize=self._matrix.tilesize(), backend=self._matrix.backend())  # expensive
                return self._matrix.matmul(xh)
            elif self.is_torch_dense(x):
                xh = self._convert_tensor(x, to='scipy')
                xh = SparseTiledMatrix(coo_matrix=xh, tilesize=self._matrix.tilesize(), backend=self._matrix.backend())  # expensive
                return self._matrix.matmul(xh)                
            elif self.is_tiled_sparse(x):
                return self._matrix.matmul(x)
            else:
                raise ValueError('Invalid input - must be SparseTiledMatrix()')
        elif self.is_torch_dense():
            xh = self._convert_tensor(x, to='torch')
            return torch.matmul(xh)
        elif self.is_numpy_dense():
            xh = self._convert_tensor(x, to='numpy')
            return np.dot(self._matrix, xh)
        elif self.is_scipy_sparse():
            xh = self._convert_tensor(x, to='scipy')
            return self._matrix.dot(xh)  # FIXME
        else:
            print(str(type(x)))
            raise ValueError('Invalid backend "%s" and matrix format "%s"' % (self.backend(), str(type(self._matrix))))                        
    
    def dot(self, x):
        if self.is_torch_sparse():
            xh = self._convert_tensor(x, to='torch')  
            y = torch.sparse.mm(self._matrix, xh) 
        elif self.is_scipy_sparse():
            xh = self._convert_tensor(x, to='numpy') 
            y = self._matrix.dot(xh)
        elif self.is_torch_dense():
            xh = self._convert_tensor(x, to='torch')              
            y = torch.matmul(self._matrix, xh)
        elif self.is_cupy_sparse():
            xh = self._convert_tensor(x, to='cupy')              
            y = from_dlpack(self._matrix.dot(xh))
        elif self.is_tiled_sparse():
            xh = self._convert_tensor(x, to='numpy')
            y = self._matrix.dot(xh)
        else:
            raise ValueError('Invalid backend "%s"' % self.backend())
        return self._convert_tensor(y, to=x)  # output same type as input
    
    def transpose(self):
        if self.is_torch() or self.is_cupy_sparse():
            self._matrix = self._matrix.t()
            self.shape = self._matrix.shape
        elif self.is_numpy_dense() or self.is_tiled_sparse() or self.is_scipy_sparse():
            self._matrix = self._matrix.transpose()
            self.shape = self._matrix.shape
        else:
            raise ValueError('Invalid matrix for transpose')            
        return self

    def nnz(self):
        if self.is_scipy_sparse():
            return self._matrix.nnz
        elif self.is_tiled_sparse():
            return self._matrix.nnz()
        elif self.is_torch() or self.is_numpy_dense():
            return self._matrix.size
        elif self.is_torch_sparse():
            return self._matrix._nnz()
        else:
            raise ValueError('Invalid matrix for nnz')            
        
    def numpy(self):
        if self.is_torch():
            return np.array(self._matrix.detach().numpy())
        elif self.is_scipy_sparse():
            return np.array(self._matrix.todense())
        elif self.is_numpy_dense():
            return self._matrix
        else:
            raise ValueError('invalid matrix for numpy conversion')

    def torch(self):
        if self.is_numpy_dense():
            return torch.as_tensor(self._matrix).type(torch.FloatTensor)
        elif self.is_torch():
            return self._matrix
        else:
            raise ValueError('invalid matrix for torch conversion')            

    def clone(self):
        return copy.deepcopy(self)
        
class SparseTiledMatrix(object):
    def __init__(self, tilesize=None, coo_matrix=None, blocktoeplitz=None, shape=None, backend='scipy'):
        self._backend = backend        
        if coo_matrix is not None:
            self.from_coomatrix(coo_matrix, tilesize)
        elif blocktoeplitz is not None:
            self.from_blocktoeplitz(shape, blocktoeplitz)
        elif shape is not None and tilesize is not None:
            self.shape = shape
            self._tilesize = tilesize
        else:
            raise ValueError('Must provide a constructor')

    def backend(self, backend=None):
        self._backend = backend
        
        
    def from_coomatrix(self, T, tilesize, verbose=False):
        """Given a sparse matrix T, split into non-overlapping nxn blocks or 'tiles' of size self._tilesize x self.Blocksize, and return an indexed representation for unique submatrices which provides memory efficient matrix vector multiplication when T is self-similar
        
        Representation
            B = [(i,j,k),...] for block index (i,j) with submatrix key k
            M = {k:np.array(), ...} a submatrix dictionary, such that the submatrix for block (i,j)=(B[u][0], B[u][1]) is M[B[u][2]]
        
        """
        self._tilesize = tilesize
        self.dtype = T.dtype
        self.shape = (T.shape[0], T.shape[1])
        self.ndim = 2
        n = tilesize
        (H,W) = self.shape
        
        T = T.tocoo()
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
                        if k not in M:
                            M[k] = SparseMatrix(scipy.sparse.coo_matrix( (vals, (blockrows, blockcols)), shape=trimshape, dtype=np.float32))._convert_matrix(self._backend) 
                        B.append( (bi, bj, k) )
        self._B = B
        self._M = M
        return self

    def from_blocktoeplitz(self, shape, B):
        """A block Toeplitz matrix has blocks repeated down the main diagonal"""
        assert B.shape[0] == B.shape[1] and B.ndim == 2, "Invalid block, must be square"
        self._tilesize = B.shape[0]
        self.shape = shape
        n = self._tilesize
        self._B = [(i//n, i//n, 0) for i in range(0, min(self.shape), n)]
        self._M = {0: SparseMatrix(scipy.sparse.coo_matrix(B, dtype=np.float32))._convert_matrix(self._backend)}

        if min(self.shape) % n != 0:
            (H,W) = shape            
            (i,j,k) = self._B[-1]
            self._B[-1] = (i,j,1)
            S = scipy.sparse.coo_matrix(np.eye(n)[0:H-i*n, 0:W-i*n]).astype(np.float32)
            self._M[1] = SparseMatrix(S)._convert_matrix(self._backend)          
        self.dtype = B.dtype
        self.ndim = 2
        return self    
                            
    def __repr__(self):
        return str('<keynet.SparseTiledMatrix: H=%d, W=%d, tilesize=%d, tiles=%d, backend=%s>' % (*self.shape, self.tilesize(), len(self.tiles()), self._backend))
                   
    def tilesize(self):
        return self._tilesize

    def tiles(self):
        return list(self._M.values())
    
    def dot(self, x):
        """Input is (C*H*W+1)xN tensor, compute right matrix multiplication T*x, return (-1)xN"""
        if isinstance(x, SparseTiledMatrix):
            return self.matmul(x)

        n = self._tilesize
        (H,W) = self.shape

        y = torch.zeros((H, x.shape[1])).type(torch.FloatTensor)  # device?
        
        for (i,j,k) in self._B:
            if k is not None:
                (H_clip, W_clip) = (min(H, i*n+n), min(W, j*n+n))
                y[i*n:H_clip, :] += self._M[k].dot(x[j*n:W_clip, :])
        return y
                
    def transpose(self):
        self._B = [(j,i,k) for (i,j,k) in self._B]
        self._M = {k:v.transpose() for (k,v) in self._M.items()}
        self.shape = (self.shape[1], self.shape[0])
        return self

    def matmul(self, other):
        """For two Tiled() object T1, T2, compute T1.dot(T2) and save in T1"""
        assert isinstance(other, SparseTiledMatrix)
        assert other._tilesize == self._tilesize
        assert other.shape[0] == self.shape[1], "Non-conformal shape"
        
        n = self.tilesize()
        (H,W) = self.shape

        # Accumulate
        M_accum = {}
        M_hash = {}
        d_product = {}        
        for (i, jj, v) in self._B:
            for (ii, j, vo) in other._B:
                if jj == ii and v is not None and vo is not None:
                    if (v,vo) not in d_product:
                        d_product[(v,vo)] = self._M[v].matmul(other._M[vo])   # cache
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
        
        self._B = B
        self._M = M
        self.shape = (self.shape[0], other.shape[1])
        return self
    
    def tocoo(self):
        """Convert to COOrdinate sparse matrix, this is an expensive operation that should be used for small matrices only and for testing purposes"""
        ((H,W), n) = (self.shape, self._tilesize)
        d = {(i*n, j*n):scipy.sparse.coo_matrix(self._M[k].numpy()) for (i,j,k) in self._B}
        B = [ [d[(i,j)] if (i,j) in d else None for j in range(0,W,n)] for i in range(0,H,n)]            
        return scipy.sparse.bmat([ [d[(i,j)] if (i,j) in d else None for j in range(0,W,n)] for i in range(0,H,n)], format='coo')

    def clone(self):
        return copy.deepcopy(self)

    def nnz(self):
        return sum([scipy.sparse.coo_matrix(m.numpy()).nnz for m in self._M.values()])

    
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



