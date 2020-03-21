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
from vipy.util import groupbyasdict, flatlist
import copy
from itertools import groupby, product
import tempfile
from vipy.util import Stopwatch
import torch.sparse
import scipy.sparse
from numpy.linalg import multi_dot 
from collections import defaultdict
from keynet.dense import random_positive_definite_matrix, random_doubly_stochastic_matrix
from keynet.util import blockview
from joblib import Parallel, delayed
import vipy


def _parallel_sparse_toeplitz_conv2d(inshape, f, bias=None, as_correlation=True, stride=1, n_processes=1):
    T = Parallel(n_jobs=n_processes)(delayed(sparse_toeplitz_conv2d)(inshape, f, bias, as_correlation, stride, n_processes=1, rowskip=i) for i in range(inshape[1]))
    R = np.sum(T).tocsr().transpose()
    R[-1] = T[0].tocsr().transpose()[-1]  # replace bias column
    return R.transpose().tocoo()
    
                                
def sparse_toeplitz_conv2d(inshape, f, bias=None, as_correlation=True, stride=1, rowskip=None, n_processes=1):
    """ Returns sparse toeplitz matrix (W) in coo format that is equivalent to per-channel pytorch conv2d (spatial correlation) of filter f with a given image with shape=inshape vectorized
        conv2d(img, f) == np.dot(W, img.flatten()), right multiplied
        Example usage: test_keynet.test_sparse_toeplitz_conv2d()
    """

    if n_processes > 1:
        return _parallel_sparse_toeplitz_conv2d(inshape, f, bias, as_correlation, stride, n_processes=n_processes)

    # Valid shapes
    assert(len(inshape) == 3 and len(f.shape) == 4)  # 3D tensor inshape=(inchannels, height, width), f.shape=(outchannels, kernelheight, kernelwidth, inchannels)
    assert(f.shape[1] == inshape[0])  # equal inchannels
    assert(f.shape[2]==f.shape[3] and f.shape[2]%2 == 1)  # filter is square, odd (FIXME)
    if bias is not None:
        assert(len(bias.shape) == 1 and bias.shape[0] == f.shape[0])  # filter and bias have composable shapes

    # Correlation vs. convolution?
    (C,U,V) = inshape
    (M,C,P,Q) = f.shape
    C_range = range(0,C)
    M_range = range(0,M)
    P_range = range(-((P-1)//2), ((P-1)//2) + 1) if P%2==1 else range(-((P-1)//2), ((P-1)//2) + 2)
    Q_range = range(-((Q-1)//2), ((Q-1)//2) + 1) if P%2==1 else range(-((Q-1)//2), ((Q-1)//2) + 2)
    (data, row_ind, col_ind) = ([],[],[])
    (U_div_stride, V_div_stride) = (U//stride, V//stride)

    # For every image_row
    for (ku,u) in enumerate(np.arange(0,U,stride)):
        if rowskip is not None and u != rowskip:
            continue
        # For every image_column
        for (kv,v) in enumerate(np.arange(0,V,stride)):
            # For every inchannel (transposed)
            for (k_inchannel, c_inchannel) in enumerate(C_range if as_correlation else reversed(C_range)):
                # For every kernel_row (transposed)
                for (i,p) in enumerate(P_range if as_correlation else reversed(P_range)):
                    if not ((u+p)>=0 and (u+p)<U):
                        continue  
                    # For every kernel_col (transposed)
                    for (j,q) in enumerate(Q_range if as_correlation else reversed(Q_range)):
                        # For every outchannel
                        if ((v+q)>=0 and (v+q)<V):
                            #c = np.ravel_multi_index( (c_inchannel, u+p, v+q), (C,U,V) )
                            c = c_inchannel*U*V + (u+p)*V + (v+q)
                            for (k_outchannel, c_outchannel) in enumerate(M_range if as_correlation else reversed(M_range)):
                                data.append(f[k_outchannel,k_inchannel,i,j])
                                #row_ind.append( np.ravel_multi_index( (c_outchannel,ku,kv), (M,U//stride,V//stride) ) )
                                row_ind.append( c_outchannel*(U_div_stride)*(V_div_stride) + ku*(V_div_stride) + kv )
                                col_ind.append( c )

    # Sparse matrix with optional bias using affine augmentation 
    T = scipy.sparse.coo_matrix((data, (row_ind, col_ind)), shape=(M*(U//stride)*(V//stride), C*U*V))
    if bias is not None:
        lastcol = scipy.sparse.coo_matrix(np.array([x*np.ones( (U//stride*V//stride), dtype=np.float32) for x in bias]).reshape( (M*(U//stride)*(V//stride),1) ))
    else:
        lastcol = scipy.sparse.coo_matrix(np.zeros( (T.shape[0],1), dtype=np.float32 ))
    lastrow = np.zeros(T.shape[1]+1, dtype=np.float32);  lastrow[-1]=np.float32(1.0);  
    return scipy.sparse.coo_matrix(scipy.sparse.vstack( (scipy.sparse.hstack( (T,lastcol)), scipy.sparse.coo_matrix(lastrow)) ))


def sparse_toeplitz_avgpool2d(inshape, filtershape, stride):
    (outchannel, inchannel, filtersize, filtersize) = filtershape
    (M,U,V) = (inshape)
    F = np.zeros(filtershape, dtype=np.float32)
    for k in range(0,outchannel):
        F[k,k,:,:] = 1.0 / (filtersize*filtersize)
    return sparse_toeplitz_conv2d(inshape, F, bias=None, stride=stride)


def sparse_channelorder_to_blockorder(shape, blockshape, homogenize=False):
    assert blockshape <= np.prod(shape), "Invalid blockshape"
    assert np.prod(shape) % blockshape == 0, "Invalid blockshape"
    (C,H,W) = shape if shape[1] != 1 and shape[2] != 1 else (1,shape[0],1)  # flatten for fc
    img_channelorder = np.array(range(0, H*W)).reshape(H,W)  # HxW, img[0:H] == first row 
    img_blockorder = blockview(img_channelorder, blockshape).flatten()  # (H//B)x(W//B)xBxB, img[0:B*B] == first block
    (rows, cols, vals) = ([], [], [])
    for c in range(0,C):
        rows.extend(np.array(range(0, H*W)) + c*H*W)
        cols.extend(img_blockorder + c*H*W)
        vals.extend(np.ones_like(img_blockorder))
    if homogenize:
        rows.append(C*H*W)
        cols.append(C*H*W)
        vals.append(1)
    return scipy.sparse.coo_matrix( (vals, (rows, cols)), dtype=np.float32).tocsr()
                                            
    
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
    P = sparse_permutation_matrix(n, dtype)
    return (P, P.transpose())


def sparse_block_permutation_matrix(squareshape, blocksize, dtype=np.float32):
    (H,W) = (squareshape, squareshape)
    n_blocks = squareshape // blocksize
    P = sparse_permutation_matrix(n_blocks).tocoo()
    P_ij = set([(i,j) for (i,j) in zip(P.row, P.col)])
    blockrows = []
    for i in range(0, n_blocks):
        blockcols = []
        for j in range(0, n_blocks):
            if (i,j) in P_ij:
                blockcols.append(scipy.sparse.eye(blocksize))
            else:
                blockcols.append(None)
        blockrows.append(blockcols)
        
    P = scipy.sparse.bmat(blockrows, dtype=dtype)
    if P.shape != (squareshape, squareshape):
        raggedshape = (squareshape - P.shape[0], squareshape - P.shape[1])
        P = scipy.sparse.bmat( [[P, None], [None, scipy.sparse.coo_matrix(np.eye(max(raggedshape))[0:raggedshape[0], 0:raggedshape[1]])]])
    return P


def sparse_block_permutation_matrix_with_inverse(squareshape, blocksize, dtype=np.float32):
    P = sparse_block_permutation_matrix(squareshape, blocksize, dtype)
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
    """Return nxn matrix with mxm blocks on main diagonal, each block is a random posiitve definite matrix"""
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


def is_scipy_sparse(A):
    return scipy.sparse.issparse(A)


class SparseMatrix(object):
    def __init__(self, A=None, n_processes=1):
        self._n_processes = n_processes        
        assert self.is_scipy_sparse(A) or self.is_numpy_dense(A), "Invalid input - %s" % (str(type(A)))
        self.shape = A.shape  # shape=(H,W)
        self._matrix = A.tocsr() if self.is_scipy_sparse(A) else A
        self.dtype = A.dtype
        self.ndim = 2

    def parallel(self, n_processes):
        self._n_processes = n_processes
        return self
    
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
    def from_torch_dense(self, A):
        assert self.is_torch_dense(A)                
        return SparseMatrix(A.detach().numpy())

    def from_scipy_sparse(self, A):
        assert self.is_scipy_sparse(A)        
        return SparseMatrix(A)
    
    def matmul(self, A):
        assert isinstance(A, SparseMatrix)
        if self.is_scipy_sparse(self._matrix):
            self._matrix = self._matrix.tocsr()
        if self.is_scipy_sparse(A._matrix):
            A._matrix = A._matrix.tocsc()
        self._matrix = scipy.sparse.csr_matrix.dot(self._matrix, A._matrix)
        self.shape = self._matrix.shape
        return self

    def dot(self, x_numpy):
        assert self.is_numpy_dense(x_numpy)
        if self.is_scipy_sparse(self._matrix):
            self._matrix = self._matrix.tocsr()
        return scipy.sparse.csr_matrix.dot(self._matrix, np.matrix(x_numpy))
        
    def torchdot(self, x_torch):
        assert self.is_torch_dense(x_torch)
        if self.is_scipy_sparse(self._matrix):
            self._matrix = self._matrix.tocsr()
        return torch.as_tensor(scipy.sparse.csr_matrix.dot(self._matrix, np.matrix(x_torch.detach().numpy())))

    def nnz(self):
        return self._matrix.nnz if self.is_scipy_sparse(self._matrix) else self._matrix.size

    def transpose(self):
        self._matrix = self._matrix.transpose()
        self.shape = self._matrix.shape
        return self

    def tocoo(self):
        return self._matrix.tocoo() if is_scipy_sparse(self._matrix) else scipy.sparse.coo_matrix(self._matrix)

    def ascsr(self):
        self._matrix = self._matrix.tocsr()
        return self

    def ascsc(self):
        self._matrix = self._matrix.tocsc()
        return self

    def from_torch_conv2d(self, inshape, w, b, stride):
        return SparseMatrix(sparse_toeplitz_conv2d(inshape, w.detach().numpy(), bias=b.detach().numpy(), stride=stride, n_processes=self._n_processes))

    
class SparseTiledMatrix(SparseMatrix):
    def __init__(self, tilesize=None, coo_matrix=None, tile_to_blkdiag=None, shape=None, n_processes=1):
        self._n_processes = n_processes
        self.dtype = None
        self.shape = None
        self.ndim = None
        self._tilesize = None        
        self._d_blockhash_to_tile = {}
        self._blocklist = []
        
        if coo_matrix is not None and tilesize is not None:
            self._from_coomatrix(coo_matrix, tilesize)
        elif tile_to_blkdiag is not None and shape is not None:
            self._from_tile_to_blkdiag(shape, tile_to_blkdiag)
        else:
            raise ValueError('Must provide a valid constructor')

    def parallel(self, n_processes):
        self._n_processes = n_processes
        return self
    
    def __repr__(self):
        return str('<keynet.SparseTiledMatrix: H=%d, W=%d, tilesize=%d, tiles=%d>' % (*self.shape, self.tilesize(), len(self.tiles())))

    def tilesize(self):
        return self._tilesize

    def tiles(self):
        return list(self._d_blockhash_to_tile.values())
    
    def _block(self, B):
        return SparseMatrix(B)
    
    def is_tiled_sparse(self, x):
        return isinstance(x, SparseTiledMatrix)

    def from_torch_conv2d(self, inshape, w, b, stride):
        return SparseTiledMatrix(coo_matrix=sparse_toeplitz_conv2d(inshape, w.detach().numpy(), bias=b.detach().numpy(), stride=stride, n_processes=self._n_processes), tilesize=self._tilesize, n_processes=self._n_processes)
        
    def from_torch_dense(self, A):
        assert self.is_torch_dense(A)
        return SparseTiledMatrix(coo_matrix=scipy.sparse.coo_matrix(A.detach().numpy()), tilesize=self.tilesize(), n_processes=self._n_processes)

    def from_scipy_sparse(self, A):
        assert self.is_scipy_sparse(A)
        return SparseTiledMatrix(coo_matrix=A.tocoo(), tilesize=self.tilesize(), n_processes=self._n_processes)

    def _from_coomatrix(self, T, tilesize, verbose=False):
        """Given a sparse matrix T, split into non-overlapping nxn blocks or 'tiles' of size self._tilesize x self.Blocksize, and 
           return an indexed representation for unique submatrices which provides memory efficient matrix vector multiplication when T is self-similar.
        
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
        
        if T.shape[0] > tilesize:
            csr_matrix = T.tocsr()
            T_rows = Parallel(n_jobs=self._n_processes)(delayed(SparseTiledMatrix)(tilesize, csr_matrix[i:i+tilesize]) for i in np.arange(0, T.shape[0], tilesize))
            self._d_blockhash_to_tile = {k:b for t in T_rows for (k,b) in t._d_blockhash_to_tile.items()}  # unique
            self._blocklist = []
            for (i,t) in enumerate(T_rows):
                self._blocklist.extend([(i,j,k) for (ii,j,k) in t._blocklist])
            self.shape = (T.shape[0], T.shape[1])
            return self

        n = tilesize
        (H,W) = self.shape        

        # Faster than groupby
        d_blockidx_to_rows_cols_vals = defaultdict(list)
        for (i,j,v) in zip(T.row, T.col, T.data):
            d_blockidx_to_rows_cols_vals[ (i//n, j//n) ].append( (i,j,v) )

        # Slow for large matrices
        #ijv = [(i,j,v, (i//n,j//n)) for (i,j,v) in zip(T.row, T.col, T.data)]  # preallocate
        #ijv.sort(key=lambda x: x[3])  # in-place sort for groupby, sort only indexes
        #d_blockidx_to_rows_cols_vals = {k:(tuple(zip(*tuple(v)))[0:3]) for (k,v) in groupby(ijv, key=lambda x: x[3])}
        
        # Single process hashing and tiling
        d = d_blockidx_to_rows_cols_vals
        (B, M, n) = ([], {}, tilesize)              
        for (bi, i) in enumerate(range(0, T.shape[0], n)):
            for (bj, j) in enumerate(range(0, T.shape[1], n)):
                if (bi, bj) in d and len(d[bi, bj])>0:
                    (rows, cols, vals) = zip(*d[bi, bj])
                    (blockrows, blockcols) = (np.array(rows)-i, np.array(cols)-j)
                    trimshape = (min(H-i, n), min(W-j, n))
                    if trimshape[0] < n or trimshape[1] < n:
                        (blockrows, blockcols, vals) = zip(*[(ii,jj,vv) for (ii,jj,vv) in zip(blockrows, blockcols, vals) if ii < trimshape[0] and jj < trimshape[1]])
                    if len(blockrows) > 0:
                        k = hash(tuple(list(trimshape) + sorted([tuple(r) for r in np.vstack( (W*np.array(blockrows)+np.array(blockcols), np.array(vals)) ).tolist()], key=lambda x: x[0])))
                        if k not in M:
                            M[k] = self._block(scipy.sparse.coo_matrix( (vals, (blockrows, blockcols)), shape=trimshape, dtype=np.float32))
                        B.append( (bi, bj, k) )
        self._blocklist = B
        self._d_blockhash_to_tile = M
        return self

    def _from_tile_to_blkdiag(self, shape, T):
        """Construct block diagonal matrix from tile T repeated down the main diagonal"""
        assert T.shape[0] == T.shape[1] and T.ndim == 2, "Invalid block, must be square"
        
        self._tilesize = T.shape[0]
        self.shape = shape
        self.dtype = T.dtype
        self.ndim = 2

        (H,W) = shape                    
        n = self._tilesize
        self._blocklist = [(i//n, i//n, 0) for i in range(0, min(self.shape), n)]
        self._d_blockhash_to_tile = {0: self._block(scipy.sparse.coo_matrix(T, dtype=np.float32))}
        
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
        assert W == x.shape[0], "Non-conformal shape for W=%s, x=%s" % (str(self.shape), str(x.shape))

        y = torch.zeros((H, x.shape[1])).type(torch.FloatTensor)  # device?        
        for (i,j,k) in self._blocklist:
            if k is not None:
                (H_clip, W_clip) = (min(H, i*n+n), min(W, j*n+n))
                y[i*n:H_clip, :] += self._d_blockhash_to_tile[k].ascsr().torchdot(x[j*n:W_clip, :])
        return y
                
    def matmul(self, other, verbose=False):
        """For two Tiled() object T1, T2, compute T1.dot(T2) and save in T1"""
        assert isinstance(other, SparseTiledMatrix) or isinstance(other, SparseMatrix), "Invalid input - Must be SparseMatrix()"
        assert other.shape[0] == self.shape[1], "Non-conformal shape"        
        if isinstance(other, SparseMatrix) and not isinstance(other, SparseTiledMatrix):
            # Downgrade to sparse matrix, multiply then upgrade to SparseTiledMatrix (expensive)
            return self._from_coomatrix(SparseMatrix(self.tocoo()).matmul(other).tocoo(), self.tilesize())

        assert other._tilesize == self._tilesize, "Non-conformal tilesize"    
        n = self.tilesize()
        (H,W) = self.shape

        # Accumulate
        M_accum = {}
        M_hash = {}
        d_product = {}        
        for (i, jj, v) in self._blocklist:
            if verbose:
                print('[keynet.SparseTiledMatrix][%d/%d]: Product...' % (i,jj))
            for (ii, j, vo) in other._blocklist:
                if jj == ii and v is not None and vo is not None:
                    if (v,vo) not in d_product:
                        d_product[(v,vo)] = self._d_blockhash_to_tile[v].ascsr().clone().matmul(other._d_blockhash_to_tile[vo].ascsc())   # cache
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



    
def sparse_diagonal_tiled_identity_matrix_with_inverse(N, tilesize, tiler=SparseTiledMatrix):
    """(NxN) identity matrix, with one (tilesize, tilesize) identity matrix repeated along main diagonal"""
    (B, Binv) = (sparse_identity_matrix(tilesize), sparse_identity_matrix(tilesize))
    return (tiler(shape=(N,N), tile_to_blkdiag=B.todense(), tilesize=tilesize),
            tiler(shape=(N,N), tile_to_blkdiag=Binv.todense(), tilesize=tilesize))


def sparse_block_diagonal_tiled_permutation_matrix_with_inverse(N, tilesize, tiler=SparseTiledMatrix):
    """(NxN) block diagonal matrix with one (tilesize x tilesize) permutation matrix repeated along main diagonal"""
    (B, Binv) = sparse_permutation_matrix_with_inverse(tilesize)
    return (tiler(shape=(N,N), tile_to_blkdiag=B.todense(), tilesize=tilesize),
            tiler(shape=(N,N), tile_to_blkdiag=Binv.todense(), tilesize=tilesize))


def sparse_block_diagonal_tiled_generalized_permutation_matrix_with_inverse(N, tilesize, beta, tiler=SparseTiledMatrix):
    """(NxN) block diagonal matrix with one (tilesize, tilesize) generalized permutation matrix repeated along main diagonal""" 
    (B, Binv) = sparse_generalized_permutation_matrix_with_inverse(tilesize, beta)
    return (tiler(shape=(N,N), tile_to_blkdiag=B.todense(), tilesize=tilesize),
            tiler(shape=(N,N), tile_to_blkdiag=Binv.todense(), tilesize=tilesize))


def sparse_block_diagonal_tiled_generalized_stochastic_matrix_with_inverse(N, tilesize, alpha, beta=0, tiler=SparseTiledMatrix):
    """(NxN) block diagonal matrix with one (tilesize, tilesize) generalized stochastic matrix repeated along main diagonal""" 
    (B, Binv) = sparse_generalized_stochastic_matrix_with_inverse(tilesize, alpha, beta)
    return (tiler(shape=(N,N), tile_to_blkdiag=B.todense(), tilesize=tilesize),
            tiler(shape=(N,N), tile_to_blkdiag=Binv.todense(), tilesize=tilesize))


def sparse_block_permutation_tiled_identity_matrix_with_inverse(squareshape, tilesize, tiler=SparseTiledMatrix):
    """(NxN) permutation matrix with every non-overlapping (tilesize, tilesize) submatrix as either identity or zero"""
    (P, Pinv) = sparse_block_permutation_matrix_with_inverse(squareshape, tilesize)
    return (tiler(shape=(squareshape, squareshape), coo_matrix=P.tocoo(), tilesize=tilesize),
            tiler(shape=(squareshape, squareshape), coo_matrix=P.tocoo(), tilesize=tilesize).transpose())


def spy(A, mindim=256, showdim=1024, range=None):
    """Visualize sparse matrix A"""

    if range is not None:
        B = A.tocoo().tocsr()[range[0]:range[1]].transpose()[range[0]:range[1]].transpose().tocoo()
        return spy(B, mindim, showdim, range=None)
    
    scale = float(mindim) / min(A.shape)
    (H, W) = np.ceil(np.array(A.shape)*scale)
    
    A = A.tocoo()

    n = 1.0 / scale    
    d_blockidx_to_vals = defaultdict(list)
    for (i,j,v) in zip(A.row, A.col, A.data):
        d_blockidx_to_vals[ (int(i//n), int(j//n)) ].append(v)

    A_spy = np.zeros( (int(H)+1, int(W)+1), dtype=np.float32)
    for ((i,j), v) in d_blockidx_to_vals.items():
        A_spy[i,j] = np.mean(v)

    return vipy.image.Image(array=A_spy, colorspace='float').mat2gray().maxdim(showdim, interp='nearest').jet()


