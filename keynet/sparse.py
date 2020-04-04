import os
import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import torch
import copy
import torch.sparse
from numpy.linalg import multi_dot 
from collections import defaultdict
from keynet.dense import random_positive_definite_matrix, random_doubly_stochastic_matrix
from joblib import Parallel, delayed
import warnings
from tqdm import tqdm
import xxhash
import vipy
from vipy.util import groupbyasdict, flatlist
from keynet.util import blockview
import keynet.globals


def sparse_affine_to_linear(A, bias=None, dtype=np.float32):
    """Convert affine function Ax+b to linear function Mx such that M = [A b; 0 1]"""
    assert is_scipy_sparse(A)    
    if bias is not None:
        assert bias.shape[0] == A.shape[0] and bias.shape[1] == 1
        lastcol = scipy.sparse.coo_matrix(bias)
    else:
        lastcol = scipy.sparse.coo_matrix((A.shape[0], 1), dtype=dtype )
    lastrow = scipy.sparse.coo_matrix( ([1], ([0], [A.shape[1]])), shape=(1, A.shape[1]+1), dtype=dtype)
    return scipy.sparse.vstack( (scipy.sparse.hstack( (A, lastcol)), lastrow) )


def diagonal_affine_to_linear(A, bias=None, withinverse=False, dtype=np.float32):
    assert is_scipy_sparse(A)
    assert A.shape[0] == A.shape[1]
    n = A.shape[0] + 1
    
    L = sparse_affine_to_linear(A, bias=bias, dtype=np.float64)
    if withinverse:
        # Woodbury matrix inversion lemma (rank one update)
        # (A + uv^T)^{-1} = A^{-1} - \frac{ A^{-1}uv^TA^{-1} }{ 1 + v^T A^{-1} u }

        if bias is not None:
            d = L.diagonal(); d[-1] = 0.5
            Ainv = scipy.sparse.spdiags(1.0 / d, 0, n, n)            
            u = scipy.sparse.csr_matrix(np.vstack( (bias, np.array([0.5]))))
            v = scipy.sparse.csr_matrix(np.hstack( (np.zeros_like(bias).flatten(), np.array([1.0]))))
            Linv = Ainv - ((Ainv.dot(u).dot(v.dot(Ainv)))/float(1+(v.dot(Ainv).dot(u).todense())))
        else:
            Linv = scipy.sparse.spdiags(1.0 / L.diagonal(), 0, n, n).tocoo()
        return (L.astype(dtype), Linv.astype(dtype))
    else:
        return L.astype(dtype)
    

def sparse_toeplitz_conv2d(inshape, f, bias=None, as_correlation=True, stride=1, _row=None):
    """ Returns sparse toeplitz matrix (W) in coo format that is equivalent to per-channel pytorch conv2d (spatial correlation) of filter f with a given image with shape=inshape vectorized
        conv2d(img, f) == np.dot(W, img.flatten()), right multiplied
        Example usage: test_keynet.test_sparse_toeplitz_conv2d()
    """
    
    if _row is None:
        # Parallelize construction by row
        irange = np.arange(0, inshape[1], stride)
        n_processes = keynet.globals.num_processes()
        T = Parallel(n_jobs=n_processes)(delayed(sparse_toeplitz_conv2d)(inshape, f, bias, as_correlation, stride, _row=i) for i in (tqdm(irange) if n_processes>1 and keynet.globals.verbose() else irange))
        (rows, cols, vals) = zip(*[(i,j,v) for t in T for (i,j,v) in zip(t.row, t.col, t.data)])  # merge
        T = scipy.sparse.coo_matrix( (vals, (rows, cols)), shape=(T[0].shape))

        # Sparse matrix with optional bias and affine augmentation         
        if bias is not None:
            (C,U,V) = inshape                
            UV = (U//stride)*(V//stride)
            (row, col, val) = zip(*[(i*UV+j, 0, x) for (i,x) in enumerate(bias) for j in range(0, UV)])
            lastcol = scipy.sparse.coo_matrix( (val, (row, col)), shape=(T.shape[0], 1))
        else:
            lastcol = scipy.sparse.coo_matrix((T.shape[0], 1), dtype=np.float32 )
        lastrow = scipy.sparse.coo_matrix( ([1], ([0], [T.shape[1]])), shape=(1, T.shape[1]+1), dtype=np.float32)
        return scipy.sparse.vstack( (scipy.sparse.hstack( (T, lastcol)), lastrow) )
        
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
        if _row is not None and _row != u:
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
                                row_ind.append( c_outchannel*(U_div_stride)*(V_div_stride) + ku*(V_div_stride) + kv )
                                col_ind.append( c )

    return scipy.sparse.coo_matrix((data, (row_ind, col_ind)), shape=(M*(U//stride)*(V//stride), C*U*V))


def sparse_toeplitz_avgpool2d(inshape, filtershape, stride):
    (outchannel, inchannel, filtersize, filtersize) = filtershape
    (M,U,V) = (inshape)
    F = np.zeros(filtershape, dtype=np.float32)
    for k in range(0,outchannel):
        F[k,k,:,:] = 1.0 / (filtersize*filtersize)
    return sparse_toeplitz_conv2d(inshape, F, bias=None, stride=stride)


def sparse_channelorder_to_blockorder(shape, blockshape, withinverse=True):
    assert isinstance(shape, tuple) and len(shape)==3, "Shape must be (C,H,W) tuple"
    
    (C,H,W) = shape
    if (H*W) % blockshape != 0:
        warnings.warn('[keynet.sparse.sparse_channelorder_to_blockorder]:  Ragged blockorder for blockshape=%d and shape=%s' % (blockshape, str(shape)))

    (H_pad, W_pad) = (int(blockshape*np.ceil((H)/float(blockshape))), int(blockshape*np.ceil((W)/float(blockshape))))    
    img_channelorder = np.array(range(0, H_pad*W_pad)).reshape(H_pad,W_pad)  # HxW, img[0:H] == first row 
    img_blockorder = blockview(img_channelorder, blockshape).flatten()  # (H//B)x(W//B)xBxB, img[0:B*B] == first block
    img_blockorder = img_blockorder[0:H*W]
    (rows, cols, vals) = ([], [], [])
    for c in range(0,C):
        rows.extend(np.array(range(0, H*W)) + c*H*W)
        cols.extend(img_blockorder + c*H*W)
        vals.extend(np.ones_like(img_blockorder))
    A =scipy.sparse.coo_matrix( (vals, (rows, cols)), dtype=np.float32).tocsr()
    return A if not withinverse else (A, A.transpose())
                                            
    
def sparse_block_diagonal(mats, shape=None, format='coo'):
    """Create a sparse matrix with elements in mats as blocks on the diagonal, faster than scipy"""
    if isinstance(mats, np.ndarray) or is_scipy_sparse(mats):
        assert shape is not None
        blocksize = mats.shape        
        mats = [mats]
        (U,V) = shape
        assert U % blocksize[0] == 0 and V % blocksize[1] == 0
    else:
        n = len(mats)
        blocksize = mats[0].shape
        (U,V) = (n*blocksize[0], n*blocksize[1])

    (rows,cols,data) = ([],[],[])
    for (k, (i,j)) in enumerate(zip(range(0,U,blocksize[0]), range(0,V,blocksize[1]))):
        b = scipy.sparse.coo_matrix(mats[k % len(mats)])
        for ii,jj,v in zip(b.row, b.col, b.data):
            rows.append(i+ii)
            cols.append(j+jj)
            data.append(v)
    return scipy.sparse.coo_matrix( (data, (rows, cols)), shape=(U, V)).asformat(format)


def sparse_orthogonal_block_diagonal(mats, shape=None, format='coo', withinverse=False):
    """Create a sparse matrix with elements in mats as blocks on the diagonal, faster than scipy.
       Mat is repeated to fill out shape, assume mats are orthogonal, and provide inverse if requested.
       This assumes that the submatrices in mats are orthogonal matrices, but does not check!
       
       Inputs:
          -mats = [array(), array(), ...] or np.array()
          -shape = (H,W) for final block diagonal shape, must be square and evenly divisible with blocksize of mats
    """
    if isinstance(mats, np.ndarray) or is_scipy_sparse(mats):
        assert shape is not None and shape[0] == shape[1]
        blocksize = mats.shape        
        mats = [mats]
        (U,V) = shape
        assert U % blocksize[0] == 0 and V % blocksize[1] == 0        
    else:
        n = len(mats)
        blocksize = mats[0].shape
        (U,V) = (n*blocksize[0], n*blocksize[1])

    assert blocksize[0] == blocksize[1]
    assert U==V
    
    (rows,cols,data) = ([],[],[])
    for (k, (i,j)) in enumerate(zip(range(0,U,blocksize[0]), range(0,V,blocksize[1]))):
        b = scipy.sparse.coo_matrix(mats[k % len(mats)])
        for ii,jj,v in zip(b.row, b.col, b.data):
            rows.append(i+ii)
            cols.append(j+jj)
            data.append(v)
    P = scipy.sparse.coo_matrix( (data, (rows, cols)), shape=(U, V)).asformat(format)
    return P if not withinverse else (P, P.transpose())


def sparse_identity_matrix(n, dtype=np.float32):
    return scipy.sparse.eye(n, dtype=dtype)


def sparse_identity_matrix_like(A):
    return scipy.sparse.eye(A.shape[0], dtype=A.dtype)


def sparse_permutation_matrix(n, dtype=np.float32, withinverse=False):
    data = np.ones(n).astype(dtype)
    row_ind = list(range(0,n))
    col_ind = np.random.permutation(list(range(0,n)))
    P = csr_matrix((data, (row_ind, col_ind)), shape=(n,n))
    return (P, P.transpose()) if withinverse else P


def sparse_orthogonal_matrix(n, k_iter, balanced=True, withinverse=False):
    """Givens rotations"""
    S = None
    G_index = []
    for k in range(0, k_iter):
        theta = np.random.rand()*2*np.pi
        if not balanced:
            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            G = scipy.sparse.bmat([[R, None], [None, scipy.sparse.eye(n-2)]])
            P = sparse_permutation_matrix(n)
            G = P.dot(G).dot(P.transpose())
        else:
            G_index = np.random.permutation(range(0,n)).tolist()+G_index if len(G_index) <= 1 else G_index            
            G = scipy.sparse.eye(n).todok()
            (i,j) = (G_index.pop(), G_index.pop())
            G[i,i] = np.cos(theta)
            G[i,j] = -np.sin(theta)
            G[j,i] = np.sin(theta)
            G[j,j] = np.cos(theta)
        S = G.dot(S) if S is not None else G
    return S if not withinverse else (S, S.transpose())


def sparse_gaussian_random_diagonal_matrix(n, mu=1, sigma=1, eps=1E-6, withinverse=False):
    """nxn diagonal matrix with diagonal entries sampled from max(N(mu, sigma), eps)"""
    D = scipy.sparse.diags(np.maximum(eps, np.array(sigma*np.random.randn(n)+mu).astype(np.float32)), dtype=np.float32)
    return (D, scipy.sparse.diags(1.0 / D.diagonal())) if withinverse else D


def sparse_uniform_random_diagonal_matrix(n, scale=1, eps=1E-6, dtype=np.float32, withinverse=False):
    """nxn diagonal matrix with diagonal entries sampled from scale*[0,1]+eps"""
    D = scipy.sparse.diags(np.array(scale*np.random.rand(n) + eps))
    return (D.astype(dtype), scipy.sparse.diags(1.0 / D.diagonal()).astype(dtype)) if withinverse else D.astype(dtype)


def sparse_random_doubly_stochastic_matrix(n, k):
    """Birkhoff's theorem states every doubly stochastic matrix is a convex combination of permutation matrices, here we choose k random nxn permutation matrices"""
    A = np.random.rand()*sparse_permutation_matrix(n)
    coef = [np.random.rand() for k in range(0,k)]
    convex_coef = coef / np.sum(coef)
    A = convex_coef[0]*sparse_permutation_matrix(n)
    for c in convex_coef[1:]:
        A = A + c*sparse_permutation_matrix(n)
    return A
    

def sparse_random_diagonally_dominant_doubly_stochastic_toeplitz_matrix(n, k):
    d = np.array([0.25, 0.5, 0.25]).reshape(3,1).dot(np.ones( (1,n) ))
    k_range = [-1, 0, 1]
    A = scipy.sparse.spdiags(d, k_range, n, n, format='coo').todense()

    B = np.tril(A)
    C = np.copy(B)
    np.fill_diagonal(C, 0)

    Ainv = (1.0 / A[0,0])*(B.dot(B.transpose()) - C.dot(C.transpose()))

    print(A)
    print(Ainv)
    print(A.dot(Ainv))
    return A


def sparse_random_diagonally_dominant_doubly_stochastic_matrix(n, k, n_iter=100, withinverse=False):
    """Return sparse matrix (nxn) such that at most k elements per row are non-zero and matrix is positive definite and doubly stochastic.
       For non-negative entries, the inverse is also doubly stochastic with the same sparsity pattern.
    """
    n_iter = 10 if k<=3 else n_iter  # speedup
    d = np.random.rand(k,n)
    d[0,:] = np.maximum(d[0,:], np.sum(d[1:,:], axis=0) + 0.1)  # first column is greater than sum of other columns 
    d = d / np.sum(d,axis=0).reshape(1,n)  # sum over cols
    k_range = list(range(-((k-1)//2), 1+((k-1)//2)) if k%2==1 else list(range(-(k//2), k//2)))
    k_range.remove(0)
    k_range = [0] + k_range  # first row is main diagonal, rest are seqential diagonals below and above main
    A = scipy.sparse.spdiags(d, k_range, n, n, format='csr')
    for k in range(0, n_iter):
        A = normalize(A, norm='l1', axis=0)
        A = normalize(A, norm='l1', axis=1)
    A = sparse_permutation_matrix(n).dot(A).dot(sparse_permutation_matrix(n))
    if withinverse and n > 8096:
        warnings.warn('direct inverse of large matrix (%dx%d)' % (n,n))
    return A if not withinverse else (A, scipy.sparse.coo_matrix(np.linalg.inv(A.todense())))  # expensive
    

def sparse_positive_definite_block_diagonal(n, m, dtype=np.float32, withinverse=False):
    """Return nxn matrix with mxm blocks on main diagonal, each block is a random positive definite matrix"""
    m = np.minimum(n,m)
    B = [random_positive_definite_matrix(m,dtype) for j in np.arange(0,n-m,m)]
    B = B + [random_positive_definite_matrix(n-len(B)*m, dtype)]
    A = sparse_block_diagonal(B, format='csr')
    if withinverse:
        Binv = [np.linalg.inv(b) for b in B]
        Ainv = sparse_block_diagonal(Binv, format='csr')
        return (A,Ainv)
    else:
        return A

    
def is_scipy_sparse(A):
    return scipy.sparse.issparse(A)


def coo_range(A, rowrange, colrange):
    (rows, cols, vals) = zip(*[(i-rowrange[0], j-colrange[0], v) for (i,j,v) in zip(A.row, A.col, A.data) if i>=rowrange[0] and i<rowrange[1] and j>=colrange[0] and j<colrange[1]])
    return scipy.sparse.coo_matrix( (vals, (rows, cols) ), shape=(rowrange[1]-rowrange[0], colrange[1]-colrange[0]))


def spy(A, mindim=256, showdim=1024, range=None, eps=None):
    """Visualize sparse matrix A by resizing the block A[range[0]:range[1], range[0]:range[1]] to (mindim, mindim) then upsapling to (showdim,showdim) and return a vipy.image.Image().
       Elements less than eps are set to zero
    """

    if range is not None:
        assert isinstance(range, tuple) and len(range) == 2, "Range must be tuple (start_dim, end_dim)"
        (i,j) = range
        B = A[i:j, i:j].tocoo()
        return spy(B, mindim, showdim, range=None)
    
    scale = float(mindim) / min(A.shape)

    if eps is not None:
        A = A.tocoo()
        (rows, cols, vals) = zip(*[(i,j,v) for (i,j,v) in zip(A.row, A.col, A.data) if np.abs(v) > eps])
        A = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=A.shape, dtype=A.dtype)
                                                              
    if scale >= 1:
        return vipy.image.Image(array=np.array(A.todense().astype(np.float32)), colorspace='float').mat2gray().maxdim(showdim, interp='nearest').jet()
    else:
        (H, W) = np.ceil(np.array(A.shape)*scale)
        A = A.tocoo()

        n = 1.0 / scale    
        d_blockidx_to_vals = defaultdict(list)
        for (i,j,v) in zip(A.row, A.col, A.data):
            d_blockidx_to_vals[ (int(i//n), int(j//n)) ].append(v)

        A_spy = np.zeros( (int(H)+1, int(W)+1), dtype=np.float32)
        for ((i,j), v) in d_blockidx_to_vals.items():
            A_spy[i,j] = np.mean(v)

        return vipy.image.Image(array=A_spy.astype(np.float32), colorspace='float').mat2gray().maxdim(showdim, interp='nearest').jet()


    
class SparseMatrix(object):
    def __init__(self, A=None):
        self._n_processes = keynet.globals.num_processes()
        assert A is None or self.is_scipy_sparse(A) or self.is_numpy_dense(A), "Invalid input - %s" % (str(type(A)))
        self.shape = A.shape if A is not None else (0,0)  # shape=(H,W)
        # self._matrix = A.tocsr() if self.is_scipy_sparse(A) else A
        self._matrix = A
        self.dtype = A.dtype if A is not None else None
        self.ndim = 2

    def parallel(self, n_processes):
        self._n_processes = n_processes
        return self
    
    def __repr__(self):
        return str('<keynet.SparseMatrix: H=%d, W=%d, backend=%s>' % (self.shape[0], self.shape[1], str(type(self._matrix))))

    def __getitem__(self, k):
        if not (isinstance(k, slice) or isinstance(k, tuple)):
            k = slice(k,k+1)  # force result to be 2D

        if isinstance(k, tuple):
            (ib, ie) = (k[0].start, k[0].stop)
            (jb, je) = (k[1].start, k[1].stop)
            assert k[0].step == 1 or k[0].step is None
            assert k[1].step == 1 or k[1].step is None             
            return SparseMatrix(self.clone()._matrix.tocsr()[ib:ie].transpose()[jb:je].transpose().tocoo())
        else:
            (ib, ie) = (k.start, k.stop)            
            assert k.step == 1 or k.step is None 
            return SparseMatrix(self.clone()._matrix.tocsr()[ib:ie].tocoo())
            
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

    def new(self):
        return SparseMatrix()
    
    def clone(self):
        return copy.deepcopy(self)

    def spy(self):
        return spy(self._matrix)
    
    # Must be overloaded
    def from_torch_dense(self, A):
        assert self.is_torch_dense(A)                
        return SparseMatrix(A.detach().numpy())

    def from_scipy_sparse(self, A):
        assert self.is_scipy_sparse(A)        
        return SparseMatrix(A)
    
    def matmul(self, A):
        assert isinstance(A, SparseMatrix) or self.is_scipy_sparse(A)
        #if self.is_scipy_sparse(self._matrix):
        #    self._matrix = self._matrix.tocoo()
        #if self.is_scipy_sparse(A._matrix):
        #    A._matrix = A._matrix.tocoo()
        self._matrix = scipy.sparse.coo_matrix.dot(self._matrix, A._matrix if isinstance(A, SparseMatrix) else A)
        self.shape = self._matrix.shape
        return self

    def dot(self, x_numpy):
        assert self.is_numpy_dense(x_numpy)
        #if self.is_scipy_sparse(self._matrix):
        #    self._matrix = self._matrix.tocsr()
        return scipy.sparse.coo_matrix.dot(self._matrix, np.matrix(x_numpy))
        
    def torchdot(self, x_torch):
        assert self.is_torch_dense(x_torch)
        #if self.is_scipy_sparse(self._matrix):
        #    self._matrix = self._matrix.tocsr()
        return torch.as_tensor(scipy.sparse.coo_matrix.dot(self._matrix, np.matrix(x_torch.detach().numpy())))

    def nnz(self):
        return self._matrix.nnz if self.is_scipy_sparse(self._matrix) else self._matrix.size

    def transpose(self):
        self._matrix = self._matrix.transpose()
        self.shape = self._matrix.shape
        return self

    def tocoo(self):
        return self._matrix.tocoo() if is_scipy_sparse(self._matrix) else scipy.sparse.coo_matrix(self._matrix)

    def tocsr(self):
        self._matrix = self._matrix.tocsr()
        return self

    def tocsc(self):
        self._matrix = self._matrix.tocsc()
        return self

    def from_torch_conv2d(self, inshape, w, b, stride):
        return SparseMatrix(sparse_toeplitz_conv2d(inshape, w.detach().numpy(), bias=b.detach().numpy(), stride=stride))


class SparseTiledMatrix(SparseMatrix):
    def __init__(self, tilesize=None, coo_matrix=None, tilediag=None, shape=None, slice=None):
        self._n_processes = keynet.globals.num_processes()
        self.dtype = None
        self.shape = (0,0) if shape is None else shape
        self.ndim = 2
        self._tilesize = 0 if tilesize is None else tilesize
        self._d_blockhash_to_tile = {}
        self._blocklist = []

        if coo_matrix is not None and tilesize is not None:
            self.from_coomatrix(coo_matrix, tilesize, slice=slice)
        elif tilediag is not None and shape is not None:
            self.from_tilediag(shape, tilediag)
        
    def __getitem__(self, k):
        if not (isinstance(k, slice) or isinstance(k, tuple)):
            k = slice(k,k+1)  # force result to be 2D

        m = self.clone()        
        (H,W) = m.shape
        if isinstance(k, tuple):
            (ib, ie) = (k[0].start // m.tilesize(), k[0].stop // m.tilesize())
            (jb, je) = (k[1].start // m.tilesize(), k[1].stop // m.tilesize())
            m.shape = (k[0].stop - k[0].start, k[1].stop - k[1].start)
            assert k[0].step == 1 or k[0].step is None
            assert k[1].step == 1 or k[1].step is None             
        else:
            (ib, jb) = (k.start // m.tilesize(), 0)
            (ie, je) = (k.stop // m.tilesize(), W // m.tilesize())
            assert k.step == 1 or k.step is None
            m.shape = (k.stop - k.start, W)
            
        m._blocklist = [(i,j,k) for (i,j,k) in m._blocklist if i>=ib and i<=ie and j>=jb and j<=je]
        m._d_blockhash_to_tile = {k:t for (k,t) in m._d_blockhash_to_tile.items() if k in set([k for (i,j,k) in m._blocklist])}
        return m

    def _block(self, B):
        return SparseMatrix(B)
    
    def __repr__(self):
        return str('<keynet.SparseTiledMatrix: H=%d, W=%d, tilesize=%d, tiles=%d>' % (*self.shape, self.tilesize(), len(self.tiles())))

    def tilesize(self):
        return self._tilesize

    def tiles(self):
        return list(self._d_blockhash_to_tile.values())

    def parallel(self, n_processes):
        self._n_processes = n_processes
        return self

    def new(self):
        return SparseTiledMatrix(tilesize=self.tilesize())
        
    def is_tiled_sparse(self, A):
        return isinstance(A, SparseTiledMatrix)

    def from_torch_conv2d(self, inshape, w, b, stride):        
        return SparseTiledMatrix(coo_matrix=sparse_toeplitz_conv2d(inshape, w.detach().numpy(), bias=b.detach().numpy(), stride=stride), tilesize=self._tilesize)
                    
    def from_torch_dense(self, A):
        assert self.is_torch_dense(A)
        return SparseTiledMatrix(coo_matrix=scipy.sparse.coo_matrix(A.detach().numpy()), tilesize=self.tilesize())

    def from_scipy_sparse(self, A):
        assert self.is_scipy_sparse(A)
        return SparseTiledMatrix(coo_matrix=A.tocoo(), tilesize=self.tilesize())

    def from_coomatrix(self, T, tilesize, slice=None):
        """Given a sparse matrix T, split into non-overlapping nxn blocks or 'tiles' of size self._tilesize x self.Blocksize, and 
           return an indexed representation for unique submatrices which provides memory efficient matrix vector multiplication when T is self-similar.
        
        Representation
            B = [(i,j,k),...] for block index (i,j) with submatrix key k
            M = {k:np.array(), ...} a submatrix dictionary, such that the submatrix for block (i,j)=(B[u][0], B[u][1]) is M[B[u][2]]
        
        """
        assert self.is_scipy_sparse(T), "input must be scipy.sparse.coo_matrix()"

        self._tilesize = tilesize
        self.dtype = T.dtype
        self.shape = (T.shape[0], T.shape[1])
        self.ndim = 2

        T = T.tocoo()
        
        if slice is None and self._n_processes > 1:
            n_rows_per_process = max(self.tilesize(), int(self.tilesize() * np.floor((T.shape[0]//(4*self._n_processes)) / self.tilesize())))  # must be multiple of tilesize
            arange = np.arange(0, T.shape[0], n_rows_per_process)
            T_rows = Parallel(n_jobs=self._n_processes, max_nbytes=1e6, mmap_mode='r')(delayed(SparseTiledMatrix)(tilesize=tilesize, coo_matrix=T, slice=((i, min(i + n_rows_per_process, T.shape[0])), (0, T.shape[1]))) for i in (tqdm(arange) if keynet.globals.verbose() else arange))
            self._d_blockhash_to_tile = {k:b for t in T_rows for (k,b) in t._d_blockhash_to_tile.items()}  # unique is overwritten
            self._blocklist = [b for t in T_rows for b in t._blocklist]  # merge 
            if keynet.globals.verbose():
                print('[keynet.SparseTiledMatrix]: from_coomatrix = %s' % str(self))
            return self

        n = tilesize
        (H,W) = self.shape        
        ((ib,ie), (jb,je)) = slice if slice is not None else ((0,H), (0,W))
        d_blockidx_to_hash = defaultdict(int)        
        for (i,j,v) in zip(T.row, T.col, T.data):
            if i>=ib and i<ie and j>=jb and j<je:
                (bi, bj) = (i//n, j//n)
                d_blockidx_to_hash[(bi, bj)] += xxhash.xxh32_intdigest(str((i-bi*n, j-bj*n, v)))
                trimshape = (min(H-i, n), min(W-j, n))
                if trimshape != (n, n):
                    d_blockidx_to_hash[(bi, bj)] += xxhash.xxh32_intdigest(str(trimshape))

        d_hash_to_blockidx = dict()
        d_blockidx_to_data = defaultdict(list)
        for (i,j,v) in zip(T.row, T.col, T.data):
            if i>=ib and i<ie and j>=jb and j<je:
                (bi, bj) = (i//n, j//n)
                h = d_blockidx_to_hash[(bi,bj)]
                if h not in d_hash_to_blockidx:
                    d_hash_to_blockidx[h] = (bi,bj)  # once
                if (bi,bj) == d_hash_to_blockidx[h]:
                    d_blockidx_to_data[(bi,bj)].append( (i-bi*n, j-bj*n, v) )  # unique tiles only

        self._d_blockhash_to_tile = dict()
        self._blocklist = []
        for ((bi,bj),h) in d_blockidx_to_hash.items():
            if (bi,bj) in d_blockidx_to_data:
                (blockrows, blockcols, vals) = zip(*d_blockidx_to_data[(bi,bj)])
                trimshape = (min(H-bi*n, n), min(W-bj*n, n))
                self._d_blockhash_to_tile[h] = self._block(scipy.sparse.coo_matrix( (vals, (blockrows, blockcols)), shape=trimshape))                
            self._blocklist.append( (bi, bj, h) )

        return self
    

    def from_tilediag(self, shape, T):
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
            (H_clip, W_clip) = (min(H, i*n+n), min(W, j*n+n))
            y[i*n:H_clip, :] += self._d_blockhash_to_tile[k].tocsr().torchdot(x[j*n:W_clip, :])
        return y
                
    def matmul(self, other):
        """For two Tiled() object T1, T2, compute T1.dot(T2) and save in T1"""
        assert isinstance(other, SparseTiledMatrix), "Invalid input - Must be SparseMatrix()"
        assert other.shape[0] == self.shape[1], "Non-conformal shape"        
        
        assert other._tilesize == self._tilesize, "Non-conformal tilesize"    
        n = self.tilesize()
        (H,W) = self.shape

        # Accumulate
        M_accum = {}
        M_hash = {}
        d_product = {}        
        for (i, jj, v) in self._blocklist:
            if keynet.globals.verbose():
                print('[keynet.SparseTiledMatrix][%d/%d]: Product "%s" * "%s"...' % (i,jj,str(self), str(other)))
            for (ii, j, vo) in other._blocklist:
                if jj == ii and v is not None and vo is not None:
                    if (v,vo) not in d_product:
                        d_product[(v,vo)] = self._d_blockhash_to_tile[v].tocsr().clone().matmul(other._d_blockhash_to_tile[vo].tocsc())   # cache
                        d_product[(v,vo)] = d_product[(v,vo)].tocoo().todense()  # faster add?  why not just make everything dense?
                    if (i,j) not in M_accum:
                        M_accum[(i,j)] = d_product[(v,vo)]
                        M_hash[(i,j)] = hash((v,vo))
                    else:
                        M_accum[(i,j)] += d_product[(v,vo)]
                        M_hash[(i,j)] += hash( (v,vo) )


        (B, M) = ([], {})
        for ((i,j), m) in M_accum.items():
            k = M_hash[(i,j)]
            if k not in M:
                M[k] = self._block(scipy.sparse.coo_matrix(m))
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
    
    def _tosparse(self, format='coo'):
        """Convert to Scipy COOrdinate sparse matrix, this is an expensive operation that should be used for small matrices only and for testing purposes"""
        ((H,W), n) = (self.shape, self._tilesize)
        d_blockhash_to_coo = {k:v.tocoo() for (k,v) in self._d_blockhash_to_tile.items()}
        d = {(i*n, j*n):d_blockhash_to_coo[k] for (i,j,k) in self._blocklist}
        #B = [ [d[(i,j)] if (i,j) in d else None for j in range(0, W, max(n,1))] for i in range(0, H, max(n,1))]
        (rows, cols, vals) = zip(*[(i+ii,j+jj,v) for ((ii,jj), b) in d.items() for (i,j,v) in zip(b.row, b.col, b.data)])
        #return scipy.sparse.bmat(B, format=format) if len(B)>0 else scipy.sparse.coo_matrix((H,W))
        return scipy.sparse.coo_matrix( (vals, (rows, cols)), shape=(H,W))

    def tocoo(self):
        return self._tosparse()

    def tocsr(self):
        return self._tosparse(format='csr')

    def tocsc(self):
        return self._tosparse(format='csc')

    def nnz(self):
        return sum([m.nnz() for m in self._d_blockhash_to_tile.values()])

    def spy(self, mindim=256, showdim=1024):
        return spy(self.tocoo(), mindim, showdim)

