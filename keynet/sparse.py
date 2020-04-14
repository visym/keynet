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
from joblib import Parallel, delayed, parallel_backend
import warnings
from tqdm import tqdm
import xxhash
import vipy
from vipy.util import groupbyasdict, flatlist, Stopwatch
from keynet.util import blockview
import keynet.globals
import itertools
import tempfile
import numba
import numba.typed


def scipy_coo_to_torch_sparse(coo, device='cpu'):
    """https://stackoverflow.com/questions/50665141/converting-a-scipy-coo-matrix-to-pytorch-sparse-tensor"""

    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    if device == 'cpu':
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))
    else:
        with torch.cuda.device(device):
            return torch.sparse.FloatTensor(i, v, torch.Size(shape)).cuda(device)


def sparse_channelorder_to_pixelorder_matrix(shape, withinverse=False):
    """Return permutation matrix that will convert an image in CxHxW memory layout (channel order) to HxWxC memory layout (pixel order)"""
    (C,H,W) = shape 
    img = np.array(range(0, np.prod(shape))).reshape(shape)
    img_permute = np.moveaxis(img, 0, 2)   # CxHxW -> HxWxC    
    cols = img_permute.flatten()    
    rows = range(0, len(cols))
    vals = np.ones_like(rows)
    P = scipy.sparse.coo_matrix( (vals, (rows, cols)), shape=(np.prod(img.shape), np.prod(img.shape)), dtype=np.float32)
    return P if not withinverse else (P, P.transpose())


def sparse_channelorder_to_blockorder_matrix(shape, blocksize, withinverse=True):
    """Return permytation matrix that will convert an image in CxHxW channel order to Cx(H//N)x(W//N)xNxN for blocksize=N"""
    
    assert isinstance(shape, tuple) and len(shape)==3, "Shape must be (C,H,W) tuple"
    
    (C,H,W) = shape
    if (H*W) % blocksize != 0:
        warnings.warn('[keynet.sparse.sparse_channelorder_to_blockorder]:  Ragged blockorder for blocksize=%d and shape=%s' % (blocksize, str(shape)))

    (H_pad, W_pad) = (int(blocksize*np.ceil((H)/float(blocksize))), int(blocksize*np.ceil((W)/float(blocksize))))    
    img_channelorder = np.array(range(0, H_pad*W_pad)).reshape(H_pad,W_pad)  # HxW, img[0:H] == first row 
    img_blockorder = blockview(img_channelorder, blocksize).flatten()  # (H//B)x(W//B)xBxB, img.flatten()[0:B*B] == first block
    img_blockorder = img_blockorder[0:H*W]
    (rows, cols, vals) = ([], [], [])
    for c in range(0,C):
        rows.extend(np.array(range(0, H*W)) + c*H*W)
        cols.extend(img_blockorder + c*H*W)
        vals.extend(np.ones_like(img_blockorder))
    A =scipy.sparse.coo_matrix( (vals, (rows, cols)), dtype=np.float32).tocsr()
    return A if not withinverse else (A, A.transpose())


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
    

@numba.jit(nopython=True, parallel=False)
def _sparse_toeplitz_conv2d(inshape, f, as_correlation, stride):
    (C,U,V) = inshape
    (M,C,P,Q) = f.shape
    C_range = range(0,C) if as_correlation else range(C-1, 0-1, -1)
    M_range = range(0,M) if as_correlation else range(M-1, 0-1, -1)
    P_range = range(-((P-1)//2), ((P-1)//2) + 1) if P%2==1 else range(-((P-1)//2), ((P-1)//2) + 2)
    P_range = P_range if as_correlation else range(max(list(P_range))-1, min(list(P_range))-1, -1)
    Q_range = range(-((Q-1)//2), ((Q-1)//2) + 1) if P%2==1 else range(-((Q-1)//2), ((Q-1)//2) + 2)
    Q_range = Q_range if as_correlation else range(max(list(Q_range))-1, min(list(Q_range))-1, -1)
    (U_div_stride, V_div_stride) = (U//stride, V//stride)
    
    rows = np.zeros(U*V*C*M*P*Q, dtype=np.int32)
    cols = np.zeros(U*V*C*M*P*Q, dtype=np.int32)
    vals = np.zeros(U*V*C*M*P*Q, dtype=np.float32)
    k_entry = 0
    
    # For every image_row
    for (ku,u) in enumerate(np.arange(0,U,stride)):
        # For every image_column
        for (kv,v) in enumerate(np.arange(0,V,stride)):
            # For every inchannel (transposed)
            for (k_inchannel, c_inchannel) in enumerate(C_range):
                # For every kernel_row (transposed)
                for (i,p) in enumerate(P_range):
                    if not ((u+p)>=0 and (u+p)<U):
                        continue  
                    # For every kernel_col (transposed)
                    for (j,q) in enumerate(Q_range):
                        # For every outchannel
                        if ((v+q)>=0 and (v+q)<V):
                            #c = np.ravel_multi_index( (c_inchannel, u+p, v+q), (C,U,V) )
                            c = c_inchannel*U*V + (u+p)*V + (v+q)
                            for (k_outchannel, c_outchannel) in enumerate(M_range):
                                rows[k_entry] = int(c_outchannel*(U_div_stride)*(V_div_stride) + ku*(V_div_stride) + kv)
                                cols[k_entry] = int(c)
                                vals[k_entry] = float(f[k_outchannel,k_inchannel,i,j])
                                k_entry += 1  # impure loop

    return (rows[0:k_entry], cols[0:k_entry], vals[0:k_entry])


def sparse_toeplitz_conv2d(inshape, f, bias=None, as_correlation=True, stride=1, format='csr'):
    """ Returns sparse toeplitz matrix (W) in coo format that is equivalent to per-channel pytorch conv2d (spatial correlation) of filter f with a given image with shape=inshape vectorized
        conv2d(img, f) == np.dot(W, img.flatten()), right multiplied
        Example usage: test_keynet.test_sparse_toeplitz_conv2d()
        
        input:
          -inshape=(inchannels, imageheight, imagewidth)
          -f.shape = (outchannels, inchannels, kernelheight, kernelwidth)
    """

    # Valid shapes
    assert(len(inshape) == 3 and len(f.shape) == 4)  # 3D tensor inshape=(inchannels, height, width)
    assert(f.shape[1] == inshape[0])  # equal inchannels
    assert(f.shape[2]==f.shape[3] and f.shape[2]%2 == 1)  # filter is square, odd (FIXME)
    if bias is not None:
        assert(len(bias.shape) == 1 and bias.shape[0] == f.shape[0])  # filter and bias have composable shapes
    (C,U,V) = inshape
    (M,C,P,Q) = f.shape

    # Sparse matrix with optional bias and affine augmentation             
    (row_ind, col_ind, data) = _sparse_toeplitz_conv2d(inshape, f, as_correlation, stride)
    A = scipy.sparse.coo_matrix((data, (row_ind, col_ind)), shape=(M*(U//stride)*(V//stride), C*U*V))

    # Optional bias and affine augmentation         
    if bias is not None:
        (C,U,V) = inshape                
        UV = (U//stride)*(V//stride)
        (row, col, val) = zip(*[(i*UV+j, 0, x) for (i,x) in enumerate(bias) for j in range(0, UV)])
        lastcol = scipy.sparse.coo_matrix( (val, (row, col)), shape=(A.shape[0], 1))
        lastrow = scipy.sparse.coo_matrix( ([1], ([0], [A.shape[1]])), shape=(1, A.shape[1]+1), dtype=np.float32)
        A = scipy.sparse.vstack( (scipy.sparse.hstack( (A, lastcol)), lastrow) )            

    if format == 'csr':
        A = A.tocsr()
    return A


def sparse_toeplitz_avgpool2d(inshape, filtershape, stride):
    (outchannel, inchannel, filtersize, filtersize) = filtershape
    (M,U,V) = (inshape)
    F = np.zeros(filtershape, dtype=np.float32)
    for k in range(0,outchannel):
        F[k,k,:,:] = 1.0 / (filtersize*filtersize)
    return sparse_toeplitz_conv2d(inshape, F, bias=np.zeros(outchannel, dtype=np.float32), stride=stride)


def sparse_block_diagonal(mats, shape=None, format='coo', dtype=np.float32):
    """Create a sparse matrix with elements in mats as blocks on the diagonal, faster than scipy"""
    if isinstance(mats, np.ndarray) or is_scipy_sparse(mats):
        assert shape is not None
        blocksize = mats.shape        
        mats = [mats]
        (U,V) = shape
    else:
        n = len(mats)
        blocksize = mats[0].shape
        (U,V) = (n*blocksize[0], n*blocksize[1]) if shape is None else shape

    (rows,cols,data) = ([],[],[])
    for (k, (i,j)) in enumerate(zip(range(0,U,blocksize[0]), range(0,V,blocksize[1]))):
        b = scipy.sparse.coo_matrix(mats[k % len(mats)])
        for ii,jj,v in zip(b.row, b.col, b.data):
            if (i+ii) < U and (j+jj) < V:            
                rows.append(i+ii)
                cols.append(j+jj)
                data.append(v)
    return scipy.sparse.coo_matrix( (data, (rows, cols)), shape=(U, V)).asformat(format)


def sparse_orthogonal_block_diagonal(mats, shape=None, format='coo', withinverse=False, dtype=np.float32):
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
    else:
        n = len(mats)
        blocksize = mats[0].shape
        (U,V) = (n*blocksize[0], n*blocksize[1]) if shape is None else shape

    assert blocksize[0] == blocksize[1]
    assert U==V
    
    (rows,cols,data) = ([],[],[])
    for (k, (i,j)) in enumerate(zip(range(0,U,blocksize[0]), range(0,V,blocksize[1]))):
        b = scipy.sparse.coo_matrix(mats[k % len(mats)])
        for ii,jj,v in zip(b.row, b.col, b.data):
            if (i+ii) < U and (j+jj) < V:
                rows.append(i+ii)
                cols.append(j+jj)
                data.append(v)
    P = scipy.sparse.coo_matrix( (data, (rows, cols)), shape=(U, V)).asformat(format)
    return P.astype(dtype) if not withinverse else (P.astype(dtype), P.transpose().astype(dtype))


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


def sparse_orthogonal_matrix(n, k_iter, balanced=True, withinverse=False, dtype=np.float32):
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
    return S.astype(dtype) if not withinverse else (S.astype(dtype), S.transpose().astype(dtype))


def sparse_gaussian_random_diagonal_matrix(n, mu=1, sigma=1, eps=1E-6, withinverse=False, dtype=np.float32):
    """nxn diagonal matrix with diagonal entries sampled from max(N(mu, sigma), eps)"""
    D = scipy.sparse.diags(np.maximum(eps, np.array(sigma*np.random.randn(n)+mu)))
    return (D.astype(dtype), scipy.sparse.diags(1.0 / D.diagonal()).astype(dtype))if withinverse else D.astype(dtype)


def sparse_uniform_random_diagonal_matrix(n, scale=1, bias=0, eps=1E-6, dtype=np.float32, withinverse=False):
    """nxn diagonal matrix with diagonal entries sampled from scale*[0,1]+eps+bias"""
    D = scipy.sparse.diags(np.array(scale*np.random.rand(n) + eps + bias))
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
        assert A is None or self.is_scipy_sparse(A) or self.is_numpy_dense(A), "Invalid input - %s" % (str(type(A)))
        self.shape = A.shape if A is not None else (0,0)  # shape=(H,W)
        self._matrix = A
        self.dtype = A.dtype if A is not None else None
        self.ndim = 2

    def __repr__(self):
        return str('<keynet.SparseMatrix: H=%d, W=%d, backend=%s>' % (self.shape[0], self.shape[1], str(type(self._matrix))))

    def __add__(self, other):
        assert isinstance(other, SparseMatrix), "Invalid input"
        assert self.shape == other.shape, "Invalid shape"
        self._matrix += other._matrix
        return self
        
    def is_torch_sparse(self, x):
        return isinstance(x, torch.sparse.FloatTensor)

    def is_torch_dense_float32(self, x):
        return isinstance(x, torch.FloatTensor)

    def is_scipy_sparse(self, x):
        return scipy.sparse.issparse(x)

    def is_torch(self, x):
        return self.is_torch_sparse(x) or self.is_torch_dense_float32(x)
    
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
        assert self.is_torch_dense_float32(A)                
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
        assert self.is_torch_dense_float32(x_torch)
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


class TiledMatrix(SparseMatrix):

    def __init__(self, T, tileshape):
        """Given a sparse matrix T, split into non-overlapping nxn blocks or 'tiles' of size self._tileshape x self.Blocksize, and 
           return an indexed representation for unique submatrices which provides memory efficient matrix vector multiplication when T is self-similar.
        
        Representation
            B = [(i,j,k),...] for block index (i,j) with submatrix key k
            M = {k:np.array(), ...} a submatrix dictionary, such that the submatrix for block (i,j)=(B[u][0], B[u][1]) is M[B[u][2]]
        
        """
        assert self.is_scipy_sparse(T), "input must be scipy.sparse.coo_matrix()"
        assert isinstance(tileshape, tuple) and len(tileshape)==2 and tileshape[0] > 0 and tileshape[1] > 0, "tileshape must be tuple (tileheight, tilewidth) > 0"
        
        self._tileshape = tileshape
        self.dtype = T.dtype
        self.shape = (T.shape[0], T.shape[1])
        self.ndim = 2

        sw = Stopwatch()
        T = T.tocoo()
        (h,w) = tileshape
        (H,W) = self.shape        
        print('[TiledMatrix] tocoo=%f' % sw.since())

        sw = Stopwatch()
        d_blockidx_to_tile = defaultdict(list)
        for (i,j,v) in zip(T.row, T.col, T.data):
            (ii,jj,vv) = (int(i), int(j), float(v))  # casting to native python types 
            (bi,bj) = (ii//h, jj//w)
            d_blockidx_to_tile[(bi, bj)].append( (ii-bi*h, jj-bj*w, vv) )                
        print('[TiledMatrix] group=%f' % sw.since())

        sw = Stopwatch()
        d_blockhash_to_index = dict()
        self._tiles = []
        self._blocks = []
        for ((bi,bj), ijv) in d_blockidx_to_tile.items():
            blockshape = (tileshape[0] if ((bi*h + h) <= H) else (H-bi*h),
                          tileshape[1] if ((bj*w + w) <= W) else (W-bj*w))
            blockhash = hash(str(sorted(ijv, key=lambda x: (x[0], x[1])))+str(blockshape))
            if blockhash not in d_blockhash_to_index:
                (row, col, val) = zip(*ijv)
                self._tiles.append(self._tiletype(scipy.sparse.coo_matrix( (val, (row, col)), shape=blockshape)))
                d_blockhash_to_index[blockhash] = len(self._tiles)-1
            self._blocks.append( (bi*h, bj*w, d_blockhash_to_index[blockhash]) )

        print('[TiledMatrix] hash=%f' % sw.since())

        sw = Stopwatch()
        self._blocks = sorted(self._blocks, key=lambda x: (x[0], x[1]))  # rowmajor order
        print('[TiledMatrix] sorted=%f' % sw.since())

    def __repr__(self):
        return str('<keynet.TiledMatrix: H=%d, W=%d, tileshape=%s, tiles=%d>' % (*self.shape, str(self.tileshape()), len(self.tiles())))

    def __iter__(self):
        for (i,j,k) in self._blocks:
            yield (i, j, k)
        
    def _tiletype(self, B):
        #return SparseMatrix(B.astype(np.float32))
        return B.astype(np.float32)
    
    def tileshape(self):
        return self._tileshape

    def tiles(self):
        return self._tiles

    def blocks(self):
        return list(self.__iter__())  # expensive!
        
    def dot(self, x):
        assert self.is_numpy_dense(x)
        return self.torchdot(torch.as_tensor(x)).numpy()
    
    def copy(self, blocks, tiles):
        self._blocks = blocks
        self._tiles = tiles
        return self


    def torchdot(self, x):
        """Input is (C*H*W+1)xN tensor, compute right matrix multiplication T*x, return (-1)xN"""
        assert self.shape[1] == x.shape[0], "Non-conformal shape for W=%s, x=%s" % (str(self.shape), str(x.shape))

        if not self.is_numpy_dense(x):
            x = x.detach().numpy()

        with Stopwatch() as sw:
            W = self.tocsr()   # slow for large matrices
        print('[TiledMatrix.torchdot]: tocsr=%f' % sw.since())
        with Stopwatch() as sw:
            y = W.dot(x).astype(np.float32)  # MKL multi-threaded with scipy-intel package
        print('[TiledMatrix.torchdot]: dot=%f' % sw.since())
        return torch.as_tensor(y) 
                        
    def transpose(self):
        self._blocks = [(j,i,k) for (i,j,k) in self._blocks] if self._blocks is not None else self._blocks
        self._tiles = [t.transpose() for t in self._tiles]
        self._tileshape = (self._tileshape[1], self._tileshape[0])
        self.shape = (self.shape[1], self.shape[0])
        return self

    def tosparse(self, format='coo'):
        """Convert to Scipy COOrdinate sparse matrix, this is an expensive operation that should be used for small matrices only and for testing purposes"""
        ((H,W), (h,w)) = (self.shape, self._tileshape)
        tiles = [t.tocoo() for t in self._tiles]  # preconvert, expensive for small range

        (rows, cols, vals) = ([],[],[])
        for (ii, jj, k) in self.__iter__():
            b = tiles[k]
            rows.extend(ii + b.row)
            cols.extend(jj + b.col)
            vals.extend(b.data)

        if format == 'csr':
            T = scipy.sparse.csr_matrix( (vals, (rows, cols)), shape=(H,W))
        elif format == 'coo':
            T = scipy.sparse.coo_matrix( (vals, (rows, cols)), shape=(H,W))
        elif format == 'csc':
            T = scipy.sparse.csc_matrix( (vals, (rows, cols)), shape=(H,W))
        else:
            raise ValueError('Invalid format "%s" - must be ["coo", "csr", "csc"]' % format)
        return T            

    def tocsr(self):
        return self.tosparse(format='csr')

    def tocoo(self):
        return self.tosparse(format='coo')
    
    def nnz(self):
        return sum([t.nnz for t in self._tiles])

    def spy(self, mindim=256, showdim=1024):
        return spy(self.tocoo(), mindim, showdim)



class DiagonalTiledMatrix(TiledMatrix):
    def __init__(self, B, shape):
        """Construct block diagonal matrix from block B repeated down the main diagonal"""
        assert B.ndim == 2, "Invalid block, must be 2D"
        assert isinstance(shape, tuple) and len(shape) == 2, "invalid shape"
        if B.shape[0] > shape[0] or B.shape[1] > shape[1]:
            B = B.tocsr()[0:shape[0], 0:shape[1]]

        self._tileshape = B.shape
        self.shape = shape
        self.dtype = B.dtype
        self.ndim = 2

        (H,W) = shape                    
        (h,w) = self._tileshape
        self._tiles = [self._tiletype(B)] 
        self._blocks = None
        
        if (H % h != 0) or (W % w != 0):
            self._tiles.append(self._tiletype(scipy.sparse.eye(max(h,w)).tocsr()[0:H%h, 0:W%w].astype(np.float32)))

    def __iter__(self):
        """Iterator for ((bi,bj), b) tuples along diagonal"""
        ((H,W), (h,w)) = (self.shape, self._tileshape)
        for (i, j) in zip(range(0, H, h), range(0, W, w)):
            yield (i, j, 0) if (i+h<H and j+w<W) else (i, j, len(self._tiles)-1)


class Conv2dTiledMatrix(TiledMatrix):    
    def __init__(self, T, inshape, outshape, tileshape, bias):
        (Cin, Hin, Win) = inshape
        (Cout, Hout, Wout) = outshape
        self._inshape = inshape
        self._outshape = outshape
        self._tileshape = tileshape
        self.shape = T.shape

        assert tileshape[0] <= T.shape[0] and tileshape[1] <= T.shape[1]
        
        if bias:
            assert T.shape[0] == np.prod(outshape) + 1
            assert T.shape[1] == np.prod(inshape) + 1
            assert (self.shape[0]-1) % tileshape[0] == 0 and (self.shape[1]-1) % tileshape[1] == 0            
        else:
            assert T.shape[0] == np.prod(outshape)
            assert T.shape[1] == np.prod(inshape)
            assert self.shape[0] % tileshape[0] == 0 and self.shape[1] % tileshape[1] == 0            
        
        # Channel tile structure
        sw = Stopwatch()
        T = T.tocsr()
        print('[TiledMatrix.init]: tocsr=%f seconds' % sw.since())

        sw = Stopwatch()
        ((H,W), (h,w)) = (self.shape, self._tileshape)
        for cout in range(0, Cout):
            for cin in range(0, Cin):
                (ib, ie) = (cout*((Hout*Wout)), (cout+1)*(Hout*Wout))
                (jb, je) = (cin*Hin*Win, (cin+1)*(Hin*Win))
                if (cout == 0 and cin == 0):
                    C = T[ib:ie, jb:je]
                    C = TiledMatrix(C, tileshape)

                    self._tiles = C._tiles  # unique per channel
                    self._uniqueblocks = [v[0] for (k,v) in groupbyasdict(C._blocks, lambda x: x[2]).items()]                    
                    self._blocks = [(i, j, k, (Hout*Wout, Hin*Win, len(self._tiles)), (Cout, Cin), 0) for (i,j,k) in C._blocks]  # repeated with stride                    
                else:
                    for (i, j, k) in self._uniqueblocks:
                        self._tiles.append(self._tiletype(T[ib+i:ib+min(i+h, H), jb+j:jb+min(j+w, W)]))  # triggers copy, unavoidably slow
                              
        print('[TiledMatrix.init]: tile=%f seconds' % sw.since())

        sw = Stopwatch()
        # Bias tile structure
        if bias:
            B = TiledMatrix(T[:,-1], tileshape=(self._tileshape[0], 1))
            #assert len(B.tiles()) <= Cout+1    # FIXME: why is this wrong?
            self._blocks += [(i, Cin*Hin*Win, k, (0,0,0), (1,1), len(self._tiles)) for (i,j,k) in B._blocks]  # repeated tiles, FIXME: repeated structure 
            self._tiles += B._tiles

        print('[TiledMatrix.init]: bias=%f seconds' % sw.since())

        # Rowmajor order
        sw = Stopwatch()
        self._blocks = sorted(self._blocks, key=lambda x: (x[0], x[1]))
        print('[TiledMatrix.init]: sort=%f seconds' % sw.since())        
        
    def __iter__(self):
        """Iterator for ((bi,bj), b) tuples with blocks repeated across channels"""
        for (i, j, k, (si,sj,sk), (Ni,Nj), k_tileoffset) in self._blocks:
            for ni in range(0, Ni):  # row repetition 
                for nj in range(0, Nj):  # col repetition 
                    yield (i + si*ni, j + sj*nj, k + k_tileoffset + sk*(ni*Nj+nj))  # offset = stride*repetitions

    @staticmethod
    @numba.jit(nopython=True, parallel=False)
    def _tosparse(tiles, blocks):

        # Get memory size
        k_rcd = 0
        #k_rcd_blockoffset = np.zeros(len(blocks), dtype=np.int64)
        # <__iter__>
        for (kb, (i, j, k, (si,sj,sk), (Ni,Nj), k_tileoffset)) in enumerate(blocks):
            #k_rcd_blockoffset[kb] = k_rcd  # for parallelization
            for ni in range(0, Ni):  # row repetition 
                for nj in range(0, Nj):  # col repetition 
                    kt = k + k_tileoffset + sk*(ni*Nj+nj)  # offset = stride*repetitions                    
                    # </__iter__>
                    k_rcd += len(tiles[kt][0])

        # Allocate memory (parallelizable)
        (rows, cols, data) = (np.zeros(k_rcd, dtype=np.int32), np.zeros(k_rcd, dtype=np.int32), np.zeros(k_rcd, dtype=np.float32))

        # Create sparse array indexes
        kb = 0  # numba requirement?
        k_rcd = 0  # numba requirement?
        # <__iter__>
        for kb in numba.prange(0, len(blocks)):   # parallel 
            (i, j, k, (si,sj,sk), (Ni,Nj), k_tileoffset) = blocks[kb]
            #k_rcd = int(k_rcd_blockoffset[kb])  # parallel loop offset
            for ni in range(0, Ni):  # row repetition (parallel)
                for nj in range(0, Nj):  # col repetition 
                    (it, jt, kt) = (i + si*ni, j + sj*nj, k + k_tileoffset + sk*(ni*Nj+nj))  # offset = stride*repetitions                    
                    # </__iter__>

                    t = tiles[kt]
                    n_rcd = len(t[0])
                    for tk in range(0, n_rcd):
                        rows[k_rcd] = it+t[0][tk]  
                        cols[k_rcd] = jt+t[1][tk]
                        data[k_rcd] = t[2][tk]
                        k_rcd += 1  # impure loop, but offset is assigned in outer loop

        return (rows, cols, data)

    
    def tosparse(self, format='coo'):

        # http://numba.pydata.org/numba-doc/latest/reference/deprecation.html#deprecation-of-reflection-for-list-and-set-types
        nb_tiles = numba.typed.List()
        tiles = [t.tocoo() for t in self._tiles]  
        [nb_tiles.append( (t.row, t.col, t.data) ) for t in tiles]

        nb_blocks = numba.typed.List()
        [nb_blocks.append(b) for b in self._blocks]
        
        (rows, cols, data) = self._tosparse(nb_tiles, nb_blocks)

        if format == 'csr':
            T = scipy.sparse.csr_matrix( (data, (rows, cols)), shape=self.shape)
        elif format == 'coo':
            T = scipy.sparse.coo_matrix( (data, (rows, cols)), shape=self.shape)
        elif format == 'csc':
            T = scipy.sparse.csc_matrix( (data, (rows, cols)), shape=self.shape)
        else:
            raise ValueError('Invalid format "%s" - must be ["coo", "csr", "csc"]' % format)
        return T            

