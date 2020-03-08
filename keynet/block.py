from numpy.linalg import multi_dot 
import numpy as np
import math
from numpy.lib.stride_tricks import as_strided
import scipy.sparse
from vipy.util import groupbyasdict
import torch


def matrix_blockview(W, inshape, n):
    """Reorder a sparse matrix W such that:  W*x.flatten() == matrix_blockview(W, x.shape, n)*blockview(x,n).flatten()"""
    d = {v:k for (k,v) in enumerate(blockview(np.array(range(0,np.prod(inshape))).reshape(inshape), n).flatten())}
    data = [v for (k,v) in zip(W.row, W.data) if k in d]    
    row = [d[k] for k in W.row if k in d]
    col = [d[k] for k in W.col if k in d]
    return scipy.sparse.coo_matrix( (data, (row, col)) )


def blockview(A, n):
    """View an np.array A such that block A[0:n, 0:n, :] is continguous, followed by block A[0:n, n:2*n, :], rather than row continuous"""
    shape = (A.shape[0]//n, A.shape[1]//n) + (n,n)
    strides = (n*A.strides[0], n*A.strides[1])+ A.strides
    return as_strided(A, shape=shape, strides=strides)

                   
class TiledMatrix(object):
    def __init__(self, T, n, blockmode='dense'):
        """Given a sparse matrix T, split into non-overlapping nxn blocks or 'tiles', and return an indexed representation for unique submatrices which provides memory efficient matrix vector multiplication when T is self-similar
        
        Outputs:
        B = [(i,j,k),...] for block index (i,j) with submatrix key k
        M = {k:np.array(), ...} a submatrix dictionary, such that the submatrix for block (i,j)=(B[u][0], B[u][1]) is M[B[u][2]]
        
        """
        assert T.shape[0] == T.shape[1]
        
        T = T.tocoo()
        (B, M) = ([], {})
        d_blockidx_to_entries = groupbyasdict( [(i,j,v, (i//n, j//n)) for (i,j,v) in zip(T.row, T.col, T.data)], lambda x: x[3])
        for i in range(0, T.shape[0], n):
            for j in range(0, T.shape[1], n):
                if (i//n,j//n) in d_blockidx_to_entries:
                    (rows, cols, vals) = zip(*[(ii-i,jj-j,v) for (ii,jj,v,ul) in d_blockidx_to_entries[(i//n,j//n)]])
                    n_clip = min(n, T.shape[0]-i)
                    m = scipy.sparse.coo_matrix( (vals, (rows,cols)), shape=(n_clip, n_clip)).todense().astype(np.float32)  # submatrix
                    k = hash(m.tostring())
                    if k not in M:
                        if blockmode == 'dense':
                            M[k] = torch.as_tensor(m)
                        elif blockmode == 'csr':
                            M[k] = m.tocsr()
                        else:
                            raise ValueError('invalid blockmode "%s"' % blockmode)
                else:
                    k = None
                B.append( (i//n, j//n, k) )
        self._B = list(set(B))
        self._M = M
        self._blocksize = n
        self._shape = T.shape
        self._numblocks = T.shape[0] // n

    def blocksize(self):
        return self._blocksize

    def leftdot(self, x):
        """Input is NxCxHxW tensor viewed as Nx(C*H*W) tensor, compute left matrix multiplication (x * T) return (NxC*H*W)"""
        if isinstance(x, TiledMatrix):
            return copy.deepcopy(self).prod(x)
        n = self._blocksize
        y = torch.zeros_like(x)
        for (i,j,k) in self._B:
            if k is not None:
                y[:, j*n:j*n+n] += torch.matmul(x[:, i*n:i*n+n], self._M[k])
        return y
    
    def prod(self, other):
        """For two Tiled() object T1, T2, compute T1.dot(T2) and save in T1"""
        assert isinstance(other, TiledMatrix)
        assert other._blocksize == self._blocksize
        
        M_accum = {}
        for (i, jj, v) in self._B:
            for (ii, j, vo) in other._B:
                if jj == ii and v is not None and vo is not None:
                    m = torch.matmul(self._M[v], other._M[vo])
                    if (i,j) not in M_accum:
                        M_accum[(i,j)] = m
                    else:
                        M_accum[(i,j)] += m

        (B, M) = ([], {})
        for ((i,j), m) in M_accum.items():
            k = hash(m)
            if k not in M:
                M[k] = m
            B.append( (i,j,k) )                        
        
        self._B = B
        self._M = M
        return self
    
    def tocoo(self):
        """Convert to COOrdinate sparse matrix, this is an expensive operation that should be used for small matrices only"""
        ((H,W), N) = (self._shape, self._blocksize)
        d = {(i*N, j*N):k for (i,j,k) in self._B}
        return scipy.sparse.bmat([ [scipy.sparse.coo_matrix(self._M[d[(i,j)]].numpy()) if ((i,j) in d and d[(i,j)] is not None) else None for j in range(0,W,N)]
                                   for i in range(0,H,N)])
