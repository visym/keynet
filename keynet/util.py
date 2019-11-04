import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import torch
import PIL
import uuid
import tempfile
import os



def sparse_block_diag(mats, format='coo', dtype=None):
    """https://github.com/scipy/scipy/issues/9223"""
    """Faster than scipy.sparse.block_diag"""
    dtype = np.dtype(dtype)
    row = []
    col = []
    data = []
    r_idx = 0
    c_idx = 0
    for ia, a in enumerate(mats):
        if scipy.sparse.issparse(a):
            a = a.tocsr()
        else:
            a = scipy.sparse.coo_matrix(a).tocsr()
        nrows, ncols = a.shape
        for r in range(nrows):
            for c in range(ncols):
                if a[r, c] is not None:
                    row.append(r + r_idx)
                    col.append(c + c_idx)
                    data.append(a[r, c])
        r_idx = r_idx + nrows
        c_idx = c_idx + ncols
    return scipy.sparse.coo_matrix((data, (row, col)), dtype=dtype).asformat(format)


def random_dense_positive_definite_matrix(n, dtype=np.float32):
    A = np.random.rand(n,n).astype(dtype)
    U, s, V = np.linalg.svd(np.dot(A.T, A))
    X = np.dot(np.dot(U, 1.0 + np.diag(np.random.rand(n).astype(dtype))), V)
    return X

def random_dense_permutation_matrix(n):
    A = np.eye(n)
    A = np.random.permutation(A)
    return A

def random_dense_doubly_stochastic_matrix(n,k,n_iter=100):
    A = np.random.rand()*random_dense_permutation_matrix(n)
    for k in range(0,k):
        A = A + np.random.rand()*random_dense_permutation_matrix(n)
    for k in range(0,n_iter):
        A = A / np.sum(A,axis=0)
        A = A / np.sum(A,axis=1)        
    return A


def gaussian_random_dense_diagonal_matrix(n,sigma=1):
    d = sigma*np.random.randn(n)
    return (np.diag(d))

def uniform_random_dense_diagonal_matrix(n,scale=1,eps=1E-6):
    d = scale*np.random.rand(n) + eps
    return (np.diag(d))

def checkerboard_256x256():
    """Random color 8x8 checkerboard at 256x256 resolution"""
    img = np.uint8(255*np.random.rand(8,8,3))
    img = np.array(PIL.Image.fromarray(img).resize( (256,256), PIL.Image.NEAREST))
    return img

def imshow(img):
    """Use system viewer for uint8 image saved as temporary png"""
    f = os.path.join(tempfile.getempdir(),'%s.png' % uuid.uuid1().hex)
    im = PIL.Image.fromarray(img.astype(np.uint8)).save(f)
    os.system('open %s' % f)

def savetemp(img):
    f = '/tmp/%s.png' % uuid.uuid1().hex
    PIL.Image.fromarray(img.astype(np.uint8)).save(f)
    return f

def sparse_permutation_matrix(n, dtype=np.float32):
    data = np.ones(n).astype(dtype)
    row_ind = list(range(0,n))
    col_ind = np.random.permutation(list(range(0,n)))
    return csr_matrix((data, (row_ind, col_ind)), shape=(n,n))

def sparse_identity_matrix(n, dtype=np.float32):
    return scipy.sparse.eye(n, dtype=dtype)

def sparse_gaussian_random_diagonal_matrix(n,mu=1,sigma=1,eps=1E-6):
    return scipy.sparse.diags(np.maximum(eps, np.array(sigma*np.random.randn(n)+mu).astype(np.float32)))

def sparse_uniform_random_diagonal_matrix(n, scale=1, eps=1E-6, dtype=np.float32):
    return scipy.sparse.diags(np.array(scale*np.random.rand(n) + eps).astype(dtype))

def sparse_diagonal_matrix(n):
    return sparse_uniform_random_diagonal_matrix(n, scale=1, eps=1E-6)

def sparse_inverse_diagonal_matrix(D):
    return scipy.sparse.diags(1.0 / D.diagonal())

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
    

def sparse_stochastic_matrix(n,m):
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


def sparse_positive_definite_block_diagonal(n, m, dtype=np.float32):
    m = np.minimum(n,m)
    B = [random_dense_positive_definite_matrix(m,dtype) for j in np.arange(0,n-m,m)]
    B = B + [random_dense_positive_definite_matrix(n-len(B)*m, dtype)]
    A = sparse_block_diag(B, format='csr')
    Binv = [np.linalg.inv(b) for b in B]
    Ainv = sparse_block_diag(Binv, format='csr')
    return(A,Ainv)


def sparse_generalized_stochastic_block_matrix(n,m,dtype=np.float32):
    """Returns (A,Ainv) for (nxn) sparse matrix of the form P*S*D, where D is uniform random diagonal, S is stochastic block matrix of size m, and P is permutation"""
    """Setting m=1 results in scaled permutation matrix"""
    #assert(k<=m and n>=m) 
    m = np.minimum(n,m)

    (M,Minv) = sparse_stochastic_matrix(n,m)
    D = sparse_uniform_random_diagonal_matrix(n)
    A = M*D
    Dinv = sparse_inverse_diagonal_matrix(D)
    Ainv = Dinv * Minv

    return(A.astype(np.float32), Ainv.astype(np.float32))

def sparse_generalized_permutation_block_matrix(n, m, dtype=np.float32):
    """Returns (A,Ainv) for (nxn) sparse matrix of the form B*P*D, where D is uniform random diagonal, B is block diagonal of size m, and P is permutation"""
    """Setting m=k=1 results in scaled permutation matrix"""
    m = np.minimum(n,m)

    (B,Binv) = sparse_positive_definite_block_diagonal(n,m,dtype=np.float64)
    D = sparse_uniform_random_diagonal_matrix(n,dtype=np.float64)
    P = sparse_permutation_matrix(n,dtype=np.float64)
    A = B*P*D
    Dinv = sparse_inverse_diagonal_matrix(D)
    Pinv = P.transpose()
    Ainv = Dinv * Pinv * Binv

    return(A.astype(dtype), Ainv.astype(dtype))




    
