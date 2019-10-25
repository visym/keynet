import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

def sparse_permutation_matrix(n):
    data = np.ones(n).astype(np.float32)
    row_ind = list(range(0,n))
    col_ind = np.random.permutation(list(range(0,n)))
    return csr_matrix((data, (row_ind, col_ind)), shape=(n,n))

def sparse_identity_matrix(n):
    return scipy.sparse.eye(n, dtype=np.float32)


def sparse_gaussian_random_diagonal_matrix(n,sigma=1):
    return scipy.sparse.diags(np.array(sigma*np.random.randn(n)).astype(np.float32))

def sparse_uniform_random_diagonal_matrix(n,scale=1,eps=1E-6):
    return scipy.sparse.diags(np.array(scale*np.random.rand(n) + eps).astype(np.float32))


def sparse_random_doubly_stochastic_matrix(n, k):
    A = np.random.rand()*sparse_permutation_matrix(n)
    for k in range(0,k):
        A = A + np.random.rand()*sparse_permutation_matrix(n)
    for k in range(0,100):
        A = normalize(A, norm='l1', axis=0)
        A = normalize(A, norm='l1', axis=1)
    return A


def sparse_toeplitz_conv2d(imshape, f, as_correlation=True):
    # Returns sparse toeplitz matrix (W) that is equivalent to per-channel pytorch conv2d (spatial correlation)
    #
    # y = W.dot(img.flatten()).reshape(img.shape[0], img.shape[1]) 
    # yh = scipy.signal.correlate(np.pad(img, ( ((P-1)//2, (P-1)//2), ((Q-1)//2, (Q-1)//2), (0,0)), mode='constant'), f, mode='valid')
    # np.allclose(y, yh)

    # Valid shapes
    fshape = f.shape
    if len(imshape) == 2:
        imshape = (imshape[0], imshape[1], 1)
    if len(f.shape) == 2:
        fshape = (fshape[0], fshape[1], 1)
        f = np.expand_dims(f, 2)
    assert(len(imshape) == 3 and len(fshape) == 3 and fshape[2] == imshape[2])  # 3D, equal channels
    assert(fshape[0]==fshape[1] and fshape[0]%2 == 1)  # odd, square

    # Correlation vs. convolution?
    (U,V,C) = imshape
    (P,Q,R) = fshape
    C_range = range(0,C)
    P_range = range(-(P-1)//2, ((P-1)//2)+1)
    Q_range = range(-(Q-1)//2, ((Q-1)//2)+1)

    # For every image_row
    (data, row_ind, col_ind) = ([],[],[])
    for u in range(0,U):
        # For every image_column
        for v in range(0,V):
            # For every channel (transposed)
            for (k,c) in enumerate(C_range if as_correlation else reversed(C_range)):
                # For every kernel_row (transposed)
                for (i,p) in enumerate(P_range if as_correlation else reversed(P_range)):
                    # For every kernel_col (transposed)
                    for (j,q) in enumerate(Q_range if as_correlation else reversed(Q_range)):
                        if ((u+p)>=0 and (v+q)>=0 and (u+p)<U and (v+q)<V):
                            data.append(f[i,j,k])
                            row_ind.append( np.ravel_multi_index( (u,v), (U,V) ) )
                            col_ind.append( np.ravel_multi_index( (u+p,v+q,c), (U,V,C) ))

    return csr_matrix((data, (row_ind, col_ind)), shape=(U*V, U*V*C))
    
