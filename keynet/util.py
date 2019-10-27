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


def torch_conv2d_in_scipy(x,f,b):
    """Torch equivalent conv2d operation in scipy, with input tensor x, filter weight f and bias b"""
    """x=[BATCH,INCHANNEL,HEIGHT,WIDTH], f=[OUTCHANNEL,INCHANNEL,HEIGHT,WIDTH], b=[OUTCHANNEL,1]"""

    assert(len(x.shape) == 4 and len(f.shape) == 4)
    assert(f.shape[1] == x.shape[1])  # equal inchannels
    assert(f.shape[2]==f.shape[3] and f.shape[1]%2 == 1)  # filter is square, odd
    assert(b.shape[0] == f.shape[0])  # weights and bias dimensionality match

    (N,C,U,V) = (x.shape)
    (M,K,P,Q) = (f.shape)
    x_spatialpad = np.pad(x, ( (0,0), (0,0), ((P-1)//2, (P-1)//2), ((Q-1)//2, (Q-1)//2)), mode='constant', constant_values=0)
    y = np.array([scipy.signal.correlate(x_spatialpad[n,:,:,:], f[m,:,:,:], mode='valid') + b[m] for n in range(0,N) for m in range(0,M)])
    return np.reshape(y, (N,M,U,V) )


def sparse_toeplitz_conv2d(inshape, f, b, as_correlation=True):
    # Returns sparse toeplitz matrix (W) that is equivalent to per-channel pytorch conv2d (spatial correlation)
    # see also: test_keynet.test_sparse_toeplitz_conv2d()

    # Valid shapes
    assert(len(inshape) == 4 and len(f.shape) == 4)  # 4D tensor inshape=(batch, height, width, inchannels), f.shape=(outchannels, kernelheight, kernelwidth, inchannels)
    assert(f.shape[1] == inshape[1])  # equal inchannels
    assert(f.shape[2]==f.shape[3] and f.shape[2]%2 == 1)  # filter is square, odd
    assert(len(b.shape) == 1 and b.shape[0] == f.shape[0])  # filter and bias have composable shapes

    # Correlation vs. convolution?
    (N,C,U,V) = inshape
    (M,K,P,Q) = f.shape
    C_range = range(0,C)
    M_range = range(0,M)
    P_range = range(-(P-1)//2, ((P-1)//2)+1)
    Q_range = range(-(Q-1)//2, ((Q-1)//2)+1)
    (data, row_ind, col_ind) = ([],[],[])

    # For every batch element
    for n in range(0,N):
        # For every image_row
        for u in range(0,U):
            # For every image_column
            for v in range(0,V):
                # For every inchannel (transposed)
                for (k_inchannel, c_inchannel) in enumerate(C_range if as_correlation else reversed(C_range)):
                    # For every kernel_row (transposed)
                    for (i,p) in enumerate(P_range if as_correlation else reversed(P_range)):
                        # For every kernel_col (transposed)
                        for (j,q) in enumerate(Q_range if as_correlation else reversed(Q_range)):
                            # For every outchannel
                            for (k_outchannel, c_outchannel) in enumerate(M_range if as_correlation else reversed(M_range)):
                                if ((u+p)>=0 and (v+q)>=0 and (u+p)<U and (v+q)<V):
                                    data.append(f[k_outchannel,k_inchannel,i,j])
                                    row_ind.append( np.ravel_multi_index( (n,c_outchannel,u,v), (N,M,U,V) ) )
                                    col_ind.append( np.ravel_multi_index( (n,c_inchannel, u+p, v+q), (N,C,U,V) ))

    # Sparse matrix (with bias using affine augmentation)
    T = csr_matrix((data, (row_ind, col_ind)), shape=(N*M*U*V, N*C*U*V))
    lastcol = csr_matrix(np.array([x*np.ones( (U*V) ) for n in range(0,N) for x in b]).reshape( (N*M*U*V,1) ))
    T_bias = scipy.sparse.hstack( (T,lastcol) )
    return T_bias


def sparse_toeplitz_avgpool2d(inshape, filtershape, stride):
    (outchannel, inchannel, filtersize, filtersize) = filtershape
    F = (1.0 / (kernelsize*kernelsize))*np.ones( (outchannel, inchannel, kernelsize, kernelsize) )
    #W = sparse_toeplitz_conv2d(inshape, F)
    #W = W[::stride]
    #return W
