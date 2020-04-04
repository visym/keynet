import numpy as np


def affine_to_linear_columnvector(x):
    assert isinstance(x, np.ndarray)
    if x.ndim == 1:
        return np.concatenate( (x, np.ones((1,1), dtype=x.dtype)))
    elif x.ndim == 2:
        return np.vstack( (x, np.ones((1,x.shape[1]), dtype=x.dtype)))
    else:
        raise ValueError('invalid input - must be 1D or 2D np.array()')

    
def affine_to_linear_rowvector(x):
    assert isinstance(x, np.ndarray)
    if x.ndim == 1:
        return np.concatenate( (x, np.ones((1,1), dtype=x.dtype)))
    elif x.ndim == 2:
        return np.hstack( (x, np.ones((x.shape[0],1), dtype=x.dtype)))        
    else:
        raise ValueError('invalid input - must be 1D or 2D np.array()')

    
def linear_to_affine_rowvector(x):
    assert isinstance(x, np.ndarray)
    if x.ndim == 1:
        assert x[-1] == 1
        return x[0:-1]
    elif x.ndim == 2:
        assert np.all(x[:,-1] == 1)
        return x[:,0:-1]
    else:
        raise ValueError('invalid input - must be 1D or 2D np.array() and last column is one')        

    
def linear_to_affine_columnvector(x):
    assert isinstance(x, np.ndarray)
    if x.ndim == 1:
        assert x[-1] == 1
        return x[0:-1]
    elif x.ndim == 2:
        assert np.all(x[-1,:] == 1)
        return x[0:-1,:]
    else:
        raise ValueError('invalid input - must be 1D or 2D np.array() and last row is one')        
        
        
def random_doubly_stochastic_matrix(n,k,n_iter=100):
    A = np.random.rand()*random_permutation_matrix(n)
    for k in range(0,k):
        A = A + np.random.rand()*random_permutation_matrix(n)
    for k in range(0,n_iter):
        A = A / np.sum(A,axis=0)
        A = A / np.sum(A,axis=1)        
    return A

def random_permutation_matrix(n):
    A = np.eye(n)
    A = np.random.permutation(A)
    return A

def random_positive_definite_matrix(n, dtype=np.float32):
    A = np.random.rand(n,n).astype(dtype)
    U, s, V = np.linalg.svd(np.dot(A.T, A))
    X = np.dot(np.dot(U, 1.0 + np.diag(np.random.rand(n).astype(dtype))), V)
    return X


def gaussian_random_diagonal_matrix(n,sigma=1):
    d = sigma*np.random.randn(n)
    return (np.diag(d))

def uniform_random_diagonal_matrix(n,scale=1,eps=1E-6):
    d = scale*np.random.rand(n) + eps
    return (np.diag(d))
