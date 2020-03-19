import numpy as np


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
