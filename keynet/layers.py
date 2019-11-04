from torch import nn
from numpy.linalg import multi_dot 
from keynet.torch import affine_augmentation_matrix, sparse_toeplitz_conv2d, sparse_toeplitz_avgpool2d, scipy_coo_to_torch_sparse
import torch
import numpy as np
import scipy.sparse
import torch.nn.functional as F
import torch.sparse

class KeyedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_torch_sparse=False):
        super(KeyedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.What = None
        self.use_torch_sparse = use_torch_sparse

    def key(self, w, b, A, Ainv, inshape):
        assert(w.shape[0] == self.out_channels and
               w.shape[1] == self.in_channels and
               w.shape[2] == self.kernel_size and
               b.shape[0] == self.out_channels)
        self.What = sparse_toeplitz_conv2d(inshape, w, bias=b, stride=self.stride).tocsr()
        self.What = A.dot(self.What.dot(Ainv))
        if self.use_torch_sparse:
            self.What = scipy_coo_to_torch_sparse(self.What.tocoo())

    def forward(self, x_affine):
        """x_affine=(C*U*V+1 x N)"""
        if self.use_torch_sparse:
            return torch.sparse.mm(self.What, x_affine)
        else:
            return torch.as_tensor(self.What.dot(x_affine.numpy()))



class KeyedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(KeyedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.What = None

    def key(self, W, b, A, Ainv):
        assert(W.shape[0] == self.out_features and W.shape[1] == self.in_features)
        self.What = affine_augmentation_matrix(torch.tensor(W), torch.tensor(b))
        self.What = np.dot(self.What, Ainv.todense())
        if A is not None:
            self.What = A.dot(self.What)
        self.What = torch.as_tensor(self.What)
        
    def forward(self, x_affine):
        """x_affine=(C*U*V+1 x N)"""
        return torch.mm(self.What, x_affine)


class KeyedRelu(nn.Module):
    def __init__(self, use_torch_sparse=False):
        super(KeyedRelu, self).__init__()
        self.What = None
        self.use_torch_sparse = use_torch_sparse

    def key(self, B, Ainv):
        self.What = B*Ainv
        if self.use_torch_sparse:
            self.What = scipy_coo_to_torch_sparse(self.What.tocoo())
        
    def forward(self, x_affine):
        if self.use_torch_sparse:
            return F.relu(torch.sparse.mm(self.What, x_affine))
        else:
            return F.relu(torch.as_tensor(self.What.dot(x_affine.numpy())))


class KeyedAvgpool2d(nn.Module):
    def __init__(self, kernel_size, stride, use_torch_sparse=False):
        super(KeyedAvgpool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_torch_sparse = use_torch_sparse

    def key(self, A, Ainv, inshape):
        self.What = sparse_toeplitz_avgpool2d(inshape, (inshape[0],inshape[0],self.kernel_size,self.kernel_size), self.stride).tocsr()
        self.What = A.dot(self.What.dot(Ainv))
        if self.use_torch_sparse:
            self.What = scipy_coo_to_torch_sparse(self.What.tocoo())

    def forward(self, x_affine):
        """x_affine=(C*U*V+1 x N)"""
        """.as_tensor() shares memory, avoids copy"""
        if self.use_torch_sparse:
            return torch.sparse.mm(self.What, x_affine)
        else:
            return torch.as_tensor(self.What.dot(x_affine.numpy()))


