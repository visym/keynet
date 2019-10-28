from torch import nn
from numpy.linalg import multi_dot 
from keynet.util import torch_affine_augmentation_matrix, sparse_toeplitz_conv2d, sparse_toeplitz_avgpool2d
import torch
import numpy as np


class KeyedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(KeyedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.What = None

    def key(self, w, b, A, Ainv, inshape):
        assert(w.shape[0] == self.out_channels and
               w.shape[1] == self.in_channels and
               w.shape[2] == self.kernel_size and
               b.shape[0] == self.out_channels)
        self.What = sparse_toeplitz_conv2d(inshape, w, bias=b, stride=self.stride)
        self.What = A.dot(self.What.dot(Ainv))

    def forward(self, x_affine):
        return torch.t(torch.tensor(self.What.dot(torch.t(x_affine))))


class KeyedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(KeyedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.What = None

    def key(self, W, b, A, Ainv):
        assert(W.shape[0] == self.out_features and W.shape[1] == self.in_features)
        self.What = torch_affine_augmentation_matrix(torch.tensor(W), torch.tensor(b))
        self.What = Ainv.dot(np.array(self.What).transpose()).transpose()
        if A is not None:
            self.What = A.dot(self.What)
        return self.What
        
    def forward(self, x_affine):
        return torch.t(torch.tensor(self.What.dot(torch.t(x_affine))))


class KeyedAvgpool2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(KeyedAvgpool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def key(self, A, Ainv, inshape):
        self.What = sparse_toeplitz_avgpool2d(inshape, (inshape[0],inshape[0],self.kernel_size,self.kernel_size), self.stride)        
        self.What = A.dot(self.What.dot(Ainv))

    def forward(self, x_affine):
        return torch.t(torch.tensor(self.What.dot(torch.t(x_affine))))


