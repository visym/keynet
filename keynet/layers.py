from torch import nn
from numpy.linalg import multi_dot 
from keynet.torch import affine_augmentation_matrix, sparse_toeplitz_conv2d, sparse_toeplitz_avgpool2d
import torch
import numpy as np
import scipy.sparse
import torch.nn.functional as F


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
        """x_affine=(C*U*V+1 x N)"""
        return torch.tensor(self.What.dot(x_affine))


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
        return self.What
        
    def forward(self, x_affine):
        """x_affine=(C*U*V+1 x N)"""
        return torch.tensor(self.What.dot(x_affine))


class KeyedRelu(nn.Module):
    def __init__(self):
        super(KeyedRelu, self).__init__()
        self.What = None

    def key(self, B, Ainv):
        self.What = B*Ainv
        return self.What
        
    def forward(self, x_affine):
        return F.relu(torch.tensor(self.What.dot(x_affine)))


class KeyedAvgpool2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(KeyedAvgpool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def key(self, A, Ainv, inshape):
        self.What = sparse_toeplitz_avgpool2d(inshape, (inshape[0],inshape[0],self.kernel_size,self.kernel_size), self.stride)        
        self.What = A.dot(self.What.dot(Ainv))

    def forward(self, x_affine):
        """x_affine=(C*U*V+1 x N)"""
        return torch.tensor(self.What.dot(x_affine))


def fuse_conv2d_and_bn(conv2d_weight, conv2d_bias, bn_running_mean, bn_running_var, bn_eps, bn_weight, bn_bias):
    """https://discuss.pytorch.org/t/how-to-absorb-batch-norm-layer-weights-into-convolution-layer-weights/16412/4"""
    w = conv2d_weight
    mean = bn_running_mean
    var_sqrt = torch.sqrt(bn_running_var + np.float32(bn_eps))
    beta = bn_weight
    gamma = bn_bias
    out_channels = conv2d_weight.shape[0]
    if conv2d_bias is not None:
        b = conv2d_bias
    else:
        b = mean.new_zeros(mean.shape)
    w = w * (beta / var_sqrt).reshape([out_channels, 1, 1, 1])
    b = (((b - mean)/var_sqrt)*beta) + gamma
    return (w,b)

