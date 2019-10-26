from torch import nn
from numpy.linalg import multi_dot 
import keynet.util

class KeyedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(KeyedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.What = None

    def key(self, W, A, x):
        imshape = x.shape[0]
        self.What = keynet.util.sparse_toeplitz_conv2d(imshape, W, as_correlation=True)

    def forward(self, x):
        return self.What.dot(x)


class KeyedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(KeyedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.What = None

    def key(self, W, A, x):
        self.What = multi_dot( (A, W, np.linalg.inv(A)) )
        return self.What
        
    def forward(self, x):
        return self.What.dot(x)


class KeyedAvgpool2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(KeyedAvgpool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def key(self, A, x):
        imshape = x.shape[0]
        self.What = keynet.util.sparse_toeplitz_avgpool2d(imshape, self.kernel_size, self.stride)        

    def forward(self, x):
        return self.What.dot(x)
