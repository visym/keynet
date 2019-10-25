from torch import nn
from numpy.linalg import multi_dot 


class KeyedConv2d(nn.Module):
    def __init__(self):
        super(KeyedConv2d, self).__init__()
        self.W = None

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return None


class KeyedLinear(nn.Module):
    def __init__(self):
        super(KeyedLinear, self).__init__()
        self.W = None

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return multi_dot( (self.W, x) )
