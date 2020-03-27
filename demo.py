import vipy
from keynet.system import PermutationKeynet
from keynet.mnist import LeNet_AvgPool
import torch
import numpy as np


def challenge_problem():
    # Create public releasable image and keynet 
    pass


def mnist():
    net = LeNet_AvgPool()
    net.load_state_dict(torch.load('./models/mnist_lenet_avgpool.pth'));
    inshape = (1,28,28)

    (sensor, model) = PermutationKeynet(inshape, net, do_output_encryption=False)
        
    img_plain = sensor.load('owl.jpg').tensor()
    img_cipher = sensor.load('owl.jpg').encrypt().tensor()

    yh = model.forward(img_cipher).detach().numpy().flatten()
    y = net.forward(img_plain).detach().numpy().flatten()
    assert np.allclose(y, yh, atol=1E-5)

    im_cipher = sensor.encrypt().image().resize(512, 512, interp='nearest').show()  # figure 1
    im_plain = sensor.decrypt().image().resize(512, 512, interp='nearest').show()   # figure 2
    
    
if __name__ == '__main__':
    mnist()
