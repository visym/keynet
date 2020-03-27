import vipy
from keynet.system import PermutationKeynet
from keynet.mnist import LeNet_AvgPool

def challenge_problem():

    # Create public releasable image and keynet 
    pass


def mnist():
    net = LeNet_AvgPool()
    net.load_state_dict(torch.load('./models/mnist_lenet_avgpool.pth'));
    inshape = (1,28,28)

    (sensor, model) = PermutationKeynet(inshape, net, do_output_encryption=False)
    
    
    img_plain = sensor.load('mnist_zero.jpg').tensor()
    img_cipher = sensor.load('mnist_zero.jpg').encrypt().tensor()

    yh = keynet.forward(img_cipher).detach().numpy().flatten()
    y = net.forward(img_plain).detach().numpy().flatten()
    
    assert np.allclose(y, yh, atol=1E-5)

    im_cipher = sensor.encrypt().image().show()
    im_plain = sensor.decrypt().image().show()
    
    
if __name__ == '__main__':
    mnist()
