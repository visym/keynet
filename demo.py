from keynet.layers import PermutationKeynet, PermutationKeyedSensor
from keynet.mnist import LeNet_AvgPool

def challenge_problem():

    # Create public releasable image and keynet 
    pass


def mnist():
    net = LeNet_AvgPool()
    net.load_state_dict(torch.load('./models/mnist_lenet_avgpool.pth'));
    inshape = (1,28,28)
    
    sensor = PermutationKeyedSensor(inshape)
    keynet = PermutationKeynet(net, inshape, inkey=sensor.key(), do_encrypted_output=False)
    
    img = torch.randn(1, *inshape)
    img_cipher = sensor.encrypt(x).tensor()
    yh = keynet.forward(img_cipher).detach().numpy().flatten()
    y = net.forward(img).detach().numpy().flatten()
    
    assert np.allclose(y, yh, atol=1E-5)

    
if __name__ == '__main__':
    mnist()
