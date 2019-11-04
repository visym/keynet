import numpy as np
import torch
import torchvision
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import keynet.layers
from keynet.util import sparse_permutation_matrix, sparse_identity_matrix, sparse_uniform_random_diagonal_matrix, sparse_inverse_diagonal_matrix
from keynet.util import sparse_generalized_stochastic_block_matrix, sparse_generalized_permutation_block_matrix, sparse_stochastic_matrix
from keynet.torch import affine_augmentation_tensor, affine_deaugmentation_tensor
import vipy.util
import keynet.cifar10


class LeNet(nn.Module):
    """Slightly modified LeNet to include padding, odd filter sizes and even image sizes"""
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1, stride=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.pool1 = F.max_pool2d
        self.pool2 = F.max_pool2d

    def forward(self, x):
        x = F.relu(self.conv1(x))                                   # (1,28,28) -> (6,28,28)
        x = self.pool1(x, kernel_size=3, stride=2, padding=1)       # (6,28,28) -> (6,14,14)
        x = F.relu(self.conv2(x))                                   # (6,14,14) -> (16,14,14)  
        x = self.pool2(x, kernel_size=3, stride=2, padding=1)       # (16,14,14) -> (16,7,7)
        x = x.view(-1, 7*7*16)                                      # (16,7,7) -> (16*7*7,1)
        x = F.relu(self.fc1(x))                                     # (16*7*7,1) -> (120,1)
        x = F.relu(self.fc2(x))                                     # (120,1) -> (84,1)
        x = self.fc3(x)                                             # (84,1) -> (10,1)
        return x

    def loss(self, x):
        return F.log_softmax(x, dim=1)

    def transform(self):
        return transforms.Compose([transforms.ToTensor(),        
                                   transforms.Normalize((0.1307,), (0.3081,))])

    def transform_cifar10_train(self):
        return transforms.Compose([transforms.Grayscale(),
                                   transforms.Resize( (28,28) ),
                                   transforms.RandomCrop(28, padding=4),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.49139968,), (0.24703223,))])

    def transform_cifar10_test(self):
        return transforms.Compose([transforms.Grayscale(),
                                   transforms.Resize( (28,28) ),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.49139968,), (0.24703223,))])

class LeNet_AvgPool(LeNet):
    """https://github.com/pytorch/examples/blob/master/mnist/main.py"""
    def __init__(self):
        super(LeNet_AvgPool, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)  
        self.pool1 = F.avg_pool2d   
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)  
        self.pool2 = F.avg_pool2d  
        self.fc1 = nn.Linear(7*7*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


class KeyNet(nn.Module):
    def __init__(self, keys=None):
        super(KeyNet, self).__init__()
        self.conv1 = keynet.layers.KeyedConv2d(1, 6, kernel_size=3, stride=1)  # assumed padding=1
        self.pool1 = keynet.layers.KeyedAvgpool2d(kernel_size=3, stride=2)
        self.conv2 = keynet.layers.KeyedConv2d(6, 16, kernel_size=3, stride=1)  # assumed padding=1
        self.pool2 = keynet.layers.KeyedAvgpool2d(3,2)
        self.fc1 = keynet.layers.KeyedLinear(7*7*16, 120)
        self.fc2 = keynet.layers.KeyedLinear(120, 84)
        self.fc3 = keynet.layers.KeyedLinear(84, 10)
        self.relu1 = keynet.layers.KeyedRelu()
        self.relu2 = keynet.layers.KeyedRelu()
        self.relu3 = keynet.layers.KeyedRelu()
        self.relu4 = keynet.layers.KeyedRelu()

        # Layer output shapes:  x1 = conv1(x0)
        self.shape = {'x0':(1,28,28),   # image input
                      'x1a':(6,28,28),   # conv1 output
                      'x1b':(6,28,28),   # relu output
                      'x2':(6,14,14),    # pool1 output
                      'x3a':(16,14,14),  # conv2 output
                      'x3b':(16,14,14),  # conv2-relu output
                      'x4':(16,7,7),     # pool2 output
                      'x5a':(120,1,1),   # fc1 output
                      'x5b':(120,1,1),   # fc1-relu
                      'x6a':(84,1,1),    # fc2 output
                      'x6b':(84,1,1),    # fc2-relu output
                      'x7':(10,1,1)}     # fc3 output

        self.keys = {'A0inv':None,
                     'A1a':sparse_identity_matrix(np.prod(self.shape['x1a'])+1),
                     'A1ainv':sparse_identity_matrix(np.prod(self.shape['x1a'])+1),
                     'A1b':sparse_identity_matrix(np.prod(self.shape['x1b'])+1),
                     'A1binv':sparse_identity_matrix(np.prod(self.shape['x1b'])+1),
                     'A2':sparse_identity_matrix(np.prod(self.shape['x2'])+1),
                     'A2inv':sparse_identity_matrix(np.prod(self.shape['x2'])+1),                     
                     'A3a':sparse_identity_matrix(np.prod(self.shape['x3a'])+1),
                     'A3ainv':sparse_identity_matrix(np.prod(self.shape['x3a'])+1),
                     'A3b':sparse_identity_matrix(np.prod(self.shape['x3b'])+1),
                     'A3binv':sparse_identity_matrix(np.prod(self.shape['x3b'])+1),
                     'A4':sparse_identity_matrix(np.prod(self.shape['x4'])+1),
                     'A4inv':sparse_identity_matrix(np.prod(self.shape['x4'])+1),
                     'A5a':sparse_identity_matrix(np.prod(self.shape['x5a'])+1),
                     'A5ainv':sparse_identity_matrix(np.prod(self.shape['x5a'])+1),
                     'A5b':sparse_identity_matrix(np.prod(self.shape['x5b'])+1),
                     'A5binv':sparse_identity_matrix(np.prod(self.shape['x5b'])+1),
                     'A6a':sparse_identity_matrix(np.prod(self.shape['x6a'])+1),
                     'A6ainv':sparse_identity_matrix(np.prod(self.shape['x6a'])+1),
                     'A6b':sparse_identity_matrix(np.prod(self.shape['x6b'])+1),
                     'A6binv':sparse_identity_matrix(np.prod(self.shape['x6b'])+1),
                     'A7':sparse_identity_matrix(np.prod(self.shape['x7'])+1),
                     'A7inv':sparse_identity_matrix(np.prod(self.shape['x7'])+1)} if keys is None else keys
        
    def load_state_dict_keyed(self, d_state, A0inv):        
        d_state = {k.replace('module.',''):v.cpu() for (k,v) in d_state.items()}  # nn.DataParallel training cruft (if needed)

        self.conv1.key(np.array(d_state['conv1.weight']), np.array(d_state['conv1.bias']), self.keys['A1a'], A0inv, self.shape['x0'])
        self.relu1.key(self.keys['A1b'], self.keys['A1ainv'])
        self.pool1.key(self.keys['A2'], self.keys['A1binv'], self.shape['x1b'])
        self.conv2.key(np.array(d_state['conv2.weight']), np.array(d_state['conv2.bias']), self.keys['A3a'], self.keys['A2inv'], self.shape['x2'])
        self.relu2.key(self.keys['A3b'], self.keys['A3ainv'])
        self.pool2.key(self.keys['A4'], self.keys['A3binv'], self.shape['x3b'])
        self.fc1.key(np.array(d_state['fc1.weight']), np.array(d_state['fc1.bias']), self.keys['A5a'], self.keys['A4inv'])
        self.relu3.key(self.keys['A5b'], self.keys['A5ainv'])
        self.fc2.key(np.array(d_state['fc2.weight']), np.array(d_state['fc2.bias']), self.keys['A6a'], self.keys['A5binv'])
        self.relu4.key(self.keys['A6b'], self.keys['A6ainv'])
        self.fc3.key(np.array(d_state['fc3.weight']), np.array(d_state['fc3.bias']), self.keys['A7'], self.keys['A6binv'])

    def forward(self, A0x0_affine):
        x1a = self.conv1(A0x0_affine)
        x1b = self.relu1(x1a)
        x2 = self.pool1(x1b)
        x3a = self.conv2(x2)
        x3b = self.relu2(x3a)
        x4 = self.pool2(x3b)
        x5a = self.fc1(x4.view(7*7*16+1, -1))  # reshape transposed
        x5b = self.relu3(x5a)  
        x6a = self.fc2(x5b)
        x6b = self.relu4(x6a)
        x7 = self.fc3(x6b)
        return x7

    def encrypt(self, A0, x):        
        return torch.tensor(A0.dot(affine_augmentation_tensor(x))) if A0 is not None else x

    def decrypt(self, x):
        if self.keys['A7inv'] is not None and self.keys['A7'] is not None:
            return affine_deaugmentation_tensor(torch.tensor(self.keys['A7inv'].dot(x)))
        else:
            return affine_deaugmentation_tensor(x)

    def loss(self, x):
        return F.log_softmax(self.decrypt(x), dim=1)


    def transform(self):
        return transforms.Compose([transforms.ToTensor(),                                    
                                   transforms.Normalize((0.1307,), (0.3081,))])


class PermutationKeyNet(KeyNet):
    def __init__(self):
        super(PermutationKeyNet, self).__init__()

        keys = {'A1a':sparse_permutation_matrix(np.prod(self.shape['x1a'])+1),
                'A1b':sparse_permutation_matrix(np.prod(self.shape['x1b'])+1),
                'A2':sparse_permutation_matrix(np.prod(self.shape['x2'])+1),
                'A3a':sparse_permutation_matrix(np.prod(self.shape['x3a'])+1),
                'A3b':sparse_permutation_matrix(np.prod(self.shape['x3b'])+1),
                'A4':sparse_permutation_matrix(np.prod(self.shape['x4'])+1),
                'A5a':sparse_permutation_matrix(np.prod(self.shape['x5a'])+1),
                'A5b':sparse_permutation_matrix(np.prod(self.shape['x5b'])+1),
                'A6a':sparse_permutation_matrix(np.prod(self.shape['x6a'])+1),
                'A6b':sparse_permutation_matrix(np.prod(self.shape['x6b'])+1),
                'A7':sparse_permutation_matrix(np.prod(self.shape['x7'])+1),}
        
        keys.update({'A0inv':None,
                     'A1ainv':keys['A1a'].transpose(),
                     'A1binv':keys['A1b'].transpose(),
                     'A2inv':keys['A2'].transpose(),
                     'A3ainv':keys['A3a'].transpose(),
                     'A3binv':keys['A3b'].transpose(),
                     'A4inv':keys['A4'].transpose(),
                     'A5ainv':keys['A5a'].transpose(),
                     'A5binv':keys['A5b'].transpose(),
                     'A6ainv':keys['A6a'].transpose(),
                     'A6binv':keys['A6b'].transpose(),
                     'A7inv':keys['A7'].transpose()})

        super(PermutationKeyNet, self).__init__(keys)


class DiagonalKeyNet(KeyNet):
    def __init__(self):
        super(DiagonalKeyNet, self).__init__()

        keys = {'A1a':sparse_uniform_random_diagonal_matrix(np.prod(self.shape['x1a'])+1),
                'A1b':sparse_uniform_random_diagonal_matrix(np.prod(self.shape['x1b'])+1),
                'A2':sparse_uniform_random_diagonal_matrix(np.prod(self.shape['x2'])+1),
                'A3a':sparse_uniform_random_diagonal_matrix(np.prod(self.shape['x3a'])+1),
                'A3b':sparse_uniform_random_diagonal_matrix(np.prod(self.shape['x3b'])+1),
                'A4':sparse_uniform_random_diagonal_matrix(np.prod(self.shape['x4'])+1),
                'A5a':sparse_uniform_random_diagonal_matrix(np.prod(self.shape['x5a'])+1),
                'A5b':sparse_uniform_random_diagonal_matrix(np.prod(self.shape['x5b'])+1),
                'A6a':sparse_uniform_random_diagonal_matrix(np.prod(self.shape['x6a'])+1),
                'A6b':sparse_uniform_random_diagonal_matrix(np.prod(self.shape['x6b'])+1),
                'A7':sparse_uniform_random_diagonal_matrix(np.prod(self.shape['x7'])+1),}
        
        keys.update({'A0inv':None,
                     'A1ainv':sparse_inverse_diagonal_matrix(keys['A1a']),
                     'A1binv':sparse_inverse_diagonal_matrix(keys['A1b']),
                     'A2inv':sparse_inverse_diagonal_matrix(keys['A2']),
                     'A3ainv':sparse_inverse_diagonal_matrix(keys['A3a']),
                     'A3binv':sparse_inverse_diagonal_matrix(keys['A3b']),
                     'A4inv':sparse_inverse_diagonal_matrix(keys['A4']),
                     'A5ainv':sparse_inverse_diagonal_matrix(keys['A5a']),
                     'A5binv':sparse_inverse_diagonal_matrix(keys['A5b']),
                     'A6ainv':sparse_inverse_diagonal_matrix(keys['A6a']),
                     'A6binv':sparse_inverse_diagonal_matrix(keys['A6b']),
                     'A7inv':sparse_inverse_diagonal_matrix(keys['A7'])})

        super(DiagonalKeyNet, self).__init__(keys)


class StochasticKeyNet(KeyNet):
    def __init__(self):
        super(StochasticKeyNet, self).__init__()

        (A1a,A1ainv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x1a'])+1, 1) 
        (A1b,A1binv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x1b'])+1, 1) 
        (A2,A2inv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x2'])+1, 1)
        (A3a,A3ainv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x3a'])+1, 1)
        (A3b,A3binv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x3b'])+1, 1)
        (A4,A4inv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x4'])+1, 1)
        (A5a,A5ainv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x5a'])+1, 1)
        (A5b,A5binv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x5b'])+1, 1)
        (A6a,A6ainv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x6a'])+1, 1)
        (A6b,A6binv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x6b'])+1, 1)
        (A7,A7inv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x7'])+1, 1)

        keys = {'A0inv':None,'A1a':A1a,'A1ainv':A1ainv,'A1b':A1b,'A1binv':A1binv,
                'A2':A2,'A2inv':A2inv,'A3a':A3a,'A3ainv':A3ainv,'A3b':A3b,'A3binv':A3binv,
                'A4':A4,'A4inv':A4inv,'A5a':A5a,'A5ainv':A5ainv,'A5b':A5b,'A5binv':A5binv,
                'A6a':A6a,'A6ainv':A6ainv,'A6b':A6b,'A6binv':A6binv,
                'A7':A7,'A7inv':A7inv}

        super(StochasticKeyNet, self).__init__(keys)


class GeneralizedStochasticKeyNet(KeyNet):
    def __init__(self, alpha=1):
        super(GeneralizedStochasticKeyNet, self).__init__()

        (A1a,A1ainv) = sparse_generalized_stochastic_block_matrix(np.prod(self.shape['x1a'])+1, alpha) 
        (A1b,A1binv) = sparse_generalized_stochastic_block_matrix(np.prod(self.shape['x1b'])+1, 1) 
        (A2,A2inv) = sparse_generalized_stochastic_block_matrix(np.prod(self.shape['x2'])+1, alpha)
        (A3a,A3ainv) = sparse_generalized_stochastic_block_matrix(np.prod(self.shape['x3a'])+1, alpha)
        (A3b,A3binv) = sparse_generalized_stochastic_block_matrix(np.prod(self.shape['x3b'])+1, 1)
        (A4,A4inv) = sparse_generalized_stochastic_block_matrix(np.prod(self.shape['x4'])+1, alpha)
        (A5a,A5ainv) = sparse_generalized_stochastic_block_matrix(np.prod(self.shape['x5a'])+1, alpha)
        (A5b,A5binv) = sparse_generalized_stochastic_block_matrix(np.prod(self.shape['x5b'])+1, 1)
        (A6a,A6ainv) = sparse_generalized_stochastic_block_matrix(np.prod(self.shape['x6a'])+1, alpha)
        (A6b,A6binv) = sparse_generalized_stochastic_block_matrix(np.prod(self.shape['x6b'])+1, 1)
        (A7,A7inv) = sparse_generalized_stochastic_block_matrix(np.prod(self.shape['x7'])+1, alpha)

        keys = {'A0inv':None,'A1a':A1a,'A1ainv':A1ainv,'A1b':A1b,'A1binv':A1binv,
                'A2':A2,'A2inv':A2inv,'A3a':A3a,'A3ainv':A3ainv,'A3b':A3b,'A3binv':A3binv,
                'A4':A4,'A4inv':A4inv,'A5a':A5a,'A5ainv':A5ainv,'A5b':A5b,'A5binv':A5binv,
                'A6a':A6a,'A6ainv':A6ainv,'A6b':A6b,'A6binv':A6binv,
                'A7':A7,'A7inv':A7inv}

        super(GeneralizedStochasticKeyNet, self).__init__(keys)


class BlockKeyNet(KeyNet):
    def __init__(self, alpha=1):
        super(BlockKeyNet, self).__init__()

        (A1a,A1ainv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x1a'])+1, alpha) 
        (A1b,A1binv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x1b'])+1, 1) 
        (A2,A2inv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x2'])+1, alpha)
        (A3a,A3ainv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x3a'])+1, alpha)
        (A3b,A3binv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x3b'])+1, 1)
        (A4,A4inv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x4'])+1, alpha)
        (A5a,A5ainv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x5a'])+1, alpha)
        (A5b,A5binv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x5b'])+1, 1)
        (A6a,A6ainv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x6a'])+1, alpha)
        (A6b,A6binv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x6b'])+1, 1)
        (A7,A7inv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x7'])+1, alpha)

        keys = {'A0inv':None,'A1a':A1a,'A1ainv':A1ainv,'A1b':A1b,'A1binv':A1binv,
                'A2':A2,'A2inv':A2inv,'A3a':A3a,'A3ainv':A3ainv,'A3b':A3b,'A3binv':A3binv,
                'A4':A4,'A4inv':A4inv,'A5a':A5a,'A5ainv':A5ainv,'A5b':A5b,'A5binv':A5binv,
                'A6a':A6a,'A6ainv':A6ainv,'A6b':A6b,'A6binv':A6binv,
                'A7':A7,'A7inv':A7inv}

        super(BlockKeyNet, self).__init__(keys)


def train(net, modelfile, mnistdir='/proj/enigma', lr=0.003, epochs=20, transform=LeNet().transform()):
    trainset = datasets.MNIST(mnistdir, download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    criterion = F.nll_loss
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    time0 = time()

    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()            
            output = net.loss(net(images))
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()        
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))

        print("Training Time (in minutes) =",(time()-time0)/60)

    torch.save(net.state_dict(), modelfile)
    return net


def validate(net, mnistdir='/proj/enigma', secretkey=None, transform=LeNet().transform()):
    valset = datasets.MNIST(mnistdir, download=True, train=False, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
    net.eval()

    with vipy.util.Stopwatch() as sw:
        (total, correct) = (0,0)
        for images,labels in valloader:
            for i in range(len(labels)):
                with torch.no_grad():
                    output = net.loss(net(images if secretkey is None else net.encrypt(secretkey,images)))
                _, pred = torch.max(output, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

    print("Mean classification accuracy = %f" % (correct/total))
    print('Validation: %s sec' % sw.elapsed)

def lenet():
    net = train(LeNet(), modelfile='./models/mnist_lenet.pth', lr=0.003, epochs=20)
    validate(net)


def lenet_avgpool():
    net = train(LeNet_AvgPool(), modelfile='./models/mnist_lenet_avgpool.pth', lr=0.003, epochs=40)
    validate(net)

def allconvnet():
    transform = transforms.Compose([transforms.Grayscale(),
                                    transforms.Resize( (32,32) ),
                                    transforms.ToTensor(),        
                                    transforms.Normalize((0.1307,), (0.3081,))])
    net = train(keynet.cifar10.AllConvNet(n_input_channels=1), modelfile='./models/mnist_allconvnet.pth', lr=0.003, epochs=40, transform=transform)
    validate(net, transform=transform)
    


def lenet_avgpool_fiberbundle(do_mean_estimation=True, mnistdir='/proj/enigma'):
    # Mean
    if do_mean_estimation:
        transform = transforms.Compose([transforms.Lambda(lambda img: keynet.fiberbundle.transform(img.convert('RGB'), (28,28))),
                                        transforms.Grayscale(),
                                        transforms.Resize( (28,28) ),
                                        transforms.ToTensor()])
    
        trainset = torchvision.datasets.MNIST(root='/proj/enigma', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=32)
        imglist = []
        for (k, (images,labels)) in enumerate(trainloader):
            imglist.append(images)
            if k > 1024:
                break
        mu = np.mean(np.array(images).flatten())
        std = np.std(np.array(images).flatten())
        print(mu,std)
    else:
        (mu, std) = (0.46616146, 0.06223659)

    # Load full transformed dataset in memory (parallelized)
    transform = transforms.Compose([transforms.Lambda(lambda img: keynet.fiberbundle.transform(img.convert('RGB'), (28,28))),
                                    transforms.Grayscale(),
                                    transforms.Resize( (28,28) ),
                                    transforms.ToTensor(),
                                    transforms.Normalize((mu,), (std,))])

    trainset = torchvision.datasets.MNIST(root=mnistdir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=32)
    trainpreload = [(x,y) for (x,y) in trainloader]
    testset = torchvision.datasets.MNIST(root=mnistdir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=32)
    testpreload = [(x,y) for (x,y) in testloader]

    # Validate (lenet)
    net1 = keynet.mnist.LeNet_AvgPool()
    net1.load_state_dict(torch.load('./models/mnist_lenet_avgpool.pth'))
    keynet.cifar10.validate(net1, testloader=testpreload)

    # Re-train and Re-validate (lenet)
    net2 = keynet.mnist.LeNet_AvgPool()
    keynet.cifar10.train(net2, modelfile='./models/mnist_lenet_avgpool_fiberbundle.pth', lr=0.004, epochs=40, trainloader=trainpreload)
    net3 = keynet.mnist.LeNet_AvgPool()
    net3.load_state_dict(torch.load('./models/mnist_lenet_avgpool_fiberbundle.pth'))
    keynet.cifar10.validate(net3, testloader=testpreload)


def allconv_fiberbundle(do_mean_estimation=True, mnistdir='/proj/enigma'):
    # Mean
    if do_mean_estimation:
        transform = transforms.Compose([transforms.Lambda(lambda img: keynet.fiberbundle.transform(img.convert('RGB'), (32,32))),
                                        transforms.Grayscale(),
                                        transforms.ToTensor()])
    
        trainset = torchvision.datasets.MNIST(root='/proj/enigma', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=32)
        for (images,labels) in trainloader:
            mu = np.mean(np.array(images), axis=(0,2,3))
            std = np.std(np.array(images), axis=(0,2,3))
            break
        print(mu,std)
    else:
        (mu, std) = ([0.34499013,], [0.12154603,])

    # Load full transformed dataset in memory (parallelized)
    transform = transforms.Compose([transforms.Lambda(lambda img: keynet.fiberbundle.transform(img.convert('RGB'), (32,32))),
                                    transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mu, std)])
                                        
    trainset = torchvision.datasets.MNIST(root=mnistdir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=32)
    trainpreload = [(x,y) for (x,y) in trainloader]
    testset = torchvision.datasets.MNIST(root=mnistdir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=32)
    testpreload = [(x,y) for (x,y) in testloader]

    # Validate (allconv)
    net1 = keynet.cifar10.AllConvNet(1)
    net1.load_state_dict(torch.load('./models/mnist_allconvnet.pth'))
    keynet.cifar10.validate(net1, testloader=testpreload)

    # Re-train and Re-validate (lenet)
    net2 = keynet.cifar10.AllConvNet(1)
    keynet.cifar10.train(net2, modelfile='./models/mnist_allconvnet_fiberbundle.pth', lr=0.003, epochs=40, trainloader=trainpreload)
    net3 = keynet.cifar10.AllConvNet(1)
    net3.load_state_dict(torch.load('./models/mnist_allconvnet_fiberbundle.pth'))
    keynet.cifar10.validate(net3, testloader=testpreload)




def keynet_alpha1():
    net = PermutationKeyNet()
    A0 = sparse_permutation_matrix(28*28*1 + 1)
    A0inv = A0.transpose()
    net.load_state_dict_keyed(torch.load('./models/mnist_lenet_avgpool.pth'), A0inv)
    validate(net, secretkey=A0)


def keynet_alpha1_allconv():
    net = keynet.cifar10.StochasticKeyNet(n_input_channels=1)
    A0 = sparse_permutation_matrix(32*32*1 + 1)
    A0inv = A0.transpose()

    transform = transforms.Compose([transforms.Grayscale(),
                                    transforms.Resize( (32,32) ),
                                    transforms.ToTensor(),        
                                    transforms.Normalize((0.1307,), (0.3081,))])

    net.load_state_dict_keyed(torch.load('./models/mnist_allconvnet.pth'), A0inv)
    validate(net, secretkey=A0, transform=transform)


