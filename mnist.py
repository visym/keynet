import numpy as np
import torch
import torchvision
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import keynet.layers
from keynet.util import sparse_permutation_matrix, sparse_identity_matrix

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
        return F.log_softmax(x, dim=1)

    @staticmethod
    def transform():
        return transforms.Compose([transforms.ToTensor(),                                    
                                   transforms.Normalize((0.1307,), (0.3081,))])

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
    def __init__(self):
        super(KeyNet, self).__init__()
        self.conv1 = keynet.layers.KeyedConv2d(1, 6, kernel_size=3, stride=1)  # assumed padding=1
        self.pool1 = keynet.layers.KeyedAvgpool2d(kernel_size=3, stride=2)
        self.conv2 = keynet.layers.KeyedConv2d(6, 16, kernel_size=3, stride=1)  # assumed padding=1
        self.pool2 = keynet.layers.KeyedAvgpool2d(3,2)
        self.fc1 = keynet.layers.KeyedLinear(7*7*16, 120)
        self.fc2 = keynet.layers.KeyedLinear(120, 84)
        self.fc3 = keynet.layers.KeyedLinear(84, 10)

        self.shape = {'x0':(1,28,28),   # image input
                      'x1':(6,28,28),   # pool1 input
                      'x2':(6,14,14),   # conv2 input
                      'x3':(16,14,14),  # pool2 input
                      'x4':(16,7,7),    # fc1 input
                      'x5':(120,1,1),   # fc2 input
                      'x6':(84,1,1),    # fc3 input
                      'x7':(10,1,1)}    # output 

        self.keys = {'A0inv':None,
                     'A1':sparse_identity_matrix(np.prod(self.shape['x1'])+1),
                     'A1inv':sparse_identity_matrix(np.prod(self.shape['x1'])+1),
                     'A2':sparse_identity_matrix(np.prod(self.shape['x2'])+1),
                     'A2inv':sparse_identity_matrix(np.prod(self.shape['x2'])+1),                     
                     'A3':sparse_identity_matrix(np.prod(self.shape['x3'])+1),
                     'A3inv':sparse_identity_matrix(np.prod(self.shape['x3'])+1),
                     'A4':sparse_identity_matrix(np.prod(self.shape['x4'])+1),
                     'A4inv':sparse_identity_matrix(np.prod(self.shape['x4'])+1),
                     'A5':sparse_identity_matrix(np.prod(self.shape['x5'])+1),
                     'A5inv':sparse_identity_matrix(np.prod(self.shape['x5'])+1),
                     'A6':sparse_identity_matrix(np.prod(self.shape['x6'])+1),
                     'A6inv':sparse_identity_matrix(np.prod(self.shape['x6'])+1)}

        
    def load_state_dict_keyed(self, d_state, A0inv):        
        self.conv1.key(np.array(d_state['conv1.weight']), np.array(d_state['conv1.bias']), self.keys['A1'], A0inv, self.shape['x0'])
        self.pool1.key(self.keys['A2'], self.keys['A1inv'], self.shape['x1'])
        self.conv2.key(np.array(d_state['conv2.weight']), np.array(d_state['conv2.bias']), self.keys['A3'], self.keys['A2inv'], self.shape['x2'])
        self.pool2.key(self.keys['A4'], self.keys['A3inv'], self.shape['x3'])
        self.fc1.key(np.array(d_state['fc1.weight']), np.array(d_state['fc1.bias']), self.keys['A5'], self.keys['A4inv'])
        self.fc2.key(np.array(d_state['fc2.weight']), np.array(d_state['fc2.bias']), self.keys['A6'], self.keys['A5inv'])
        self.fc3.key(np.array(d_state['fc3.weight']), np.array(d_state['fc3.bias']), None, self.keys['A6inv'])


    def forward(self, A0x0):
        (N,C,U,V) = A0x0.shape
        A0x0_affine = torch.cat( (A0x0.view(N,C*U*V), torch.ones(N,1)), dim=1) # N x (C*U*V+1)
        x1 = F.relu(self.conv1(A0x0_affine))
        x2 = self.pool1(x1)
        x3 = F.relu(self.conv2(x2))
        x4 = self.pool2(x3)
        x5 = F.relu(self.fc1(x4.view(-1, 7*7*16 + 1)))
        x6 = F.relu(self.fc2(x5))
        x7 = self.fc3(x6)
        x7_linear = torch.narrow(x7, 1, 0, self.shape['x7'][0])
        return F.log_softmax(x7_linear, dim=1)

    @staticmethod
    def transform():
        return transforms.Compose([transforms.ToTensor(),                                    
                                   transforms.Normalize((0.1307,), (0.3081,))])


def validate(net, mnistdir='/proj/enigma'):
    valset = datasets.MNIST(mnistdir, download=True, train=False, transform=net.transform())
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
    net.eval()

    (total, correct) = (0,0)
    for images,labels in valloader:
        for i in range(len(labels)):
            with torch.no_grad():
                output = net(images)

            _, pred = torch.max(output, 1)
            #pred = output.argmax(dim=1, keepdim=True) 
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    print("Mean classification accuracy = %f" % (correct/total))


def train(net, modelfile, mnistdir='/proj/enigma', lr=0.003, epochs=20):
    trainset = datasets.MNIST(mnistdir, download=True, train=True, transform=net.transform())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    #criterion = nn.CrossEntropyLoss()
    criterion = F.nll_loss
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    time0 = time()

    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()            
            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()        
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))

        print("Training Time (in minutes) =",(time()-time0)/60)

    torch.save(net.state_dict(), modelfile)
    return net

def lenet():
    net = train(LeNet(), modelfile='/proj/enigma/jebyrne/mnist_lenet.pth', lr=0.003, epochs=20)
    validate(net)

def lenet_avgpool():
    net = train(LeNet_AvgPool(), modelfile='/proj/enigma/jebyrne/mnist_lenet_avgpool.pth', lr=0.003, epochs=20)
    validate(net)

def keynet_alpha1():
    net = KeyNet()
    A0inv = sparse_identity_matrix(28*28*1 + 1)
    net.load_state_dict_keyed(torch.load('/proj/enigma/jebyrne/mnist_lenet_avgpool.pth'), A0inv)
    net.eval()
    return net
