import numpy as np
import torch
import torchvision
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from vipy.util import Stopwatch, tempdir


class LeNet(nn.Module):
    """Slightly modified LeNet to include padding, odd filter sizes and even image sizes"""
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1, stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()        
        self.fc3 = nn.Linear(84, 10)

        
    def forward(self, x):
        x = self.relu1(self.conv1(x))    # (1,28,28) -> (6,28,28)
        x = self.pool1(x)                # (6,28,28) -> (6,14,14)
        x = self.relu2(self.conv2(x))    # (6,14,14) -> (16,14,14)  
        x = self.pool2(x)                # (16,14,14) -> (16,7,7)
        x = x.view(-1, 7*7*16)           # (16,7,7) -> (16*7*7,1)
        x = self.relu3(self.fc1(x))          # (16*7*7,1) -> (120,1)
        x = self.relu4(self.fc2(x))          # (120,1) -> (84,1)
        x = self.fc3(x)                  # (84,1) -> (10,1)
        return x

    def loss(self, x):
        return F.log_softmax(x, dim=1)

    def transform(self):
        return transforms.Compose([transforms.ToTensor(),        
                                   transforms.Normalize((0.1307,), (0.3081,))])

    
class LeNet_AvgPool(LeNet):
    """https://github.com/pytorch/examples/blob/master/mnist/main.py"""
    def __init__(self):
        super(LeNet_AvgPool, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(7*7*16, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        

def train(net, modelfile, mnistdir=tempdir(), lr=0.003, epochs=20, transform=LeNet().transform()):
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

    with Stopwatch() as sw:
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



