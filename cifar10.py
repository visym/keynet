import numpy as np
import torch
import torchvision
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F


class LeNet(nn.Module):
    """https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"""
    def __init__(self):
        super(LeNet, self).__init__()
        # 3 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    @staticmethod
    def transform():
        return transforms.Compose([transforms.Resize( (32,32) ),
                                   transforms.ToTensor(),       
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])                             


class AllConvNet(nn.Module):
    """https://github.com/StefOe/all-conv-pytorch/blob/master/cifar10.ipynb"""
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out

    @staticmethod
    def transform():
        return transforms.Compose([transforms.Resize( (32,32) ),
                                   transforms.ToTensor(),       
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])                             



def validate(net, cifardir='/proj/enigma', modelfile='/proj/enigma/jebyrne/cifar10.pth'):
    testset = torchvision.datasets.CIFAR10(root=cifardir, train=False, download=True, transform=net.transform())
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    net.load_state_dict(torch.load(modelfile))
    net.eval()

    (total, correct) = (0,0)
    for images,labels in testloader:
        for i in range(len(labels)):
            with torch.no_grad():
                output = net(images)

            _, pred = torch.max(output, 1)
            #pred = output.argmax(dim=1, keepdim=True) 
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    print("Mean classification accuracy = %f" % (correct/total))


def train(net, modelfile, cifardir='/proj/enigma', epochs=40, lr=0.001):
    """https://github.com/pytorch/examples/tree/master/mnist"""
    trainset = torchvision.datasets.CIFAR10(root=cifardir, train=True, download=True, transform=net.transform())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    dataiter = iter(trainloader)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 250, 300], gamma=0.1)
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

        scheduler.step()
        print("Training Time (in minutes) =",(time()-time0)/60)

    torch.save(net.state_dict(), modelfile)
    return net


def lenet():
    net = train(LeNet(), modelfile='/proj/enigma/jebyrne/cifar_lenet.pth', lr=0.001, epochs=80)
    validate(net)

def allconv():
    net = train(AllConvNet(3), modelfile='/proj/enigma/jebyrne/cifar_allconv.pth', lr=0.001, epochs=350)
    validate(net)
