import numpy as np
import torch
import torchvision
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from vipy.util import tempdir
import multiprocessing


class AllConvNet(nn.Module):
    """https://github.com/StefOe/all-conv-pytorch/blob/master/cifar10.ipynb"""
    def __init__(self, n_input_channels=3, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.dropout0 = nn.Dropout(p=0.2) 
        self.conv1 = nn.Conv2d(n_input_channels, 96, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv3_bn = nn.BatchNorm2d(96)   # Necessary to get this to train!
        self.dropout3 = nn.Dropout(p=0.5)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv6_bn = nn.BatchNorm2d(192)  # required naming convention 'xyz_bn'        
        self.dropout6 = nn.Dropout(p=0.5) 
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.relu8 = nn.ReLU()
        self.conv9 = nn.Conv2d(192, n_classes, 1)
        self.relu9 = nn.ReLU()
        self.fc1 = nn.Linear(10*8*8, 100)
        self.relu10 = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x_drop = self.dropout0(x)                            # (3,32,32) -> (3,32,32)
        conv1_out = self.relu1(self.conv1(x_drop))           # (3,32,32) -> (96,32,32)
        conv2_out = self.relu2(self.conv2(conv1_out))        # (96,32,32) -> (96,32,32)
        conv3_out = self.conv3(conv2_out)                    # (96,32,32) -> (96,16,16)
        conv3_out_bn = self.conv3_bn(conv3_out)              #
        conv3_out_drop = self.relu3(self.dropout3(conv3_out_bn)) 
        conv4_out = self.relu4(self.conv4(conv3_out_drop))   # (96,16,16) -> (192,16,16)
        conv5_out = self.relu5(self.conv5(conv4_out))        # (192,16,16) -> (192,16,16) 
        conv6_out = self.conv6(conv5_out)                    # (192,16,16) -> (192,8,8) 
        conv6_out_bn = self.conv6_bn(conv6_out)              # 
        conv6_out_drop = self.relu6(self.dropout6(conv6_out_bn)) 
        conv7_out = self.relu7(self.conv7(conv6_out_drop))   # (192,8,8) -> (192,8,8)
        conv8_out = self.relu8(self.conv8(conv7_out))        # (192,8,8) -> (192,8,8)
        conv9_out = self.relu9(self.conv9(conv8_out))        # (192,8,8) -> (10,8,8)
        x = conv9_out.view(-1, 10*8*8)                       # (10,8,8) -> (10*8*8,1)
        x = self.relu10(self.fc1(x))                         # This is not exactly an all-conv, but still in the spirit...
        x = self.fc2(x)                                      # This is not exactly an all-conv, but still in the spirit...
        return x


    @staticmethod
    def transform_train():
        return transforms.Compose([transforms.RandomCrop(32, padding=4),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.49139968,  0.48215841,  0.44653091], [0.24703223,  0.24348513,  0.26158784])])

    @staticmethod
    def transform_test():
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.49139968,  0.48215841,  0.44653091], [0.24703223,  0.24348513,  0.26158784])])

    def loss(self, x):
        return F.log_softmax(x, dim=1)


def validate(net, cifardir=tempdir(), secretkey=None, transform=AllConvNet().transform_test(), num_workers=1, testloader=None):
    if testloader is None:
        testset = torchvision.datasets.CIFAR10(root=cifardir, train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=num_workers)
    net.eval()

    torch.set_grad_enabled(False)
    (total, correct) = (0,0)
    for images,labels in testloader:
        for i in range(len(labels)):
            with torch.no_grad():
                output = net.loss(net(images if secretkey is None else net.encrypt(secretkey,images)))
            _, pred = torch.max(output, 1)
            #pred = output.argmax(dim=1, keepdim=True) 
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    print("Mean classification accuracy = %f" % (correct/total))


def train(net, modelfile, cifardir=tempdir(), epochs=350, lr=0.01, transform=AllConvNet().transform_train(), num_workers=2, trainloader=None):
    if trainloader is None:
        trainset = torchvision.datasets.CIFAR10(root=cifardir, train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=num_workers)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: %s' % device)

    net = nn.DataParallel(net)
    net.to(device)
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 250, 300], gamma=0.1)  
    time0 = time()
    
    torch.set_grad_enabled(True)
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            (images, labels) = (images.to(device), labels.to(device))
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

    torch.save({k.replace('module.',''):v for (k,v) in net.state_dict().items()}, modelfile)
    return net


