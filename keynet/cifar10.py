import numpy as np
import torch
import torchvision
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import keynet.mnist 
import keynet.layers
import keynet.fiberbundle
from keynet.util import sparse_permutation_matrix, sparse_identity_matrix, sparse_uniform_random_diagonal_matrix, sparse_inverse_diagonal_matrix
from keynet.torch import affine_augmentation_tensor, affine_deaugmentation_tensor
from keynet.util import sparse_generalized_permutation_block_matrix
import multiprocessing
import keynet.torch


class AllConvNet(nn.Module):
    """https://github.com/StefOe/all-conv-pytorch/blob/master/cifar10.ipynb"""
    def __init__(self, n_input_channels=3, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.dropout0 = nn.Dropout(p=0.2) 
        self.conv1 = nn.Conv2d(n_input_channels, 96, 3, padding=1) 
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv3_bn = nn.BatchNorm2d(96)   # Necessary to get this to train!
        self.dropout3 = nn.Dropout(p=0.5) 
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.dropout6 = nn.Dropout(p=0.5) 
        self.conv6_bn = nn.BatchNorm2d(192)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.conv9 = nn.Conv2d(192, n_classes, 1)
        self.fc1 = nn.Linear(10*8*8, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x_drop = self.dropout0(x)                            # (3,32,32) -> (3,32,32)
        conv1_out = F.relu(self.conv1(x_drop))               # (3,32,32) -> (96,32,32)
        conv2_out = F.relu(self.conv2(conv1_out))            # (96,32,32) -> (96,32,32)
        conv3_out = self.conv3(conv2_out)                    # (96,32,32) -> (96,16,16)
        conv3_out_bn = self.conv3_bn(conv3_out)              #
        conv3_out_drop = F.relu(self.dropout3(conv3_out_bn)) # 
        conv4_out = F.relu(self.conv4(conv3_out_drop))       # (96,16,16) -> (192,16,16)
        conv5_out = F.relu(self.conv5(conv4_out))            # (192,16,16) -> (192,16,16) 
        conv6_out = self.conv6(conv5_out)                    # (192,16,16) -> (192,8,8) 
        conv6_out_bn = self.conv6_bn(conv6_out)              # 
        conv6_out_drop = F.relu(self.dropout6(conv6_out_bn)) # 
        conv7_out = F.relu(self.conv7(conv6_out_drop))       # (192,8,8) -> (192,8,8)
        conv8_out = F.relu(self.conv8(conv7_out))            # (192,8,8) -> (192,8,8)
        conv9_out = F.relu(self.conv9(conv8_out))            # (192,8,8) -> (10,8,8)
        x = conv9_out.view(-1, 10*8*8)                       # (10,8,8) -> (10*8*8,1)
        x = F.relu(self.fc1(x))                              # This is not exactly an all-conv...
        x = self.fc2(x)                                      # This is not exactly an all-conv...
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


class StochasticKeyNet(AllConvNet):
    def __init__(self, keys=None, n_input_channels=3, alpha=1, use_torch_sparse=False):
        super(StochasticKeyNet, self).__init__()

        self.conv1 = keynet.layers.KeyedConv2d(n_input_channels, 96, kernel_size=3, stride=1, use_torch_sparse=use_torch_sparse)  # assumed padding=1
        self.relu1 = keynet.layers.KeyedRelu(use_torch_sparse=use_torch_sparse)
        self.conv2 = keynet.layers.KeyedConv2d(96, 96, kernel_size=3, stride=1, use_torch_sparse=use_torch_sparse)  # assumed padding=1
        #self.relu2 = keynet.layers.KeyedRelu(use_torch_sparse=use_torch_sparse)
        self.conv3 = keynet.layers.KeyedConv2d(96, 96, kernel_size=3, stride=2, use_torch_sparse=use_torch_sparse)  # assumed padding=1
        self.relu3 = keynet.layers.KeyedRelu(use_torch_sparse=use_torch_sparse)
        self.conv4 = keynet.layers.KeyedConv2d(96, 192, kernel_size=3, stride=1, use_torch_sparse=use_torch_sparse)  # assumed padding=1
        self.relu4 = keynet.layers.KeyedRelu(use_torch_sparse=use_torch_sparse)
        self.conv5 = keynet.layers.KeyedConv2d(192, 192, kernel_size=3, stride=1, use_torch_sparse=use_torch_sparse)  # assumed padding=1
        #self.relu5 = keynet.layers.KeyedRelu(use_torch_sparse=use_torch_sparse)
        self.conv6 = keynet.layers.KeyedConv2d(192, 192, kernel_size=3, stride=2, use_torch_sparse=use_torch_sparse)  # assumed padding=1
        self.relu6 = keynet.layers.KeyedRelu(use_torch_sparse=use_torch_sparse)
        self.conv7 = keynet.layers.KeyedConv2d(192, 192, kernel_size=3, stride=1, use_torch_sparse=use_torch_sparse)  # assumed padding=1
        #self.relu7 = keynet.layers.KeyedRelu(use_torch_sparse=use_torch_sparse)
        self.conv8 = keynet.layers.KeyedConv2d(192, 192, kernel_size=1, stride=1, use_torch_sparse=use_torch_sparse)  # assumed padding=0
        #self.relu8 = keynet.layers.KeyedRelu(use_torch_sparse=use_torch_sparse)
        self.conv9 = keynet.layers.KeyedConv2d(192, 10, kernel_size=1, stride=1, use_torch_sparse=use_torch_sparse)  # assumed padding=0
        self.relu9 = keynet.layers.KeyedRelu(use_torch_sparse=use_torch_sparse)
        self.fc1 = keynet.layers.KeyedLinear(10*8*8, 100)  
        self.relu10 = keynet.layers.KeyedRelu(use_torch_sparse=use_torch_sparse)
        self.fc2 = keynet.layers.KeyedLinear(100, 10)  

        # Layer output shapes:  x1 = conv1(x0)
        self.shape = {'x0':(n_input_channels,32,32),   # image input
                      'x1':(96,32,32),    
                      'x2':(96,32,32), 
                      'x3':(96,16,16), 
                      'x4':(192,16,16),   
                      'x5':(192,16,16),   
                      'x6':(192,8,8),  
                      'x7':(192,8,8),  
                      'x8':(192,8,8),   
                      'x9':(10*8*8,1,1),
                      'x10':(100,1,1),
                      'x11':(10,1,1)}   

        (A1a,A1ainv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x1'])+1, alpha) 
        (A1b,A1binv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x1'])+1, 1) 
        (A2a,A2ainv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x2'])+1, alpha) 
        #(A2b,A2binv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x2'])+1, 1)   # reused A1b
        (A3a,A3ainv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x3'])+1, alpha) 
        (A3b,A3binv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x3'])+1, 1) 
        (A4a,A4ainv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x4'])+1, alpha) 
        (A4b,A4binv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x4'])+1, 1) 
        (A5a,A5ainv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x5'])+1, alpha) 
        #(A5b,A5binv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x5'])+1, 1)  # Reused A4b
        (A6a,A6ainv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x6'])+1, alpha) 
        (A6b,A6binv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x6'])+1, 1) 
        (A7a,A7ainv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x7'])+1, alpha) 
        #(A7b,A7binv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x7'])+1, 1)  # Reused A6b
        (A8a,A8ainv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x8'])+1, alpha) 
        #(A8b,A8binv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x8'])+1, 1)  # Reused A6b
        (A9a,A9ainv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x9'])+1, alpha) 
        (A9b,A9binv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x9'])+1, 1) 
        (A10a,A10ainv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x10'])+1, alpha) 
        (A10b,A10binv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x10'])+1, 1) 
        (A11,A11inv) = sparse_generalized_permutation_block_matrix(np.prod(self.shape['x11'])+1, alpha) 
        
        self.keys = {'A0inv':None,'A1a':A1a,'A1ainv':A1ainv,'A1b':A1b,'A1binv':A1binv,
                     'A2a':A2a,'A2ainv':A2ainv, 'A2b':A1b,'A2binv':A1binv,  
                     'A3a':A3a,'A3ainv':A3ainv, 'A3b':A3b,'A3binv':A3binv,
                     'A4a':A4a,'A4ainv':A4ainv, 'A4b':A4b,'A4binv':A4binv,
                     'A5a':A5a,'A5ainv':A5ainv, 'A5b':A4b,'A5binv':A4binv,
                     'A6a':A6a,'A6ainv':A6ainv, 'A6b':A6b,'A6binv':A6binv,
                     'A7a':A7a,'A7ainv':A7ainv, 'A7b':A6b,'A7binv':A6binv,
                     'A8a':A8a,'A8ainv':A8ainv, 'A8b':A6b,'A8binv':A6binv,
                     'A9a':A9a,'A9ainv':A9ainv, 'A9b':A9b,'A9binv':A9binv,
                     'A10a':A10a,'A10ainv':A10ainv, 'A10b':A10b,'A10binv':A10binv,
                     'A11':A11,'A11inv':A11inv}

        
    def load_state_dict_keyed(self, d_state, A0inv):        
        d_state = {k.replace('module.',''):v.cpu() for (k,v) in d_state.items()}  # nn.DataParallel training cruft (if needed)

        self.keys['A0inv'] = A0inv
        self.conv1.key(np.array(d_state['conv1.weight']), np.array(d_state['conv1.bias']), self.keys['A1a'], self.keys['A0inv'], self.shape['x0'])
        self.relu1.key(self.keys['A1b'], self.keys['A1ainv'])
        self.conv2.key(np.array(d_state['conv2.weight']), np.array(d_state['conv2.bias']), self.keys['A1a'], self.keys['A1binv'], self.shape['x1'])
        #self.relu2.key(self.keys['A2b'], self.keys['A2ainv'])
        (conv3bn_weight, conv3bn_bias) = keynet.torch.fuse_conv2d_and_bn(d_state['conv3.weight'], d_state['conv3.bias'], 
                                                                          d_state['conv3_bn.running_mean'], d_state['conv3_bn.running_var'], 1E-5,
                                                                          d_state['conv3_bn.weight'], d_state['conv3_bn.bias'])
        self.conv3.key(np.array(conv3bn_weight), np.array(conv3bn_bias), self.keys['A3a'], self.keys['A2binv'], self.shape['x2'])
        self.relu3.key(self.keys['A3b'], self.keys['A3ainv'])
        self.conv4.key(np.array(d_state['conv4.weight']), np.array(d_state['conv4.bias']), self.keys['A4a'], self.keys['A3binv'], self.shape['x3'])
        self.relu4.key(self.keys['A4b'], self.keys['A4ainv'])
        self.conv5.key(np.array(d_state['conv5.weight']), np.array(d_state['conv5.bias']), self.keys['A4a'], self.keys['A4binv'], self.shape['x4'])
        #self.relu5.key(self.keys['A5b'], self.keys['A5ainv'])
        (conv6bn_weight, conv6bn_bias) = keynet.torch.fuse_conv2d_and_bn(d_state['conv6.weight'], d_state['conv6.bias'], 
                                                                          d_state['conv6_bn.running_mean'], d_state['conv6_bn.running_var'], 1E-5,
                                                                          d_state['conv6_bn.weight'], d_state['conv6_bn.bias'])
        self.conv6.key(np.array(conv6bn_weight), np.array(conv6bn_bias), self.keys['A6a'], self.keys['A5binv'], self.shape['x5'])
        self.relu6.key(self.keys['A6b'], self.keys['A6ainv'])
        self.conv7.key(np.array(d_state['conv7.weight']), np.array(d_state['conv7.bias']), self.keys['A6a'], self.keys['A6binv'], self.shape['x6'])
        #self.relu7.key(self.keys['A7b'], self.keys['A7ainv'])
        self.conv8.key(np.array(d_state['conv8.weight']), np.array(d_state['conv8.bias']), self.keys['A6a'], self.keys['A7binv'], self.shape['x7'])
        #self.relu8.key(self.keys['A8b'], self.keys['A8ainv'])
        self.conv9.key(np.array(d_state['conv9.weight']), np.array(d_state['conv9.bias']), self.keys['A9a'], self.keys['A8binv'], self.shape['x8'])
        self.relu9.key(self.keys['A9b'], self.keys['A9ainv'])

        self.fc1.key(np.array(d_state['fc1.weight']), np.array(d_state['fc1.bias']), self.keys['A10a'], self.keys['A9binv'])
        self.relu10.key(self.keys['A10b'], self.keys['A10ainv'])
        self.fc2.key(np.array(d_state['fc2.weight']), np.array(d_state['fc2.bias']), self.keys['A11'], self.keys['A10binv'])


    def forward(self, A0x0_affine):
        #x_drop = self.dropout0(x)                        #   Identity at test time
        conv1_out = self.relu1(self.conv1(A0x0_affine))       # (3,32,32) -> (96,32,32)
        conv2_out = self.relu1(self.conv2(conv1_out))         # (96,32,32) -> (96,32,32), reuse relu1
        conv3_out = self.relu3(self.conv3(conv2_out))         # (96,32,32) -> (96,16,16)
        #conv3_out_bn = self.conv3_bn(conv3_out)          #   Merged into conv3
        #conv3_out_drop = self.dropout3(conv3_out_bn)     #   Identity at test time
        conv4_out = self.relu4(self.conv4(conv3_out))         # (96,16,16) -> (192,16,16)
        conv5_out = self.relu4(self.conv5(conv4_out))         # (192,16,16) -> (192,16,16), reuse relu4
        conv6_out = self.relu6(self.conv6(conv5_out))         # (192,16,16) -> (192,8,8) 
        #conv6_out_bn = self.conv6_bn(conv6_out)          #   Merged into conv6  
        #conv6_out_drop = self.dropout6(conv6_out_bn)     #   Identity at test time
        conv7_out = self.relu6(self.conv7(conv6_out))         # (192,8,8) -> (192,8,8), reuse relu6
        conv8_out = self.relu6(self.conv8(conv7_out))         # (192,8,8) -> (192,8,8), reuse relu6  
        conv9_out = self.relu9(self.conv9(conv8_out))         # (192,8,8) -> (10,8,8)
        x = conv9_out.view(10*8*8+1, -1)                  # (10,8,8) -> (10*8*8,1), C*U*VxN transposed
        x = self.relu10(self.fc1(x))                           # This is not exactly an all-conv...
        x = self.fc2(x)                                   # This is not exactly an all-conv...
        return x

    def encrypt(self, A0, x):        
        return torch.tensor(A0.dot(affine_augmentation_tensor(x))) if A0 is not None else x

    def decrypt(self, x):
        if self.keys['A11inv'] is not None and self.keys['A11'] is not None:
            return affine_deaugmentation_tensor(torch.tensor(self.keys['A11inv'].dot(x)))  
        else:
            return affine_deaugmentation_tensor(x)

    def loss(self, x):
        return self.decrypt(x)


def validate(net, cifardir='/proj/enigma', secretkey=None, transform=AllConvNet().transform_test(), num_workers=1, testloader=None):
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


def train(net, modelfile, cifardir='/proj/enigma', epochs=350, lr=0.01, transform=AllConvNet().transform_train(), num_workers=2, trainloader=None):
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


def allconv():
    train(AllConvNet(), modelfile='./models/cifar10_allconv.pth', lr=0.01, epochs=350)
    testmodel = AllConvNet()
    testmodel.load_state_dict(torch.load('./models/cifar10_allconv.pth'))
    validate(testmodel)

def lenet():
    net = keynet.mnist.LeNet()
    train(net, modelfile='./models/cifar10_lenet.pth', lr=0.01, epochs=350, transform=net.transform_cifar10_train())
    testnet = keynet.mnist.LeNet()
    testnet.load_state_dict(torch.load('./models/cifar10_lenet.pth'))
    validate(testnet, transform=testnet.transform_cifar10_test())

def lenet_avgpool():
    net = keynet.mnist.LeNet_AvgPool()
    train(net, modelfile='./models/cifar10_lenet_avgpool.pth', lr=0.01, epochs=350, transform=net.transform_cifar10_train())
    testnet = keynet.mnist.LeNet_AvgPool()
    testnet.load_state_dict(torch.load('./models/cifar10_lenet_avgpool.pth'))
    validate(testnet, transform=testnet.transform_cifar10_test())

def lenet_avgpool_fiberbundle(do_mean_estimation=True, cifardir='/proj/enigma'):
    # Mean
    if do_mean_estimation:
        transform = transforms.Compose([transforms.Lambda(lambda img: keynet.fiberbundle.transform(img, (28,28))),
                                        transforms.Grayscale(),
                                        transforms.Resize( (28,28) ),
                                    transforms.ToTensor()])
    
        trainset = torchvision.datasets.CIFAR10(root='/proj/enigma', train=True, download=True, transform=transform)
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
    transform = transforms.Compose([transforms.Lambda(lambda img: keynet.fiberbundle.transform(img, (28,28))),
                                    transforms.Grayscale(),
                                    transforms.Resize( (28,28) ),
                                    transforms.ToTensor(),
                                    transforms.Normalize((mu,), (std,))])

    trainset = torchvision.datasets.CIFAR10(root=cifardir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=32)
    trainpreload = [(x,y) for (x,y) in trainloader]
    testset = torchvision.datasets.CIFAR10(root=cifardir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=32)
    testpreload = [(x,y) for (x,y) in testloader]

    # Validate (lenet)
    net1 = keynet.mnist.LeNet_AvgPool()
    net1.load_state_dict(torch.load('./models/cifar10_lenet_avgpool.pth'))
    validate(net1, testloader=testpreload)

    # Re-train and Re-validate (lenet)
    net2 = keynet.mnist.LeNet_AvgPool()
    train(net2, modelfile='./models/cifar10_lenet_avgpool_fiberbundle.pth', lr=0.01, epochs=350, trainloader=trainpreload)
    net3 = keynet.mnist.LeNet_AvgPool()
    net3.load_state_dict(torch.load('./models/cifar10_lenet_avgpool_fiberbundle.pth'))
    validate(net3, testloader=testpreload)


def allconv_fiberbundle(do_mean_estimation=True, cifardir='/proj/enigma'):
    # Mean
    if do_mean_estimation:
        transform = transforms.Compose([transforms.Lambda(lambda img: keynet.fiberbundle.transform(img, (32,32))),
                                        transforms.ToTensor()])
    
        trainset = torchvision.datasets.CIFAR10(root='/proj/enigma', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=32)
        for (images,labels) in trainloader:
            mu = np.mean(np.array(images), axis=(0,2,3))
            std = np.std(np.array(images), axis=(0,2,3))
            break
        print(mu,std)
    else:
        (mu, std) = ([0.5864967,  0.58052236, 0.48031753], [0.08658934, 0.09825305, 0.04734877])

    # Load full transformed dataset in memory (parallelized)
    train_transform = transforms.Compose([transforms.Lambda(lambda img: keynet.fiberbundle.transform(img, (32,32))),
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mu, std)])
    test_transform = transforms.Compose([transforms.Lambda(lambda img: keynet.fiberbundle.transform(img, (32,32))),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mu, std)])
                                        
    trainset = torchvision.datasets.CIFAR10(root=cifardir, train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=32)
    trainpreload = [(x,y) for (x,y) in trainloader]
    testset = torchvision.datasets.CIFAR10(root=cifardir, train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=32)
    testpreload = [(x,y) for (x,y) in testloader]

    # Validate (allconv)
    net1 = AllConvNet(3)
    net1.load_state_dict(torch.load('./models/cifar10_allconv.pth'))
    validate(net1, testloader=testpreload)

    # Re-train and Re-validate (lenet)
    net2 = AllConvNet(3)
    train(net2, modelfile='./models/cifar10_allconv_fiberbundle.pth', lr=0.01, epochs=350, trainloader=trainpreload)
    net3 = AllConvNet(3)
    net3.load_state_dict(torch.load('./models/cifar10_allconv_fiberbundle.pth'))
    validate(net3, testloader=testpreload)


def keynet_alpha1():
    A0 = sparse_permutation_matrix(32*32*3 + 1)
    A0inv = A0.transpose()
    net = keynet.cifar10.StochasticKeyNet()
    net.load_state_dict_keyed(torch.load('./models/cifar10_allconv.pth'), A0inv=A0inv)
    validate(net, secretkey=A0)

def keynet_alpha1_lenet_avgpool():
    A0 = sparse_permutation_matrix(28*28*1 + 1)
    A0inv = A0.transpose()
    net = keynet.mnist.StochasticKeyNet()
    net.load_state_dict_keyed(torch.load('./models/cifar10_lenet_avgpool.pth'), A0inv=A0inv)
    validate(net, secretkey=A0, transform=keynet.mnist.LeNet_AvgPool().transform_cifar10_test())

