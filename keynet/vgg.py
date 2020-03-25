import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F


def prepare_vggface_image(img):
    """
    Convert an RGB byte image to a FloatTensor suitable for processing with the network.
    This function assumes the image has already been resized, cropped, jittered, etc.
    """
    # Convert to BGR
    img_bgr = np.array(img)[...,[2,1,0]]
    # Subtract mean pixel value
    img_bgr_fp = img_bgr - np.array((93.5940, 104.7624, 129.1863))
    # Permute dimensions so output is 3xRxC
    img_bgr_fp = np.rollaxis(img_bgr_fp, 2, 0)
    return torch.from_numpy(img_bgr_fp).float()


def vggface_preprocess(jitter=False, blur_radius=None, blur_prob=1.0):
    transform_list = [transforms.Resize(256),]
    if jitter:
        transform_list.append(transforms.RandomCrop((224,224)))
        transform_list.append(transforms.RandomHorizontalFlip())
        #transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    else:
        transform_list.append(transforms.CenterCrop((224,224)))
    if blur_radius is not None and blur_prob > 0:
        transform_list.append(transforms.Lambda(generate_random_blur(blur_radius, blur_prob)))
    # finally, convert PIL RGB image to FloatTensor
    transform_list.append(transforms.Lambda(prepare_vggface_image))
    return transforms.Compose(transform_list)


class VGG16(nn.Module):
    """
    The VGG16 network
    """
    def __init__(self, num_classes=2622):
        super(VGG16, self).__init__()
        
        self.conv1_1 = nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1))
        self.relu1_1 = nn.ReLU()        
        self.conv1_2 = nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1))
        self.relu1_2 = nn.ReLU()
        self.pool1_2 = nn.AvgPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True)  # should be max
        
        self.conv2_1 = nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1))
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1))
        self.relu2_2 = nn.ReLU()
        self.pool2_2 = nn.AvgPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True)  # should be max
                
        self.conv3_1 = nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1))
        self.relu3_1 = nn.ReLU()        
        self.conv3_2 = nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1))
        self.relu3_2 = nn.ReLU()        
        self.conv3_3 = nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1))
        self.relu3_3 = nn.ReLU()
        self.pool3_3 = nn.AvgPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True)  # should be max        
        
        self.conv4_1 = nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1))
        self.relu4_1 = nn.ReLU()                
        self.conv4_2 = nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1))
        self.relu4_2 = nn.ReLU()                        
        self.conv4_3 = nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1))
        self.relu4_3 = nn.ReLU()
        self.pool4_3 = nn.AvgPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True)  # should be max                

        self.conv5_1 = nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1))
        self.relu5_1 = nn.ReLU()        
        self.conv5_2 = nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1))
        self.relu5_2 = nn.ReLU()                
        self.conv5_3 = nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1))
        self.relu5_3 = nn.ReLU()
        self.pool5_3 = nn.AvgPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True)  # should be max                        

        self.fc6 = nn.Linear(25088,4096)
        self.relu6 = nn.ReLU()
                             
        self.dropout7 = nn.Dropout(0.5)                             
        self.fc7 = nn.Linear(4096,4096)        
        self.relu7 = nn.ReLU()

        self.dropout8 = nn.Dropout(0.5)                                     
        self.fc8 = nn.Linear(4096, num_classes)


    def forward(self, input):
        """
        Run the network.
        Input should be Nx3x224x224.
        Based on self.mode, return output of fc7, fc8, or both.
        """
        assert len(input.size()) == 4

        e1_1 = self.relu1_1(self.conv1_1(input))
        e1_2 = self.pool1_2(self.relu1_2(self.conv1_2(e1_1)))

        e2_1 = self.relu2_1(self.conv2_1(e1_2))
        e2_2 = self.pool2_2(self.relu2_2(self.conv2_2(e2_1)))

        e3_1 = self.relu3_1(self.conv3_1(e2_2))
        e3_2 = self.relu3_2(self.conv3_2(e3_1))
        e3_3 = self.pool3_3(self.relu3_3(self.conv3_3(e3_2)))

        e4_1 = self.relu4_1(self.conv4_1(e3_3))
        e4_2 = self.relu4_2(self.conv4_2(e4_1))
        e4_3 = self.pool4_3(self.relu4_3(self.conv4_3(e4_2)))

        e5_1 = self.relu5_1(self.conv5_1(e4_3))
        e5_2 = self.relu5_2(self.conv5_2(e5_1))
        e5_3 = self.pool5_3(self.relu5_3(self.conv5_3(e5_2)))

        e5_3_flat = e5_3.view(e5_3.size(0), -1)

        e6 = self.relu6(self.fc6(e5_3_flat))
        e7_pre = self.fc7(self.dropout7(e6))
        e7 = self.relu7(e7_pre)

        e8 = self.fc8(self.dropout8(e7))
        return e8


def vgg16(pthfile):
    """
    Constructs a VGG-16 model
    """
    model = VGGFace()
    model.load_state_dict(torch.load(pthfile))
    return model
