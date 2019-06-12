# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:46:49 2019

@author: Interpol
"""

import torch
import torchvision
#import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from PIL import Image
from torchvision import transforms, utils


#%% define dataloader
class SIGN(torch.utils.data.Dataset):
    def __init__(self,csv_file,height,width, transforms=None):
        self.data = pd.read_csv(csv_file)
        self.labels = np.asarray(self.data.iloc[:, 0])
        self.height = height
        self.width = width
        self.transforms = transforms

    def __getitem__(self,index):
        label = self.labels[index]
        img_as_np = np.asarray(self.data.iloc[index][1:]).reshape(28,28).astype('uint8')
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        return (img_as_tensor,label)
    def __len__(self):
       return len(self.data.index)
   
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

train_path = 'SIGN/sign_mnist_train.csv'
test_path = 'SIGN/sign_mnist_test.csv'
batch_size = 4

### the normalize factor augments the image even more, maybe it's 
### a bit redundant since it turns the already grey input images into RGB data ([0..1] for floats or [0..255] for integers)
data_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#data_transforms = transforms.Compose([transforms.ToTensor()])

#use dataloader to translate csv file into images
trainset = SIGN(train_path,224,224,data_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size= batch_size,
                                          shuffle=True, num_workers=4)
classes = ('A', 'B', 'C', 'D',
           'E', 'F', 'G', 'H',
           'I', 'J', 'K', 'L', 
           'M', 'N', 'O', 'P',
           'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X',
           'Y', 'Z')

if __name__ == '__main__':
    # iterate over the data 
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    # upscale image to 224 x 224 required size for the Model
    up_images = F.interpolate(images, size=[224,224], mode='bilinear', align_corners=True)  
        
    # show images
    imshow(torchvision.utils.make_grid(up_images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


###### desirableee
#data_transforms = {
#    'train': transforms.Compose([
#        transforms.RandomResizedCrop(input_size),
#        transforms.RandomHorizontalFlip(),
#        transforms.ToTensor(),
#        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#    ]),
#    'val': transforms.Compose([
#        transforms.Resize(input_size),
#        transforms.CenterCrop(input_size),
#        transforms.ToTensor(),
#        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#    ]),
#}

##### workableeee
#data_transforms = {
#    'train': transforms.Compose([
#        transforms.ToTensor(),
#        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#    ]),
#    'val': transforms.Compose([
#        transforms.ToTensor(),
#        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#    ]),
#}

