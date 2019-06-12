# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:12:24 2019

@author: Zhouxin
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
        img_as_img = img_as_img.convert('RGB')
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
            img_as_tensor = img_as_tensor.reshape1(1,224,224)
            #img_as_tensor =  torch.stack((img_as_tensor, torch.zeros(224,224), torch.zeros(224,224)))
            #print(img_as_tensor.size())
            #img_as_tensor =  torch.stack((img_as_tensor, img_as_tensor, img_as_tensor))

        return (img_as_tensor,label)
    def __len__(self):
       return len(self.data.index)

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#transforms.Lambda(lambda x: x.expand(3, 1, 1))

#transforms.Lambda(lambda img_as_img: img_as_img.repeat(3, 1, 1))
