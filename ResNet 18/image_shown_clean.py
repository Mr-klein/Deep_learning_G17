# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 16:51:48 2019

@author: Interpol
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from PIL import Image
from data_loader_SIGN import SIGN, imshow

# ---- Parameters ----
train_path = 'SIGN/sign_mnist_train.csv'  #Path to training csv
test_path = 'SIGN/sign_mnist_test.csv'    #Path to test csv
N_classes = 26                            #Number of classes

# Define classes
classes = ('A', 'B', 'C', 'D',
           'E', 'F', 'G', 'H', 'I',
            'J','K','L','M','N','O',
            'P','Q','R','S','T','U','V','W','X','Y','Z')

if __name__ == '__main__': 
    transform = transforms.Compose([transforms.ToTensor()])   # transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    trainset = SIGN(train_path,28,28,transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=1)
    testset = SIGN(test_path,28,28,transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=1)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))