# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:43:42 2019

@author: Zhouxin
"""
#https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from train_model import train_model
from data_loader_SIGN import SIGN, imshow
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
#data_dir = "./SIGN/"
train_path = 'SIGN/sign_mnist_train.csv'
test_path = 'SIGN/sign_mnist_test.csv'


# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 26

# Batch size for training (change depending on how much memory you have)
batch_size = 4

# Number of epochs to train for
num_epochs = 10

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
#print(model_ft)

# Data augmentation and normalization for training
# Just normalization for validation

### the normalize factor augments the image even more, maybe it's
### a bit redundant since it turns the already grey input images into RGB data ([0..1] for floats or [0..255] for integers)
data_transforms = transforms.Compose([transforms.Resize(224, interpolation=2),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#data_transforms = data_transforms.unsqueeze_(0)
    #transforms.ToPILImage(mode=None),
#data_transforms = data_transforms.repeat(3,1,1)

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
#old system of dataloader and augmentation
# Create training and validation datasets
#image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                          data_transforms[x]) for x in ['train', 'val']}
#trainset = SIGN(train_path,224,224,data_transforms)
#testset = SIGN(test_path,224,224,data_transforms)
#
#if __name__ == '__main__':

image_datasets = {x:  SIGN(x,224,224,data_transforms) for x in [train_path, test_path]}

# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                   batch_size=batch_size,
                                                   shuffle=True, num_workers=4) for x in [train_path, test_path]}

#up_image_dict = {x: F.interpolate(dataloaders_dict, size=[224,224],
#                                  mode='bilinear', align_corners=True) for x in [train_path, test_path]}

#trainloader = torch.utils.data.DataLoader(trainset,
#                                          batch_size= batch_size,
#                                          shuffle=True, num_workers=4)
#testloader = torch.utils.data.DataLoader(testset,
#                                             batch_size= batch_size,
#                                             shuffle=True, num_workers=4)


classes = ('A', 'B', 'C', 'D',
           'E', 'F', 'G', 'H',
           'I', 'J', 'K', 'L',
           'M', 'N', 'O', 'P',
           'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X',
           'Y', 'Z')


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
if __name__ == '__main__':
    # show images
#    imshow(torchvision.utils.make_grid(up_image_dict))
#    # print labels
#    print(' '.join('%5s' % classes[labels[0][j]] for j in range(4)))

    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,
                                 num_epochs=num_epochs, is_inception=(model_name=="inception"))
    print('Saving Model Parameters...')
    torch.save(model_ft.state_dict(), 'resmodel_weights.pth')
    print('done')
