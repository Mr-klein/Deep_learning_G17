# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:30:12 2019

@author: Zhouxin
"""
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


import os
import sys
train_path = 'SIGN/sign_mnist_train.csv'
test_path = 'SIGN/sign_mnist_test.csv'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
 
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    try:
        # simulate large output (your code replaces this loop)
#        for x in range(10000):
#            print("y")

        since = time.time()
    
        val_acc_history = []
    
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
    
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
    
            # Each epoch has a training and validation phase
            for phase in [train_path, test_path]:
                if phase == train_path:
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
    
                running_loss = 0.0
                running_corrects = 0
                
                
                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    
                    inputs = inputs.to(device)
                    labels = labels.to(device)
    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == train_path):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == train_path:
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
    
                        _, preds = torch.max(outputs, 1)
    
                        # backward + optimize only if in training phase
                        if phase == train_path:
                            loss.backward()
                            optimizer.step()
    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
    
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
    
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    
                # deep copy the model
                if phase == test_path and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == test_path:
                    val_acc_history.append(epoch_acc)
    
            print()
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
    
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history
        # flush output here to force SIGPIPE to be triggered
        # while inside this try block.
        sys.stdout.flush()
    except BrokenPipeError:
        # Python flushes standard streams on exit; redirect remaining output
        # to devnull to avoid another BrokenPipeError at shutdown
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(1)  # Python exits with error code 1 on EPIPE
        
#        
#if __name__ == '__main__':
#    train_model()