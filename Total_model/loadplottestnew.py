#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#
"""
Created on Mon May 20 10:46:58 2019

@author: RJ  (classifier loading and testing)
        : KK (plot intermediate filter outputs and save confusion matrix)

       Loads the neural network weights. plots the intermediate filter outputs and
       tests the model accuracy using the test data set. also outpus a confusion matrix
       as npy file
"""
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import torchviz as tv
from PIL import Image

#%% define parameters
param_path = 'weights/weightfs3lr1mo3.pth'     # path to model param
test_path = 'SIGN/sign_mnist_test.csv'  # path to test csv

N_classes = 26          # number of classes
batch = 8               # batch size

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

#%% load data
transform = transforms.Compose([transforms.ToTensor()])
testset = SIGN(test_path,28,28,transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch,
                                         shuffle=True, num_workers=1)

#%% define classes
classes = ('A', 'B', 'C', 'D',
           'E', 'F', 'G', 'H',
           'I', 'J', 'K', 'L',
           'M', 'N', 'O', 'P',
           'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X',
           'Y', 'Z')
fs = [48,96,144,192]
fc1 = 120
fc2 = 80
#%% define neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
        #        dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(1, 48, 3, padding=1)
        self.conv2 = nn.Conv2d(48, 96, 3, padding=1)
        # MaxPool2d(kernel_size, stride=None, padding=0, dilation=1,
        #           return_indices=False, ceil_mode=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(96,144, 3, padding=1)
        self.conv4 = nn.Conv2d(144,192, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(192 * 7 * 7, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, N_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 192 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#%% testing network
if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    net = Net()
    if use_gpu:
        net = net.cuda()
    net.load_state_dict(torch.load(param_path,map_location='cpu'))
    net.eval()

# iets van visualisatie
    for data in testloader:
        images, labels = data
        x1 = net.conv1(images)
        x1_activation = F.relu(x1)

        x2 = net.conv2(x1_activation)
        x2_activation = F.relu(x2)

        x3 = net.pool(x2_activation)

        x4 = net.conv3(x3)
        x4_activation = F.relu(x4)

        x5 = net.conv4(x4_activation)
        x5_activation = F.relu(x5)

        x6 = net.pool(x5_activation)

        y = x6.view(-1, 192 * 7 * 7)

        x7 = net.fc1(y)
        x7_activation = F.relu(x7)

        x8 = net.fc2(x7_activation)
        x8_activation = F.relu(x8)

        x9 = net.fc3(x8_activation)

        y = x5_activation.detach().numpy()


        # ----- PARAMETER FOR PLOTTING
        toplot = x5_activation   #Convolutional activation to plot
        letter_label = 2        # letter to plot activation for

        #look for matching labels
        if labels[0] == letter_label:
            x_grid = toplot[0].view(toplot[0].size()[0],1,toplot[0].size()[1],toplot[0].size()[2])
            #print(x_grid.size())
            #plt.imshow(torchvision.utils.make_grid(x_grid,nrow=16).detach().numpy()[0],cmap = 'gray')
            #plt.colorbar(orientation='horizontal',pad=0.01,fraction=0.064)
            #plt.show()


        weight = net.conv1.weight.data.numpy()



    print('Start testing overall')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

    print('Start testing per class')
     #initialize confusion matrix
    confusion = np.zeros((26, 26))

    class_correct = list(0. for i in range(N_classes))
    class_total = list(0. for i in range(N_classes))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

         #for each image in batch, add entry to confusion matrix keeping track of the actual label
        # for each prediction
            for i in range(4):
                label = labels[i]
                prediction = predicted[i]
                confusion[label,prediction] = confusion[label,prediction]+1
                class_correct[label] += c[i].item()
                class_total[label] += 1



    for i in range(N_classes):
        if i == 9 or i == 25:   # skip J and Z since they are not included
            pass
        else:
            for k in range(N_classes):
                if k == 9 or k == 25:   # skip J and Z since they are not included
                    pass
                else:
                    #normalize the confusion matrix based on the number of instances for each class
                    confusion[i,k] = confusion[i,k]/class_total[i]*100
            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    #save the confusion matrix to a seperate file
    np.save( 'confusion_matrix', confusion)
