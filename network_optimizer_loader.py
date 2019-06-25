#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% include all libraries
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image

#%% define parameters
test_path = 'SIGN/sign_mnist_test.csv'  # path to test csv

# init number of filters
fs = [[6,12,18,24],[12,24,36,48],[24,48,72,96],[48,96,144,192]]
fc1 = 120
fc2 = 80

# only for print feedback
Lr = [0.01, 0.005, 0.001]
Momentum = [0.8, 0.85, 0.9, 0.95]

nclasses = 26   # number of classes
batch = 8       # batch size

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
testloader = torch.utils.data.DataLoader(testset, batch_size=batch, shuffle=False, num_workers=1)

#%% define classes
classes = ('A', 'B', 'C', 'D','E', 'F', 'G', 'H','I', 'J', 'K', 'L', 'M', 'N', 'O', 'P','Q', 'R', 'S', 'T','U', 'V', 'W', 'X','Y', 'Z')
nclasses = 26

#%% define neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, fs1, 3, padding=1)
        self.conv2 = nn.Conv2d(fs1, fs2, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(fs2, fs3, 3, padding=1)
        self.conv4 = nn.Conv2d(fs3, fs4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(fs4 * 7 * 7, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, nclasses)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, fs4 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
#%% testing network
if __name__ == '__main__':
    nnetwork = 4 * 3 * 4        # total number of networks to loop over
    for l in range(len(Momentum)):  # loop over Momentum
        for k in range(len(Lr)):  # loop over Lr
            for j in range(len(fs)):  # loop over filters
                cnetwork = (l+1) * (k+1) * (j+1)    # current network
                
                # define network parameters
                fs1 = fs[j][0]
                fs2 = fs[j][1]
                fs3 = fs[j][2]
                fs4 = fs[j][3]
                
                # define where weights and loss files need to be loaded
                param_path = 'weights/weightfs%slr%smo%sextended.pth' % (j,k,l)
                result_path = 'results/resultfs%slr%smo%sextended.txt' % (j,k,l)
                
                # check for graphics card availability
                use_gpu = torch.cuda.is_available()
                net = Net()
                if use_gpu:
                    net = net.cuda()
                    device = torch.device("cuda")
                net.load_state_dict(torch.load(param_path))
            #    net.load_state_dict(torch.load(param_path,map_location='cpu'))
                net.eval()

                # start testing for overall accuracy
                print('Start testing for (',cnetwork,'/',nnetwork,') filterset:',j,'learning rate:',Lr[k], 'momentum:',Momentum[l])
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in testloader:
                        images, labels = data
                        outputs = net(images.to(device))
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels.to(device)).sum().item()
                result_file = open(result_path,"a")
                result_file.write(repr(100 * correct / total)+ '\n')
                result_file.close()
                print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

                # start testing for class accuracy
                print('Start testing per class')
                class_correct = list(0. for i in range(nclasses))
                class_total = list(0. for i in range(nclasses))
                with torch.no_grad():
                    for data in testloader:
                        images, labels = data
                        outputs = net(images.to(device))
                        _, predicted = torch.max(outputs, 1)
                        c = (predicted == labels.to(device)).squeeze()
                        for i in range(4):
                            label = labels[i]
                            class_correct[label] += c[i].item()
                            class_total[label] += 1
                
                for i in range(nclasses):
                    if i == 9 or i == 25:   # skip J and Z since they are not included
                        pass
                    else:
                        result_file = open(result_path,"a")
                        result_file.write(repr(100 * class_correct[i] / class_total[i])+ '\n')
                        result_file.close()
                        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
