#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:46:58 2019

@author: RJ
"""
import random
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from Predictor import predict
import torch.nn.functional as F
import pandas as pd
from PIL import Image

#%% define parameters
param_path = 'model_weights_google.pth'     # path to model param
test_path = 'SIGN/sign_mnist_test.csv'  # path to test csv

N_classes = 26          # number of classes
batch = 16               # batch size

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
                                         shuffle=False, num_workers=1)

#%% define classes
classes = ('a', 'b', 'c', 'd',
           'e', 'f', 'g', 'h',
           'i', 'j', 'k', 'l',
           'm', 'n', 'o', 'p',
           'q', 'r', 's', 't',
           'u', 'v', 'w', 'x',
           'y', 'z')

#%% define neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
        #        dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(1, 12, 3, padding=1)
        self.conv2 = nn.Conv2d(12, 24, 3, padding=1)
        # MaxPool2d(kernel_size, stride=None, padding=0, dilation=1,
        #           return_indices=False, ceil_mode=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(24,48, 3, padding=1)
        self.conv4 = nn.Conv2d(48,96, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(96 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, N_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 96 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#load Ngrams
bigrams = np.load('savedNgram/bigrams.npy')
trigrams = np.load('savedNgram/trigrams.npy')
unigrams = np.load('savedNgram/unigrams.npy')
quadgrams = np.load('savedNgram/quadgrams.npy')

#Load words to use for testing
wordlist = np.load('processed_data/testwords.npy')
images = []
labels = []
if __name__ == '__main__':
    #create list with images and labels for indexing
    for data in testloader:
        image, label = data
        images.append(image)
        labels.append(label)



    #%% testing network

    # initializing performance parameters
    total_words = 0
    words_correct = 0
    letters_correct = 0
    letters_total = 0

    # loop over all words in the test dataset
    print('testing network on sequences ... ')
    for k,word in enumerate(wordlist,0):
        #print(k)
        total_words += 1

        # initialize incorrect translation flag
        word_wrong = False
        letters_so_far = []
        for j,letter in enumerate(word,0):
            letter_wrong = False
            letters_total += 1

            # if the letter is not the first letter, use the predicted letter from NGRAM
            if j != 0:
                previous_letter_prediction = next_letter_prediction
            use_gpu = torch.cuda.is_available()
            net = Net()
            device = torch.device("cpu")
            if use_gpu:
                net = net.cuda()
                device = torch.device("cuda")
            #net.load_state_dict(torch.load(param_path))
            #load the CNN classifier weights
            net.load_state_dict(torch.load(param_path,map_location='cpu'))
            net.eval()



            #initialize label search flag (flag is true if program is looking for image with desired label)
            labelsearch = True
            with torch.no_grad():
                #Find random image corresponding to current letter
                while(labelsearch):
                    # select a random batch from the dataset
                    batchtouse = random.randint(0,448)
                    for i,label in enumerate(labels[batchtouse],0):

                        # find desired image in chosen batch
                        if label == classes.index(letter):
                            image = images[batchtouse][i]
                            image = image.view(1,1,28,28)
                            outputs = net(image.to(device))
                            A = outputs.data.numpy()

                            #obtain probabilities from output
                            sm = torch.nn.Softmax(dim=1)
                            probabilities = sm(outputs)
                            #print(probabilities)
                            probabilities = probabilities.squeeze()

                            #outputs = net(images)

                            _, predicted = torch.max(outputs.data, 1)
                            # if it's the first letter, only rely on the classifier
                            if j == 1:
                                weight = 0
                            if j == 2:
                                weight = 0.2
                            if j == 3:
                                weight = 0.5

                            if j == 0:
                                weight = 0
                                previous_letter_prediction = probabilities

                            #weight = 0
                            # predict the next letter based on current classification
                            next_letter_prediction = predict(letters_so_far,unigrams,bigrams,trigrams,quadgrams,classes)
                            ## use a weighted some of the probability distribution from the classifier and the predictor
                            #print(np.asarray(previous_letter_prediction))
                            #predicted = np.argmax(weight * np.asarray(previous_letter_prediction) + (1-weight) * probabilities.numpy())

                            #when a letter is wrong, the entire word is wrong
                            if predicted != label:
                        #        print(probabilities)
                        #        plt.plot(probabilities.numpy())
                        #        plt.show()
                                word_wrong = True
                                letter_wrong = True
                            labelsearch = False
                            break

            if letter_wrong == False:
                letters_correct += 1
        if word_wrong == False:
            words_correct += 1

    print('Done')
    print('The accuracy of the network on sequences is ' + str((words_correct/total_words)*100))
