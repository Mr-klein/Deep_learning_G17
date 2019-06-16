#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Loads the classifier and uses the words from the 'google10000' dataset
# to feed sequences of letters into the classifier
# the Ngram probabilities are combined with the classifier probabilities
# in order to create a final estimate of the sign letter

# accuracy is calculated on the word-level
# eg. if a letter in a word is wrong, the entire word is wrong

# parameters to be adjusted:
# minimum length: the minimum length of words to use from the text data set
# prediction weight: how much weight the Ngram prediction has on the overall estimate (Ngram has 0 weight for initial estimate)

import time
from tqdm import tqdm
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
param_path = 'weights/weightfs3lr1mo3.pth'     # path to model param
test_path = 'SIGN/sign_mnist_test.csv'  # path to test csv

N_classes = 26          # number of classes
batch = 16               # batch size

#probability parameters
Use_Ngram = True
minimum_length = 0    #minimum length of words to use


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


#load Ngrams
bigrams = np.load('savedNgram/bigrams.npy')
trigrams = np.load('savedNgram/trigrams.npy')
unigrams = np.load('savedNgram/unigrams.npy')
quadgrams = np.load('savedNgram/quadgrams.npy')

#load confusion_matrix
confusion = np.load('confusion_matrix.npy')
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

    # load network
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
    #%% testing network

    # initializing performance parameters
    total_words = 0
    words_correct = 0
    letters_correct = 0
    letters_total = 0

    # loop over all words in the test dataset
    print('testing network on sequences ... ')
    with tqdm(total = len(wordlist)) as pbar:
        for k,word in enumerate(wordlist,0):
            if len(word)<minimum_length:
                continue
            #update count with total number of words
            total_words += 1
            # initialize incorrect translation flag
            word_wrong = False
            # initialize list with current translated letters
            letters_so_far = []

            #some parameters for probabilities
            tree1 = []
            tree2 = []
            double_counter = 0

            #determine whether its possible to predict two letters
            for j,letter in enumerate(word,0):
                letters_total += 1
                if len(word)-j<2 and double_counter == 0:
                    double_prediction = False
                else:
                    double_prediction = True

                #double_prediction = False



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

                                _, classified = torch.max(outputs.data, 1)

                                #get probability distribution from confusion matrix
                                probabilities = confusion[:,classified]/100
                                #probabilitiy of certain class
                                P_evidence = predict([],unigrams,bigrams,trigrams,quadgrams,classes)

                                if double_prediction:
                                    if (j+1)%2 != 0:
                                        double_counter = 1
                                        tree1 = np.multiply(probabilities,predict(letters_so_far,unigrams,bigrams,trigrams,quadgrams,classes))
                                    elif (j+1)%2 == 0:
                                        double_counter = 0
                                        tree2=[]
                                        for a,letter in enumerate(classes,0):
                                            prob_predict = np.asarray(predict(letters_so_far+[letter],unigrams,bigrams,trigrams,quadgrams,classes))
                                            total_prob = tree1[a]*np.multiply(probabilities,prob_predict)
                                            for probability in total_prob:
                                                tree2.append(probability)

                                        tree2_index = np.argmax(tree2)
                                        #print(tree2_index)
                                        letter1 = int(tree2_index/26)
                                        #print(letter1)
                                        #print(len(tree2))
                                        letter2 = tree2_index%26
                                        #print(letters_so_far)
                                        letters_so_far.append(classes[letter1])
                                        letters_so_far.append(classes[letter2])

                                # translate single letter
                                if double_prediction == False:
                                    #plt.plot(probabilities)
                                    final_prob = np.multiply(probabilities,predict(letters_so_far,unigrams,bigrams,trigrams,quadgrams,classes))
                                    #print(final_prob)
                                    #time.sleep(1)
                                    #final_prob = weight * np.asarray(predict(letters_so_far,unigrams,bigrams,trigrams,quadgrams,classes)) + (1-weight) * probabilities
                                    final_prob_index = np.argmax(final_prob)
                                    if Use_Ngram == False:
                                        final_prob_index = classified
                                    letters_so_far.append(classes[final_prob_index])
                                    #plt.plot(final_prob)
                                    #plt.show()


                                #check if word has been correctly translated
                                if j == len(word)-1:
                                    #print(letters_so_far)
                                    for b,predicted_letter in enumerate(letters_so_far,0):
                                        if predicted_letter != word[b]:
                                            word_wrong = True
                                        elif predicted_letter == word[b]:
                                            letters_correct += 1

                                labelsearch = False
                                break


            if word_wrong == False:
                words_correct += 1
            pbar.update(1)
    print('Done')
    print('The accuracy of the network on sequences on the word level is ' + str((words_correct/total_words)*100))

    print('The accuracy of the network on sequences on the letter level is ' + str((letters_correct/letters_total)*100))
