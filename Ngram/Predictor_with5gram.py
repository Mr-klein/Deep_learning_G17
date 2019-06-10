

# --- PREDICT NEXT LETTER ----
# Author : Keith Klein

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import torchviz as tv
from PIL import Image
first_letters = 'MAT'
print(first_letters)







#load Ngrams
bigrams = np.load('savedmodel/bigrams.npy')
trigrams = np.load('savedmodel/trigrams.npy')
unigrams = np.load('savedmodel/unigrams.npy')
quadgrams = np.load('savedmodel/quadgrams.npy')
pentagrams = np.load('savedmodel/pentagrams.npy')
while(True):

    plt.xticks(np.arange(0, 26), ['A', 'B', 'C', 'D',
               'E', 'F', 'G', 'H', 'I',
                'J','K','L','M','N','O',
                'P','Q','R','S','T','U','V','W','X','Y','Z'])
    plt.yticks(np.arange(0, 26), ['A', 'B', 'C', 'D',
               'E', 'F', 'G', 'H', 'I',
                'J','K','L','M','N','O',
                'P','Q','R','S','T','U','V','W','X','Y','Z'])

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.imshow(bigrams,cmap='PuBu',vmin=0., vmax=1.)
    plt.Normalize(vmin=0.,vmax=100.)
    plt.colorbar(pad=0.01).ax.set_title('prediction %')


    #plt.show()
    letters = ['A', 'B', 'C', 'D',
               'E', 'F', 'G', 'H', 'I',
                'J','K','L','M','N','O',
                'P','Q','R','S','T','U','V','W','X','Y','Z']


    def predict(first_letters,unigrams,bigrams,trigrams,quadgrams,pentagrams):
        sequence_length = len(first_letters)

        if sequence_length >4:
            start_i = sequence_length-4
        else:
            start_i = 0


        indeces = []
        for i in range(start_i,sequence_length):
            #Find index of first letters in word to access Ngram
            indeces.append(letters.index(first_letters[i]))
        print(indeces)

        if sequence_length == 0:
            Ngram = unigrams
            possible_predictions = Ngram
            prediction = np.argmax(possible_predictions)
        elif sequence_length == 1:
            Ngram = bigrams
            possible_predictions = Ngram[indeces[0]]
            prediction = np.argmax(possible_predictions)
        elif sequence_length == 2:
            Ngram = trigrams
            possible_predictions = Ngram[indeces[0],indeces[1]]
            prediction = np.argmax(possible_predictions)
        elif sequence_length == 3:
            Ngram = quadgrams
            possible_predictions = Ngram[indeces[0],indeces[1],indeces[2]]
            prediction = np.argmax(possible_predictions)
        elif sequence_length >= 4:
            Ngram = pentagrams
            possible_predictions = Ngram[indeces[0],indeces[1],indeces[2], indeces[3]]
            prediction = np.argmax(possible_predictions)
        return prediction

    a= predict(first_letters,unigrams,bigrams,trigrams,quadgrams,pentagrams)
    a= np.asscalar(np.asarray(a))

    predicted_letter = letters[a]
    #print(predicted_letter)
    first_letters = first_letters + predicted_letter
    print(first_letters)
    time.sleep(1)
