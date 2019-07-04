

# --- PREDICT NEXT LETTER based on ngram ----
# Author : Keith Klein
# requires calculated Ngrams bigrams to count_tetragrams
# ngrams trained using NgramModelTrainer.py
# use input first letters in variable below, code will predict next letters

# This code is used to test the Ngram predictions


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
import sys


first_letters = 'AN'


#load Ngrams
bigrams = np.load('../savedNgram/bigrams.npy')
trigrams = np.load('../savedNgram/trigrams.npy')
unigrams = np.load('../savedNgram/unigrams.npy')
quadgrams = np.load('../savedNgram/quadgrams.npy')

#
# plt.xticks(np.arange(0, 26), ['A', 'B', 'C', 'D',
#            'E', 'F', 'G', 'H', 'I',
#             'J','K','L','M','N','O',
#             'P','Q','R','S','T','U','V','W','X','Y','Z'])
# plt.yticks(np.arange(0, 26), ['A', 'B', 'C', 'D',
#            'E', 'F', 'G', 'H', 'I',
#             'J','K','L','M','N','O',
#             'P','Q','R','S','T','U','V','W','X','Y','Z'])
#
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.imshow(bigrams,cmap='PuBu',vmin=0., vmax=1.)
# plt.Normalize(vmin=0.,vmax=100.)
# plt.colorbar(pad=0.01).ax.set_title('prediction %')


#plt.show()
letters = ['A', 'B', 'C', 'D',
           'E', 'F', 'G', 'H', 'I',
            'J','K','L','M','N','O',
            'P','Q','R','S','T','U','V','W','X','Y','Z']


def predict(first_letters,unigrams,bigrams,trigrams,quadgrams,letters):
    sequence_length = len(first_letters)

    if sequence_length >3:
        start_i = sequence_length-3
    else:
        start_i = 0


    indeces = []
    for i in range(start_i,sequence_length):
        indeces.append(letters.index(first_letters[i]))

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
    elif sequence_length >= 3:
        Ngram = quadgrams
        possible_predictions = Ngram[indeces[0],indeces[1],indeces[2]]
        prediction = np.argmax(possible_predictions)
    return possible_predictions


#a= predict(first_letters,unigrams,bigrams,trigrams,quadgrams,letters)
#plt.plot(a)
#plt.title('Trigram predition example for sequence "AN"')
#plt.ylabel('Probability')
#plt.xlabel('Letter')
#plt.xticks(np.arange(0, 26), ['A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'I',
             'J','K','L','M','N','O',
             'P','Q','R','S','T','U','V','W','X','Y','Z'])
#plt.show()
#a= np.asscalar(np.asarray(a))

#predicted_letter = letters[a]
#print(predicted_letter)
