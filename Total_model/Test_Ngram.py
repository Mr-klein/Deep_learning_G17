
# author: Keith
# tests the prediction accuracy of Ngrams given that the previous letters are correct

from Predictor import predict
import numpy as np

#load Ngrams
bigrams = np.load('savedNgram/bigrams.npy')
trigrams = np.load('savedNgram/trigrams.npy')
unigrams = np.load('savedNgram/unigrams.npy')
quadgrams = np.load('savedNgram/quadgrams.npy')

#Load words to use for testing
wordlist = np.load('processed_data/testwords.npy')
classes = ('a', 'b', 'c', 'd',
           'e', 'f', 'g', 'h',
           'i', 'j', 'k', 'l',
           'm', 'n', 'o', 'p',
           'q', 'r', 's', 't',
           'u', 'v', 'w', 'x',
           'y', 'z')
total_bigrams = 0
total_trigrams = 0
total_quadgrams = 0
total_bigrams_correct = 0
total_trigrams_correct = 0
total_quadgrams_correct = 0
for k,word in enumerate(wordlist,0):
    letters_so_far = []
    for i,letter in enumerate(word,0):
        if i==0:
            current_prediction = letter
        else:
            current_prediction = next_prediction
        letters_so_far.append(letter)
        next_prediction_label = np.argmax(predict(letters_so_far,unigrams,bigrams,trigrams,quadgrams,classes))
        next_prediction = classes[next_prediction_label]



        if i == 1:
            if current_prediction == letter:
                total_bigrams_correct += 1
            total_bigrams += 1
        if i == 2:
            if current_prediction == letter:
                total_trigrams_correct += 1
            total_trigrams +=1
        if i >= 3:
            if current_prediction == letter:
                total_quadgrams_correct +=1
            total_quadgrams +=1


print("bigram prediction accuracy: "+str(total_bigrams_correct/total_bigrams*100)+"%")
print("trigram prediction accuracy: "+str(total_trigrams_correct/total_trigrams*100)+"%")
print("quadgram prediction accuracy: "+str(total_quadgrams_correct/total_quadgrams*100)+"%")
