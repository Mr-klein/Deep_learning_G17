
#preprocesses the data-set used to test the model
import numpy as np


# remove words with letters j and z 
alphawords = '../data/google10000.txt'
words = [line.rstrip('\n') for line in open(alphawords)]
i = 0
while i < len(words):
    word = words[i]
    if 'j' in word or 'z' in word:
        del words[i]
        i -=1
    i += 1



print(len(words))
np.save( '../processed_data/testwords', words)
