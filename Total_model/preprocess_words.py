import numpy as np


# remove words with letters j and z and shorten list
alphawords = 'words_alpha.txt'
words = [line.rstrip('\n') for line in open(alphawords)]
i = 0
while i < len(words):
    word = words[i]
    if 'j' in word or 'z' in word:
        del words[i]
        i -=1
    i += 1

i = 0
while i < len(words):

    del words[i]
    i += 2
i = 0
while i < len(words):

    del words[i]
    i += 2
i = 0
while i < len(words):

    del words[i]
    i += 2
i = 0
while i < len(words):

    del words[i]
    i += 2

i = 0
while i < len(words):

    del words[i]
    i += 2

i = 0
while i < len(words):

    del words[i]
    i += 2

i = 0
while i < len(words):

    del words[i]
    i += 2

i = 0
while i < len(words):

    del words[i]
    i += 2
i = 0
while i < len(words):

    del words[i]
    i += 2

i = 0
while i < len(words):

    del words[i]
    i += 2

i = 0
while i < len(words):

    del words[i]
    i += 3




print(len(words))
np.save( 'testwords', words)
