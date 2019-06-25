#processes the text file such that each word is on a seperate line
#characters our of alphabet are also removed


corpuspath = 'data/brown.txt'
import numpy as np
seperator = ''
allowed = ['a', 'b', 'c', 'd',
           'e', 'f', 'g', 'h',
           'i', 'j', 'k', 'l',
           'm', 'n', 'o', 'p',
           'q', 'r', 's', 't',
           'u', 'v', 'w', 'x',
           'y', 'z','A', 'B',
           'C','D', 'E', 'F',
           'G', 'H', 'I','J',
           'K','L','M','N','O',
           'P','Q','R','S','T',
           'U','V','W','X','Y','Z']
corpuslist = []
""
print("processing corpus...")
with open('corpuspath','r') as f:
    for line in f:
        for word in line.split():
            wordnew = []
            wordnewstring = ""
            for i,letter in enumerate(word,0):
                prohibited_symbol = True
                for symbol in allowed:
                    if symbol == letter:
                        prohibited_symbol = False
                if prohibited_symbol == True:
                    pass
                else:
                    wordnew.append(letter)
            corpuslist.append(wordnewstring.join(wordnew))
corpuslist_lower = [word.lower() for word in corpuslist]
processed_corpus = open("processed_data/processed_corpus.txt", "w")
for line in corpuslist_lower:
    if len(line) > 0:
        processed_corpus.write(line)
        processed_corpus.write("\n")
processed_corpus.close()
print("done")
