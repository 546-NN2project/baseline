import sys
import numpy as np
import operator
import json
import scipy.io as sio
import time

def readGloveData(glove_word_vec_file):
    f = open(glove_word_vec_file, 'r')
    rawData = f.readlines()
    word_vec_dict = {}
    for line in rawData:
        line = line.strip().split()
        tag = line[0]
        vec = line[1:]
        word_vec_dict[tag] = np.array(vec, dtype=float)
    return word_vec_dict

def getWordVector(word, word_vec_dict):
	if word in word_vec_dict:
		return word_vec_dict[word]
	return np.zeros_like(word_vec_dict['hi'])



def main():
    glove_word_vec_file = '../../glove.6B/glove.6B.50d.txt'
    word_vec_dict = readGloveData(glove_word_vec_file)
    t = time.time()
    print getWordVector('hello', word_vec_dict)
    elapsed = time.time() - t
    print elapsed
if __name__ == "__main__":
    main()
