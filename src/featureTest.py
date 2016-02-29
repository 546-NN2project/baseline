import numpy as np
import wordToVecConvertor as word2vec
import time

'''
test file that demonstrates how to create the word2vec dectionary
and use that dictionary to generate word embeddings for the training data.

load the word2vec dictionary and test a simple word to get it vector 
'''
#wordToVecDictFile = '../../glove.6B/glove.6B.50d.txt'
#wordVecDict = word2vec.readDictData(wordToVecDictFile)
#print word2vec.getWordVector('what', wordVecDict)

'''
read the training data file, load word2vec dictionary are return vector
embeddings + 
'''
trainingDataFile = '../data/traindata.txt'
trainingLabelFile = '../data/trainlabel.txt'
wordToVecDictFile = '../../glove.6B/glove.6B.50d.txt'
ctxWin = 2 #context window, 2 words before and 2 after (total 5)
X,Y = word2vec.createVecFeatsLabels(trainingDataFile,trainingLabelFile,wordToVecDictFile,ctxWin)
print 'we are done'
np.savetxt('features_X_2.txt', X)
np.savetxt('labels_Y_2.txt', Y)