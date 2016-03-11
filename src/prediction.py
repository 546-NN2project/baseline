## prediction function for the NN based modelimport os
import sys
import timeit

import numpy as np
import wordToVecConvertor as word2vec
import time

try:
   import cPickle as pickle
except:
   import pickle

import theano
import theano.tensor as T
from FFNN import *

# function for the prediction using NN model
def predict(classifier, test_x):
    # assuming that classifier object is alreay instantiated with optimal params
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.y_pred(inputs),
        givens={
            x: test_x,
         }
    )
    y_predict = np.zeros([n_out,len(test_x)])
    y_predict[i] = [test_model(i) for i in range(test_x)]
    return y_predict
    
# function to get the     
def dummifier(vector):
    return list(vector).index(1)
    
if __name__ == '__main__':
    n_pos=45 # dimension of POS embeddings
    n_in=551 + n_pos
    n_out = 27
    
    # assuming that the input data is already vectorized word embeddings
    Xt = pickle.load(open('../data/test_data.pkl','rb'))
    Yt = pickle.load(open('../data/test_label.pkl','rb'))
    params = pickle.load(open('../data/optim_params.pkl','rb'))
    
    classifier = FFNN(
        input=params,
        n_in=n_in,
        n_hidden=n_hidden,
        n_out=n_out
    )
    
    Yt_hat = predict(classifier, Xt, paramas)
    

    