# Loading data, training model, validation, testing functions
#

import os
import sys
import timeit

import numpy as np
import wordToVecConvertor as word2vec
import time
from sklearn.cross_validation import train_test_split

try:
   import cPickle as pickle
except:
   import pickle

import theano
import theano.tensor as T

def load_data():
    '''
    Loads the data, turns into word2vec representation, and splits
    into training, validation, and testing sets with ratio 8:1:1
    '''
    trainingDataFile = '../data/traindata.txt'
    trainingLabelFile = '../data/trainlabel.txt'
    wordToVecDictFile = '../data/glove/glove.6B.50d.txt'
    print('Vectorizing the features and labels...')
    start_time = timeit.default_timer()
    X,Y = word2vec.createVecFeatsLabels(trainingDataFile,trainingLabelFile,wordToVecDictFile,window_size)
    end_time = timeit.default_timer()
    print('Pickling the vectorization files')
    # pickling X-file
    clean_data = open('../data/clean_data.pkl','wb')
    pickle.dump(X, clean_data)
    clean_data.close()
    # pickling the labels-file
    clean_label = open('../data/clean_label.pkl', 'wb')
    pickle.dump(Y, clean_label)
    clean_label.close()
    print(('The vectorization ran for %.2fm' % ((end_time - start_time) / 60.)))
    print('Splitting into training, validation, and testing sets ...')
    X_train, X_rest, y_train, y_rest = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_rest,y_rest, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def splitting_data():
    X = pickle.load(open('../data/clean_data.pkl','rb'))
    Y = pickle.load(open('../data/clean_label.pkl','rb'))
    unit = len(Y)/10
    X_train, X_rest, y_train, y_rest = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_rest,y_rest, test_size=0.5, random_state=42)



def build_model():
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = FFNN(
        rng=rng,
        input=x,
        n_in=n_in,
        n_hidden=n_hidden,
        n_out=n_out
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

def train_model():
    print('... training')

    # early-stopping parameters
    patience = 10000  # look at this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                    # save the best model
                    #with open('best_model.pkl', 'wb') as f:
                    #    pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ran for %.2fm' % ((end_time - start_time) / 60.)))
    best_model = open('best_model.pkl','wb')
    pickle.dump(classifier, best_model)
    best_model.close()

if __name__ == '__main__':
    # word2vec dimension	
    word_vec_dim = 50
    # context window size, default of 5 means look at 5 words before and after the current word
    window_size = 5
    # the dimension of each feature vector
    n_in = (2*window_size + 1)*word_vec_dim
    # number of hidden nodes
    n_hidden = 300
    train_set_x, valid_set_x, test_set_x, train_set_y, valid_set_y, test_set_y = load_data()
    build_model()
    train_model()
    
