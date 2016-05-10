# -*- coding: utf-8 -*-
"""
Created on Fri May  6 18:04:31 2016

@author: ananddeshmukh
"""

# module to train the relations model
import os
import sys
import timeit
sys.dont_write_bytecode = True

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
from FFNN import *
#import evaluate 
import FeatureProcessing
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import itertools

# load the data
def loadData(pctNull):
    '''
    Loads the data, turns into word2vec representation, and splits
    into training, validation, and testing sets with ratio 8:1:1
    '''
    wordToVecDictFile = '../../data/glove.6B/glove.6B.50d.txt'
    relfile = '../data/mention.pkl'
    print relfile
    wvecdim = 50
    print('Vectorizing the features and labels...')
    start_time = timeit.default_timer()
    X,Y = FeatureProcessing.featureProcessRel(relfile,wordToVecDictFile, wvecdim,pctNull)
    end_time = timeit.default_timer()
    
    print("the size of the dataset and the label sets are %d and %d") %(len(X),len(Y))
    print(('The vectorization ran for %.2fm' % ((end_time - start_time) / 60.)))
    print('Splitting into training, validation, and testing sets ...')
    X_train, X_rest, y_train, y_rest = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_rest,y_rest, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# build and train the FFNN (multi layer perceptron) model 
def buildAndTrain(optimizing_function,n_labels):
    print('building the model ... ')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = np.random.RandomState(1234)

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
    test_model_accuracy = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size],
        y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    test_model = theano.function(
    inputs=[index],
    outputs=[y, classifier.y_pred],
    givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size],
        y: test_set_y[index * batch_size: (index + 1) * batch_size]
    }
)
    validate_model_accuracy = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
        x: valid_set_x[index * batch_size: (index + 1) * batch_size],
        y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    validate_model = theano.function(
    inputs=[index],
    outputs=[y, classifier.y_pred],
    givens={
        x: valid_set_x[index * batch_size: (index + 1) * batch_size],
        y: valid_set_y[index * batch_size: (index + 1) * batch_size]
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
    print('training ... ')

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

    best_validation_loss = np.inf
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

                # Otimizing for accuracy
                validation_loss = [validate_model_accuracy(i) for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_loss)

                # Optimizing for precision
                validation_precision = [np.mean(precision_score(*validate_model(i),labels=range(n_labels),pos_label=None,average=None)) for i in range(n_valid_batches)]
                this_validation_precision = np.mean(validation_precision) # average precision for all the labels

                # Optimizing for recall
                validation_recall = [np.mean(recall_score(*validate_model(i),labels=range(n_labels),pos_label=None,average=None)) for i in range(n_valid_batches)]
                this_validation_recall = np.mean(validation_recall) # average recall for all the labels

                # Optimizing for f-score
                validation_fscore = [np.mean(f1_score(*validate_model(i),labels=range(n_labels),pos_label=None,average=None)) for i in range(n_valid_batches)]
                this_validation_fscore = np.mean(validation_fscore) # average f-score for all the labels



                print(
                    'epoch %i, minibatch %i/%i, average validation precision %f, validation recall %f, validation fscore %f, and loss %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_precision * 100,
                        this_validation_recall * 100,
                        this_validation_fscore * 100,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [np.mean(optimizing_function(*test_model(i),pos_label=None,average=None)) for i
                                   in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                    # save the best model
                    #with open('best_rel_model.pkl', 'wb') as f:
                    #    pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    #valid_predic = [validate_model(i) for i in range(n_valid_batches)]
    valid_prec = [precision_score(*validate_model(i),labels=range(n_labels),pos_label=None,average=None) for i in range(n_valid_batches)]
    valid_rec = [recall_score(*validate_model(i),labels=range(n_labels),pos_label=None,average=None) for i in range(n_valid_batches)]
    valid_fsc = [f1_score(*validate_model(i),labels=range(n_labels),pos_label=None,average=None) for i in range(n_valid_batches)]
  
    # correct the label scores for batches which dont have cetrain labels at all in the true labels    
#    for batch in range(n_valid_batches):
#        if len(valid_prec[batch]) < n_labels:
#            valid_prec[batch] = np.append(valid_prec[batch],np.mean(valid_prec[batch]))
#        if len(valid_rec[batch]) < n_labels:
#            valid_rec[batch] =  np.append(valid_rec[batch],np.mean(valid_rec[batch]))
#        if len(valid_fsc[batch]) < n_labels:
#            valid_fsc[batch] = np.append(valid_fsc[batch],np.mean(valid_fsc[batch]))
#    
    validation_precision = np.mean(valid_prec,axis=0)
    validation_recall = np.mean(valid_rec,axis=0)
    validation_fscore = np.mean(valid_fsc,axis=0) 
    
    end_time = timeit.default_timer()
    print('----------------------------------------------------------')
    print('Optimization complete. Saving the best model.')
#    with open('best_relations_model.pkl', 'wb') as f:
#        pickle.dump(classifier, f)    
    #print('saved the best relations model')
    
    #print(('Optimization complete. Best validation loss of %f %% '
    #       'obtained at iteration %i, with test performance %f %%') %
    #      (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('Best validation loss: %f %%' %(best_validation_loss*100))      
    print('Optimal results corresponding to the best validation loss:')
    print('Precision: (average) -- %f %%' %(this_validation_precision*100))
    print(validation_precision)
    print('Recall: (average) -- %f %%' %(this_validation_recall*100))
    print(validation_recall)
    print('F-score: (average) -- %f %%' %(this_validation_fscore*100))
    print(validation_fscore)
    print('----------------------------------------------------------')
    print(('The code for file ran for %.2fm' % ((end_time - start_time) / 60.)))
    #best_model_weights = classifier.save_weights()
    #pickle.dump(classifier, best_model)
    #best_model.close()
    #return classifier
    #predict_model = theano.function(
    #    inputs=[classifier.input],
    #    outputs=classifier.y_pred
    #    )
    #predicted_values = predict_model(test_set_x.eval()[:20])
    #print("Predicted values for the first 20 examples in test set and true values:")
    #for pred,true in itertools.izip_longest(predicted_values, test_set_y[:20]): print pred,true


def sharedDataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

if __name__ == '__main__':
    print "training the relations extraction model ..."
    n_in=154
    n_out = 7
    window_size=0
    word_vec_dim=50
    n_hidden=400
    learning_rate=0.001
    L1_reg=0.00
    L2_reg=0.0001
    n_epochs=10000
    batch_size=20
    n_labels=n_out
    pctNull = 0.2 #percentage of null relations to be randomly included in training

    train_set_x, valid_set_x, test_set_x, train_set_y, valid_set_y, test_set_y = loadData(pctNull)
    train_set_x, train_set_y = sharedDataset((train_set_x,train_set_y),borrow=True)
    valid_set_x, valid_set_y = sharedDataset((valid_set_x,valid_set_y),borrow=True)
    test_set_x, test_set_y = sharedDataset((test_set_x,test_set_y),borrow=True)

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    buildAndTrain(precision_score,n_labels)
    #classifier = build_and_train(precision_score)
    #predict(classifier)
