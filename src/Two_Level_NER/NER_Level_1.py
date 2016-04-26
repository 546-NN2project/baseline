# 1. train and save a binary model
# 2. training data -> binary model -> non-O enter multiclass model

import os
import sys
import timeit

import numpy as np
#import wordToVecConvertor as word2vec
import time
from sklearn.cross_validation import train_test_split

try:
   import cPickle as pickle
except:
   import pickle as pickle

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
import random

def sample(X,Y):
   """
   sample part of the 'O' labels for the
   training of second-level NER
   Input: feature list: X, label list: Y
   Output: sampled feautre list sampledX, sampled label list sampledY
   """
   sampledX = []
   sampledY = []
   nullPos = []

   # get all indices of 'O'-labeled data
   nonNullNum = 0
   for pos in range(len(Y)):
      label = Y[pos]
      if (getLabelIndex('O') == label):
         nullPos.append(pos)
      else:
         nonNullNum += 1
    
   # randomly sample part of 'O'-labeled data
   num = nonNullNum / 26 + 1
   sampledNullPos = random.sample(nullPos, num)

   for pos in range(len(Y)):
      if ((pos in nullPos) and (pos not in sampledNullPos)):
         continue
      else:
         feature = X[pos]
         label = Y[pos]
         sampledX.append(feature)
         sampledY.append(label)
   print sampledY
   return sampledX, sampledY


def build_and_train(optimizing_function):
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
                validation_precision = [np.mean(precision_score(*validate_model(i),pos_label=None,average=None)[0:26]) for i in range(n_valid_batches)]
                this_validation_precision = np.mean(validation_precision)

                # Optimizing for recall
                validation_recall = [np.mean(recall_score(*validate_model(i),pos_label=None,average=None)[0:26]) for i in range(n_valid_batches)]
                this_validation_recall = np.mean(validation_recall)

                # Optimizing for f-score
                validation_fscore = [np.mean(f1_score(*validate_model(i),pos_label=None,average=None)[0:26]) for i in range(n_valid_batches)]
                this_validation_fscore = np.mean(validation_fscore)



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
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [np.mean(optimizing_function(*test_model(i),pos_label=None,average=None)[0:26]) for i
                                   in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                    return classifier
                    # save the best model
                    #with open('binary_NER_model.pkl', 'wb') as f:
                    #   pickle.dump(classifier, f)
                    

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
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

def load_binary_data():
    """
    Load the data, get features as X, labels transformed to binary label as Y,
    split into train, valid and test set with ratio 8:1:1
    sampling has not been used 
    """
    trainingDataFile = '../data/traindata_test.txt'
    trainingLabelFile = '../data/trainlabel_test.txt'
    binaryLabelFile = '../data/binaryLabel.txt'
    wordToVecDictFile = '../data/glove/glove.6B.50d.txt'
    print('Vectorizing the features and labels...')
    start_time = timeit.default_timer()

    # vectorize the training data
    X, Y = FeatureProcessing.binaryFeatureProcess(trainingDataFile,trainingLabelFile,wordToVecDictFile,window_size)
    end_time = timeit.default_timer()
    print(('The vectorization ran for %.2fm' % ((end_time - start_time) / 60.)))

    # split the dataset
    X_train, X_rest, y_train, y_rest = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_rest,y_rest, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_data():
    '''
    Loads the data, turns into word2vec representation, and splits
    into training, validation, and testing sets with ratio 8:1:1
    '''
    trainingDataFile = '../data/traindata_test.txt'
    trainingLabelFile = '../data/trainlabel_test.txt'
    wordToVecDictFile = '../data/glove/glove.6B.50d.txt'
    print('Vectorizing the features and labels...')
    start_time = timeit.default_timer()
    X, Y = FeatureProcessing.featureProcess(trainingDataFile,trainingLabelFile,wordToVecDictFile,window_size)
    sampledX, sampledY = sample(X,Y)
    end_time = timeit.default_timer()
    print(('The vectorization ran for %.2fm' % ((end_time - start_time) / 60.)))
    X_train, X_rest, y_train, y_rest = train_test_split(sampledX, sampledY, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_rest,y_rest, test_size=0.5, random_state=42)
    print y_train
    print y_val
    print y_test
    return X_train, X_val, X_test, y_train, y_val, y_test

def getLabelIndex(label):
    labList = ['B-Contact-Info','B-Crime','B-FAC','B-GPE','B-Job-Title','B-LOC','B-Numeric','B-ORG','B-PER','B-Sentence','B-TIME','B-VEH','B-WEA','I-Contact-Info','I-Crime','I-FAC','I-GPE','I-Job-Title','I-LOC','I-Numeric','I-ORG','I-PER','I-Sentence','I-TIME','I-VEH','I-WEA','O']
    return labList.index(label)

def shared_dataset(data_xy, borrow=True):
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

def predictBinaryLabel(classifier, xData):
    """
    Given test feature (including the gold previous label)
    make predictions about label
    """
    predictLabel = []
    prevLabel = 0
    predict_model = theano.function(
        inputs = [classifier.input],
        outputs=classifier.y_pred
        )

    for i in range(len(xData)):
       feature = xData[i]
       #feature.pop()
       feature = feature + [prevLabel]
       feature = [feature]
       feature = np.asarray(feature,dtype=theano.config.floatX)
       label = predict_model(feature)[0]
       predictLabel.append(label)
       prevLabel = label
    return predictLabel
    
def predictLabelTwo(classifier, xData):
   """
   Given model, predict the label from 27 candidates
   """
   predictLabel = [] # contain label ids
   prevLabel = [0]*27
   prevLabel[getLabelIndex('O')] = 1
    
   predict_model = theano.function(
      inputs = [classifier.input],
      outputs=classifier.y_pred
      )
   

   for i in range(len(xData)):
       feature = xData[i]
       #del feature[-27:]
       feature = feature + prevLabel
       feature = [feature]
       feature = np.asarray(feature,dtype=theano.config.floatX)
       label = predict_model(feature)[0]
       print "predicted label is:", label
       predictLabel.append(label)
       prevLabel = [0]*27
       prevLabel[label-1] = 1
   return predictLabel

def predictLabel(c1, c2, xData):
   """
   Input: two classifiers: c1 and c2, data features
   Output: label assigned to each data item
   """
   predOne = predictBinaryLabel(c1,xData)
   predTwo = predictLabelTwo(c2, xData)
   pred = []
   n = len(predOne)
   nullIndex = getLabelIndex('O')
   for i in range(n):
      p1 = predOne[i]
      p2 = predTwo[i]
      if ((p1 == 0) or (p2==nullIndex)):
         pred.append(nullIndex)
      else:
         pred.append(p2)
   return pred

if __name__=="__main__":
    """
    train the binary model for NER
    """
    n_in = 552 # 550+1+1
    n_out = 2
    window_size=5
    word_vec_dim=50
    n_hidden=300
    learning_rate=0.01
    L1_reg=0.00
    L2_reg=0.0001
    n_epochs=1000
    #batch_size=20
    batch_size = 1
    
    print("Running NER training -- Step 1: binary model training")
    
    train_set_x, valid_set_x, test_set_x, train_set_y, valid_set_y, test_set_y = load_binary_data()
    train_set_x, train_set_y = shared_dataset((train_set_x,train_set_y),borrow=True)
    valid_set_x, valid_set_y = shared_dataset((valid_set_x,valid_set_y),borrow=True)
    test_x  = []
    for i in range(len(test_set_x)):
       test_x.append(test_set_x[i][:-1])
    test_y = test_set_y
    test_set_x, test_set_y = shared_dataset((test_set_x,test_set_y),borrow=True)

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    # binary model
    binaryClassifier = build_and_train(precision_score)
    print "test_set_x shared variable:", test_set_x
    prediction = predictBinaryLabel(binaryClassifier, test_x)
    print "prediction is:", prediction
    print "truth is:", test_y


    """
      train multiclass classification model
    """
    n_in = 578 # 550+1+27
    n_out = 27
    print("Running NER training -- Step 2: Second-Layer Classification")

    """
    -- already sample data for the training of second-level model
   -- final prediction to merge the predictions of both classifiers
   To be filled:
   choose training data from the first layer prediction
   -- training based on binary output
   -- training based on gold data, sample some 'O'-labels
   -- predict: run two classifiers simultaneously. product of them is the prediction
   """

    
    train_set_x, valid_set_x, test_set_x, train_set_y, valid_set_y, test_set_y = load_data()
    train_set_x, train_set_y = shared_dataset((train_set_x,train_set_y),borrow=True)
    valid_set_x, valid_set_y = shared_dataset((valid_set_x,valid_set_y),borrow=True)
    test_x  = []
    # test feature without the previous label
    for i in range(len(test_set_x)):
       test_x.append(test_set_x[i][:-27])
       
    test_y = test_set_y
    test_set_x, test_set_y = shared_dataset((test_set_x,test_set_y),borrow=True)

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    # second-level model
    classifierTwo = build_and_train(precision_score)
    prediction = predictLabelTwo(classifierTwo, test_x)
    print "prediction is:", prediction
    print "truth is:", test_y


    """
    Predict based on two-level models
    """
    pred = predictLabel(binaryClassifier, classifierTwo,test_x)
    print "prediction is:", prediction
    print "truth is:", test_y
