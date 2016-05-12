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
   num = nonNullNum / 20 + 1
   sampledNullPos = random.sample(nullPos, num)

   for pos in range(len(Y)):
      if ((pos in nullPos) and (pos not in sampledNullPos)):
         continue
      else:
         feature = X[pos]
         label = Y[pos]
         sampledX.append(feature)
         sampledY.append(label)
   #print sampledY
   return sampledX, sampledY


def build_and_train(optimizing_function, numOfOut):
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
##    test_model_accuracy = theano.function(
##        inputs=[index],
##        outputs=classifier.errors(y),
##        givens={
##        x: test_set_x[index * batch_size: (index + 1) * batch_size],
##        y: test_set_y[index * batch_size: (index + 1) * batch_size]
##        }
##    )
##
##    test_model = theano.function(
##    inputs=[index],
##    outputs=[y, classifier.y_pred],
##    givens={
##        x: test_set_x[index * batch_size: (index + 1) * batch_size],
##        y: test_set_y[index * batch_size: (index + 1) * batch_size]
##    }
##)
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
    patience = 25000  # look at this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_classifier = classifier
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
                validation_precision = [np.mean(precision_score(*validate_model(i),pos_label=None,average=None)[0:numOfOut]) for i in range(n_valid_batches)]
                this_validation_precision = np.mean(validation_precision)

                # Optimizing for recall
                validation_recall = [np.mean(recall_score(*validate_model(i),pos_label=None,average=None)[0:numOfOut]) for i in range(n_valid_batches)]
                this_validation_recall = np.mean(validation_recall)

                # Optimizing for f-score
                validation_fscore = [np.mean(f1_score(*validate_model(i),pos_label=None,average=None)[0:numOfOut]) for i in range(n_valid_batches)]
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

                    best_classifier = classifier
                    

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print ("the total iterations are: "+str(iter))
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ran for %.2fm' % ((end_time - start_time) / 60.)))
    return best_classifier
   
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
    trainingDataFile = 'ner_data/traindata_ner.txt'
    trainingLabelFile = 'ner_data/trainlabel_ner.txt'
    #binaryLabelFile = '../data/binaryLabel.txt'
    wordToVecDictFile = '../data/glove/glove.6B.50d.txt'
    print('Vectorizing the features and labels...')
    start_time = timeit.default_timer()

    # vectorize the training data
    X, Y = FeatureProcessing.binaryFeatureProcess(trainingDataFile,trainingLabelFile,wordToVecDictFile,window_size)
    end_time = timeit.default_timer()
    print(('The vectorization ran for %.2fm' % ((end_time - start_time) / 60.)))
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

def load_data():
    '''
    Loads the data, turns into word2vec representation, and splits
    into training, validation, and testing sets with ratio 8:1:1
    '''
    trainingDataFile = 'ner_data/traindata_ner.txt'
    trainingLabelFile = 'ner_data/trainlabel_ner.txt'
    wordToVecDictFile = '../data/glove/glove.6B.50d.txt'
    print('Vectorizing the features and labels...')
    start_time = timeit.default_timer()
    X, Y = FeatureProcessing.featureProcess(trainingDataFile,trainingLabelFile,wordToVecDictFile,window_size)
    sampledX, sampledY = sample(X,Y)
    end_time = timeit.default_timer()
    print(('The vectorization ran for %.2fm' % ((end_time - start_time) / 60.)))
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

def load_test_data():
   """
   Load the test data, get the feature
   """
   testDataFile = "ner_data/testdata_ner.txt"
   wordToVecDictFile = "../data/glove/glove.6B.50d.txt"
   X = FeatureProcessing.testFeatureProcess(testDataFile,wordToVecDictFile,window_size)
   return X

def load_test_label():
   """
   Load the test label
   """
   testLabelFile = "ner_data/testlabel_ner.txt"
   rawLabelData = open(testLabelFile, 'r').readlines()
   Y = []
   for line in rawLabelData:
      labelSeq = line.strip().split()
      for label in labelSeq:
         Y.append(getLabelIndex(label))
   return Y
   

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
       #print "predicted label is:", label
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

def predictLabel2(c1, c2, xData):
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
      if ((p1 == 0) and (p2==nullIndex)):
         pred.append(nullIndex)
      else:
         pred.append(p2)
   return pred

def getFscore(y,y_pred):
   # Confusion Matrix: row is the real label, col is the predicted label
   confusion_mat = []
   for i in range(27):
      row = [0]*27
      confusion_mat.append(row)
   for i in range(len(y)):
      trueLabel = y[i]
      predLabel = y_pred[i]
      confusion_mat[trueLabel][predLabel] += 1

   print(confusion_mat)
   
   precision = []
   recall = []
   for i in range(27):
      tp = confusion_mat[i][i]
      colSum = 0
      rowSum = 0
      for j in range(27):
         colSum += confusion_mat[j][i]
         rowSum += confusion_mat[i][j]
      if (colSum == 0):
         precision.append(None)
      else:
         precision.append(1.0*tp/colSum)
      if (rowSum == 0):
         recall.append(None)
      else:
         recall.append(1.0*tp/rowSum)
   print("precision is: ")
   print(precision)
   print("recall is: ")
   print(recall)
   

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
    batch_size = 20

    # load test data
    test_x = load_test_data()
    test_y = load_test_label()
    
    print("Running NER training -- Step 1: binary model training")
    
    train_set_x, valid_set_x, train_set_y, valid_set_y = load_binary_data()
    train_set_x, train_set_y = shared_dataset((train_set_x,train_set_y),borrow=True)
    valid_set_x, valid_set_y = shared_dataset((valid_set_x,valid_set_y),borrow=True)

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size

    # binary model
    binaryClassifier = build_and_train(precision_score,2)
    """
    prediction = predictBinaryLabel(binaryClassifier, test_x)
    print "prediction is:", prediction
    print "truth is:", test_y
    """


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
   """
    
    #train_set_x, valid_set_x, test_set_x, train_set_y, valid_set_y, test_set_y = load_data()
    train_set_x, valid_set_x, train_set_y, valid_set_y = load_data()
    train_set_x, train_set_y = shared_dataset((train_set_x,train_set_y),borrow=True)
    valid_set_x, valid_set_y = shared_dataset((valid_set_x,valid_set_y),borrow=True)
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    # second-level model
    classifierTwo = build_and_train(precision_score,27)
    """
    prediction = predictLabelTwo(classifierTwo, test_x)
    print "prediction is:", prediction
    print "truth is:", test_y
    """


    """
    Predict based on two-level models
    """
    pred = predictLabel(binaryClassifier, classifierTwo,test_x)
    f = open("pred_res.txt","w")
    for label in pred:
       print >> f, label
    f.close()
    #print "prediction is:", prediction
    #print "truth is:", test_y
    getFscore(test_y, pred)

    print "-------------------------------------"
    print "------Another Method's results-------"
    print "-------------------------------------"
    pred2 = predictLabel2(binaryClassifier, classifierTwo,test_x)
    g = open("pred_res2.txt","w")
    for label in pred2:
       print >> g, label
    g.close()
    getFscore(test_y, pred2)
