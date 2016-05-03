


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
import mention_data_processor
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import itertools
import random


def splitting_data(X,Y):
	print('Splitting the data into train, validation, test sets with ratio 8:1:1 ...')
	X_train, X_rest, y_train, y_rest = train_test_split(X, Y, test_size=0.2, random_state=42)
	X_val, X_test, y_val, y_test = train_test_split(X_rest,y_rest, test_size=0.5, random_state=42)
	return X_train, X_val, X_test, y_train, y_val, y_test


def balancer(X,Y,SampleRatio=3):
	"""
	sample part of the 0 labels for the
	training and tripple sample from 1 labels
	Input: feature list: X, label list: Y
	Output: sampled feauture list sampledX, 
	sampled label list sampledY
	"""
	sampledX = []
	sampledY = []
	nullPos = []
	# get all indices of 'O'-labeled data
	nonNullNum = 0
	for pos in range(len(Y)):
		label = Y[pos]
		if label == 0:
			nullPos.append(pos)
		else:
			nonNullNum += 1
	# randomly sample part of 0 
	num = nonNullNum * SampleRatio
	sampledNullPos = random.sample(nullPos, num)
	for pos in range(len(Y)):
		if ((pos in nullPos) and (pos not in sampledNullPos)):
			continue
		elif pos not in nullPos:
			feature = X[pos]
			label = Y[pos]
			print "Not null label, sampling 3 times",label
			for i in range(SampleRatio):
				sampledX.append(feature)
				sampledY.append(label)
		else:
			feature = X[pos]
			label = Y[pos]
			print "Null label that is sampled",label
			sampledX.append(feature)
			sampledY.append(label)
	return sampledX, sampledY


def build_and_train(optimizing_function):
	print('building the model ... ')
	# allocate symbolic variables for the data
	index = T.lscalar()  # index to a [mini]batch
	x = T.matrix('x')  # the data is presented as rasterized images
	y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
	rng = np.random.RandomState(1234)
	classifier = FFNN(
        rng=rng,
        input=x,
        n_in=n_in,
        n_hidden=n_hidden,
        n_out=n_out
        )
	cost = (
    	classifier.negative_log_likelihood(y)
    	+ L1_reg * classifier.L1
    	+ L2_reg * classifier.L2_sqr
    	)
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
	gparams = [T.grad(cost, param) for param in classifier.params]
	updates = [(param, param - learning_rate * gparam) for param, gparam in zip(classifier.params, gparams)]
	train_model = theano.function(
    	inputs = [index],
    	outputs = cost,
    	updates = updates,
    	givens = {
    		x: train_set_x[index*batch_size: (index + 1) * batch_size],
    		y: train_set_y[index*batch_size: (index + 1) * batch_size]
    		}
    		)
	print('training ... ')
	patience = 10000  # look at this many examples regardless
	patience_increase = 2  # wait this much longer when a new best is found
	improvement_threshold = 0.995  # a relative improvement of this much is
	validation_frequency = min(n_train_batches, patience // 2)
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
				validation_loss = [validate_model_accuracy(i) for i in range(n_valid_batches)]
				this_validation_loss = np.mean(validation_loss)
				validation_precision = [np.mean(precision_score(*validate_model(i),pos_label=None,average=None)) for i in range(n_valid_batches)]
				this_validation_precision = np.mean(validation_precision)
				validation_recall = [np.mean(recall_score(*validate_model(i),pos_label=None,average=None)) for i in range(n_valid_batches)]
				this_validation_recall = np.mean(validation_recall)
				validation_fscore = [np.mean(f1_score(*validate_model(i),pos_label=None,average=None)) for i in range(n_valid_batches)]
				this_validation_fscore = np.mean(validation_fscore)
				print('epoch %i, minibatch %i/%i, average validation precision %f, validation recall %f, validation fscore %f, and loss %f %%' %(epoch,minibatch_index + 1,n_train_batches,this_validation_precision * 100,this_validation_recall * 100,this_validation_fscore * 100,this_validation_loss * 100.))
				if this_validation_loss < best_validation_loss:
					if (this_validation_loss < best_validation_loss * improvement_threshold):
						patience = max(patience, iter * patience_increase)
					
					best_validation_loss = this_validation_loss
					best_iter = iter
					test_losses = [np.mean(optimizing_function(*test_model(i),pos_label=None,average=None)) for i in range(n_test_batches)]
					test_score = np.mean(test_losses)
					print(('epoch %i, minibatch %i/%i, test performace for the optimizing function %f %%') %(epoch, minibatch_index + 1, n_train_batches,test_score * 100.))
					#return classifier
			if patience <= iter:
				done_looping = True
				break
	end_time = timeit.default_timer()
	print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
	print(('The code for file ran for %.2fm' % ((end_time - start_time) / 60.)))

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




if __name__ == '__main__':
    n_out = 2
    window_size=5
    word_vec_dim=50
    n_hidden=300
    learning_rate=0.01
    L1_reg=0.00
    L2_reg=0.0001
    n_epochs=1000
    batch_size=20
    n_in = (2+word_vec_dim) + window_size * 2 * word_vec_dim

    print "RUNNING MENTION HEAD DETECTION MODEL"
    coref_jsonPath = '../../coref_data'
    rel_jsonPath = '../../relation_data'
    wordToVecDictFile = '../../data/glove/glove.6B.50d.txt'
    print "processing feature space"
    mention_data = mention_data_processor.mention_meta_data_processor(coref_jsonPath, rel_jsonPath)
    wordVecDic = FeatureProcessing.readDictData(wordToVecDictFile)
    XX, YY = FeatureProcessing.featureProcess_mention_head(mention_data,wordVecDic,window_size) 
    print "balancing the dataset"
    new_XX, new_YY = balancer(XX,YY,3)
    pickle.dump(new_XX,open('../../data/balanced_mention_data.pkl','wb'))
    pickle.dump(new_YY,open('../../data/balanced_mention_label.pkl','wb'))
    train_set_x, valid_set_x, test_set_x, train_set_y, valid_set_y, test_set_y = splitting_data(new_XX,new_YY)
    train_set_x, train_set_y = shared_dataset((train_set_x,train_set_y),borrow=True)
    valid_set_x, valid_set_y = shared_dataset((valid_set_x,valid_set_y),borrow=True)
    test_set_x, test_set_y = shared_dataset((test_set_x,test_set_y),borrow=True)

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    classifier = build_and_train(recall_score)
