#
# Master file that performs end-to-end prediction
# 

# Import libraries
from Mentions import *

import os
import sys
import timeit
import numpy as np

try:
   import cPickle as pickle
except:
   import pickle as pickle
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import json



if __name__ == '__main__':
	ROOT_DIR = "/Users/bolor/Documents/Statistics/CS546/project/baseline"
	WORD_VEC_DIM = 50
	WINDOW_SIZE = 5
	N_HIDDEN = 300
	LEARNING_RATE = 0.01
	L1_REG = 0.00
	L2_REG = 0.0001
	COREF_DATA_DIR = ROOT_DIR + "/coref_data"
	REL_DATA_DIR = ROOT_DIR + "/relation_data"
	PICKLE_FILE_PATH = ROOT_DIR + "/data/train_mention_data_balanced.pkl"
	training_file_list = [pos_json.strip() for pos_json in os.listdir(COREF_DATA_DIR) if pos_json.endswith('.json')]
	testing_file_list = None

	if os.path.exists(PICKLE_FILE_PATH):
		print("Unpickling existing balanced datasets ... ")
		X_train, X_val, y_train, y_val = pickle.load(open(PICKLE_FILE_PATH))
	else:
		feature_processor = FeatureProcessor(WORD_VEC_DIM, WINDOW_SIZE, COREF_DATA_DIR, REL_DATA_DIR, training_file_list, testing_file_list)
		feature_processor.readDictData()
		print("Loading the mentions data ... ")
		feature_processor.processMentionData()
		print("Feature processing ... ")
		feature_processor.featureProcessMentionHead(data_type="train")
		#feature_processor.featureProcessMentionHead(data_type="test")
		#test_set_x, test_set_y = feature_processor.test_mention_data_featured
		print("Balancing the training set ... ")
		feature_processor.balanceTrainingSet()
		print "the number of training observations: %i " %len(feature_processor.mention_data_featured_balanced[0])
		print("Splitting the training set into train and validation sets with ratio 9:1 ...")
		X_train, X_val, y_train, y_val = feature_processor.trainValidSplit()
		print("Pickling the balanced and split datasets ... ")
		pickle.dump((X_train, X_val, y_train, y_val),open(PICKLE_FILE_PATH,'wb'))

	mention_model = MentionHeadModel(X_train, y_train, X_val, y_val, WORD_VEC_DIM, N_HIDDEN, WINDOW_SIZE, LEARNING_RATE, L1_REG, L2_REG)
	mention_model.setDatasetsShared()
	mention_model.train()
	test_set_y_pred = mention_model.predictLabels(test_set_x)
	f_score = f1_score(test_set_y_pred, test_set_y)
	precision = precision_score(test_set_y_pred, test_set_y)
	recall = recall_score(test_set_y_pred, test_set_y)






