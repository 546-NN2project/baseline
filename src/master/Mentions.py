# Mention head model 
# 
#


import os
import sys
import timeit

import numpy as np
import time
from sklearn.cross_validation import train_test_split

try:
   import cPickle as pickle
except:
   import pickle as pickle

import theano
import theano.tensor as T
from FFNN import *
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import itertools
import random

import editdistance
import json
import collections
import nltk

class FeatureProcessor(object):
	def __init__(self, word_vec_dim, window_size,coref_data_dir, rel_data_dir, training_file_list=None, testing_file_list=None):
		self.word_vec_dim = word_vec_dim
		self.window_size = window_size
		self.coref_data_dir = coref_data_dir
		self.rel_data_dir = rel_data_dir
		self.training_file_list = training_file_list
		self.testing_file_list = testing_file_list

		if word_vec_dim == 50:
			self.wordToVecDictFile = '../../data/glove/glove.6B.50d.txt'
		elif word_vec_dim == 100:
			self.wordToVecDictFile = '../../data/glove/glove.6B.100d.txt'
		elif word_vec_dim == 150:
			self.wordToVecDictFile = '../../data/glove/glove.6B.150d.txt'
		else:
			self.wordToVecDictFile = '../../data/glove/glove.6B.200d.txt'
		
	def readDictData(self):
		f = open(self.wordToVecDictFile, 'r')
		rawData = f.readlines()
		wordVecDict = {}
		for line in rawData:
			line = line.strip().split()
			word = line[0]
			vec = line[1:]
			wordVecDict[word] = np.array(vec, dtype=float)
		self.wordVecDict = wordVecDict 
	
	def get_word_vector(self,word):
		"""
		Given a word or a PAD vector, returns the word2vec vector or the zero vector
		"""
		if word != 'PAD' and word.lower() in self.wordVecDict:
			return list(self.wordVecDict[word.lower()])
		return list(np.zeros_like(self.wordVecDict['hi']))

	def context_tokens(self,tokenized_sentence, token):
		"""
		Given a token in a sentence, returns (window_size) tokens before and after. 

		:param tokenized_sentence: list of tokens that make up a sentence
		:param token: the current token for which we want the context words
		:window_size: how many words before and after the current token to take

		Output: list of tokens of size 2*(window_size)
		"""
		token_pos = tokenized_sentence.index(token) #position of the token in the sentence
		n_sentence = len(tokenized_sentence) 
		frontContext_tokens = []
		backContext_tokens = []
		for i in range(self.window_size):
			current_front_vector = token_pos - self.window_size + i
			if current_front_vector < 0 :
				frontContext_tokens.append('PAD')
			else:
				frontContext_tokens.append(tokenized_sentence[current_front_vector])
			current_back_vector = token_pos + i + 1
			
			if current_back_vector >= n_sentence:
				backContext_tokens.append('PAD')
			else:
				backContext_tokens.append(tokenized_sentence[current_back_vector])
		return frontContext_tokens + backContext_tokens

	def is_word_upper(self,word):
		if word[0].isupper():
			return [1]
		else:
			return [0]

	def POSlist(self,tokenized_sent):
		tagged_sent = nltk.pos_tag(tokenized_sent)
		POSlist = [tag[1] for tag in tagged_sent]
		return POSlist 

	def POStagger(self,token_idx, tokenized_sent):
		tagged_sent = nltk.pos_tag(tokenized_sent)
		return tagged_sent[token_idx][1]


	def get_label_POS(self,label):
		NN_list = ["NN","NNP","NNPS","NNS","PRP","PRP$","WDT","WP","WP$"]
		if label in NN_list:
			return [1]
		else:
			return [0]

	def featureProcessMentionHead(self,data_type):
		"""
		input: mention data obtained from mention_meta_data_processor
		"""
		if data_type == "train":
			mention_data = self.train_mention_data
		elif data_type == "test":
			if self.test_mention_data is not None:
				mention_data = self.test_mention_data
			else:
				print("No test file list was supplied.")
				raise NotImplementedError
		XX = []
		YY = []
		for document_data in mention_data:
			for sent_data in document_data:
				tokenized_sent = sent_data['sentence'].strip().split()
				POS_sent = self.POSlist(tokenized_sent)
				labels = self.find_mention_head_labels(sent_data)
				if len(tokenized_sent) != len(labels):
					raise NotImplementedError
				for idx, token in enumerate(tokenized_sent):
					X = self.is_word_upper(token) + self.get_word_vector(token) + self.get_label_POS(POS_sent[idx])
					context = self.context_tokens(tokenized_sent, token)
					for context_token in context:
						X = X + self.get_word_vector(context_token)
					XX.append(X)
					YY.append(labels[idx])
		if len(XX) != len(YY):
			raise NotImplementedError
		if data_type == "train":
			self.train_mention_data_featured = (XX,YY)
		elif data_type == "test":
			self.test_mention_data_featured = (XX,YY)

	def find_mention_head_labels(self,mention_sentence_data):
		"""
		Given a sentence data with mentions identified, gives 0-1 label for each 
		token identifying if the token is a mention head or not.
		mentions format:
		(within_sent_start, within_sent_end, mention_extent, mention_head, original_start_char, original_end_char)
		the code first identifies the position of mention_head within the metnion_extent
		then adds this position to the beginning of the mention_extent (i.e. 
		within_sent_start)
		"""
		labels = []
		sentence = str(mention_sentence_data['sentence'])
		mentions = mention_sentence_data['mentions']  
		tokenized_sent = sentence.strip().split()
		token_starts = [sentence.find(token) for token in tokenized_sent]
		mention_head_starts = []
		for mention in mentions:
			mention_extent = str(mention[2])
			mention_head = str(mention[3])
			mention_head_within_extent = mention_extent.find(mention_head)
			mention_head_start = mention[0] + mention_head_within_extent
			mention_head_starts.append(mention_head_start)
		labels = []
		for idx,token in enumerate(tokenized_sent):
			if token_starts[idx] in mention_head_starts:
				labels.append(1)
			elif (token_starts[idx] + 1) in mention_head_starts:
				labels.append(1)
			elif (token_starts[idx] - 1) in mention_head_starts:
				labels.append(1)
			else:
				labels.append(0)
		return labels

	def extract_list_of_mentions(self,document_coref_data):
		list_of_mentions = []
		for entity_dic in document_coref_data['entities']:
			mentions = {}
			for men_dic in entity_dic['mentions']:
				mention_extent = men_dic['entity mention extent']
				mention_head = men_dic['entity mention head']
				start = int(men_dic['entity mention extent start'])
				end = int(men_dic['entity mention extent end'])
				list_of_mentions.append([start,end,mention_extent,mention_head,start,end])
		list_of_mentions.sort()
		return list_of_mentions

	def find_mention(self,sentence, mention_zero):
		return str(sentence).find(str(mention_zero[2]))

	def mention_cleaner(self,list_of_mentions):
		new_list = []
		for mention in list_of_mentions:
			new_list.append(mention[2:])
		return new_list

	def sent_divider(self,length, list_of_mentions):
		if len(list_of_mentions) == 0:
			return [], []
		else:
			for idx, mention in enumerate(list_of_mentions):
				if mention[0] > length:
					return list_of_mentions[:idx], list_of_mentions[idx:]
		return list_of_mentions, [] 

	def mentions_in_sents(self,list_of_mentions, sentences):
		"""
		given a list of documents and sentences it comes from, 
		returns a list of dictionaries with a sentence and the mentions
		within the sentence
		"""
		data = []
		lengths = [len(sent) for sent in sentences]
		list_of_mentions = self.prune_absent_mentions(list_of_mentions,sentences)
		mention_zero_index = self.find_mention(sentences[0], list_of_mentions[0])
		#if mention_zero_sent > 0:
		#	sentences = sentences[mention_zero_sent:]
		#	lengths = lengths[mention_zero_sent:]
		first_diff = list_of_mentions[0][0] - mention_zero_index 
		shifted_list_of_mentions = [(a-first_diff,b-first_diff,mention_extent,mention_head,h,t) for (a,b,mention_extent,mention_head,h,t) in list_of_mentions]
		for idx, diff in enumerate(lengths): 
			sentence_data = {}
			sentence_data['sentence'] = sentences[idx]
			sent_prev, list_of_mentions = self.sent_divider(diff, shifted_list_of_mentions)
			sentence_data['mentions'] = sent_prev
			if len(list_of_mentions) > 0:
				shifted_list_of_mentions = [(a-diff-2, b-diff-2, mention_extent,mention_head,h,t) for (a,b,mention_extent,mention_head,h,t) in list_of_mentions]					
			else:
				shifted_list_of_mentions = [] 
			data.append(sentence_data)
		return data 

	def prune_absent_mentions(self,list_of_mentions, sentences):
		for mention_idx, mention in enumerate(list_of_mentions):
			mention_matched = False
			for sent in sentences:
				if str(mention[2]) in str(sent):
					mention_matched = True
			if mention_matched == False:
				list_of_mentions.pop(mention_idx)
		return list_of_mentions

	def extract_list_of_relations(self,document_rel_data):
		list_of_relations = []
		for idx, sent_idx in enumerate(document_rel_data):
			list_of_relations = list_of_relations + document_rel_data[sent_idx]["relations"]
		return list_of_relations 

	def mention_map(self,mention_coord, list_of_mentions ):
		for mention in list_of_mentions:
			if (mention[2],mention[3]) == mention_coord:
				return mention
		return None 

	def assign_relations1(self,sent_with_mentions, list_of_relations):
		mentions = sent_with_mentions['mentions']
		mention_coord = [(h,t) for (mention_extent,mention_head, h,t) in mentions]
		relations = []
		for relation in list_of_relations:
			rel = {}
			start1 = int(relation['arg1_start'])
			end1 = int(relation['arg1_end'])
			start2 = int(relation['arg2_start'])
			end2 = int(relation['arg2_end'])		
			if (start1,end1) in mention_coord:
				rel['arg1'] = mention_map((start1,end1),mentions)
				rel['arg2'] = mention_map((start2,end2),mentions)
				rel['relation_type'] = relation['relation_type']
				relations.append(rel)
		return relations 
	def assign_relations(self,sent_with_mentions,list_of_relations):
		mentions = sent_with_mentions['mentions']
		mention_starts = [h for (a,b,mention_extent,mention_head,h,t) in mentions]
		relations = []
		#mentions = [ (mention,mention_head,h,t) for (h,t)  ]
		for relation in list_of_relations:
			start1 = int(relation['arg1_start'])
			end1 = int(relation['arg1_end'])
			start2 = int(relation['arg2_start'])
			end2 = int(relation['arg2_end'])
			if start1 in mention_starts:
				relations.append(relation)
		return relations 


	def mention_sentence_matcher(self,document_coref_data, document_rel_data):
		list_of_mentions = self.extract_list_of_mentions(document_coref_data)
		mentions_in_sents_data = self.mentions_in_sents(list_of_mentions, document_coref_data['sentences'])
		list_of_relations = self.extract_list_of_relations(document_rel_data)
		for idx, sent_data in enumerate(mentions_in_sents_data):
			try:
				sent_data["relations"] = self.assign_relations(sent_data,list_of_relations)
				#sent_idx = idx + 1
				#sent_data["relations"] = document_rel_data[str(sent_idx)]["relations"]
			except:
				sent_data["relations"] = []
		return mentions_in_sents_data

	def processMentionData(self):
		if self.training_file_list is not None:
			self.train_mention_data = self.process_mention_data_from_list("train")
		else:
			print("No Training file list given.")
			raise NotImplementedError
		if self.testing_file_list is not None:
			self.test_mention_data = self.process_mention_data_from_list("test")
		else:
			self.test_mention_data = None
			print("Warning: No Testing file list given. ")


	def process_mention_data_from_list(self,data_type):
		meta_data = []
		if data_type == "train":
			list_of_files = self.training_file_list
		elif data_type == "test":
			lsit_of_files = self.testing_file_list
		else:
			print("Unrecognized data_type given.")
			raise NotImplementedError

		for js in list_of_files:
			try:
				document_rel_data = json.load(open(os.path.join(self.rel_data_dir, js)))
				document_coref_data = json.load(open(os.path.join(self.coref_data_dir, js)))
				try:
					meta_data.append(self.mention_sentence_matcher(document_coref_data, document_rel_data))
				except:
					print "there was a problem with {}".format(js)
			except:
				print "file {} not found".format(js)
		return meta_data


	def balanceTrainingSet(self,SampleRatio=3):
		"""
		sample part of the 0 labels for the
		training and tripple sample from 1 labels
		Input: feature list: X, label list: Y
		Output: sampled feauture list sampledX, 
		sampled label list sampledY
		"""
		X,Y = self.train_mention_data_featured 
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
				for i in range(SampleRatio):
					sampledX.append(feature)
					sampledY.append(label)
			else:
				feature = X[pos]
				label = Y[pos]
				sampledX.append(feature)
				sampledY.append(label)
		self.mention_data_featured_balanced = (sampledX, sampledY)
		return sampledX, sampledY

	def trainValidSplit(self):
		X,Y = self.mention_data_featured_balanced
		X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=42)
		return X_train, X_val, y_train, y_val

class MentionHeadModel(object):
	def __init__(self, train_set_x, train_set_y, valid_set_x, valid_set_y, word_vec_dim, n_hidden, window_size, learning_rate, L1_reg, L2_reg):
		self.train_set_x = train_set_x
		self.train_set_y = train_set_y
		self.valid_set_x = valid_set_x
		self.valid_set_y = valid_set_y
		self.word_vec_dim = word_vec_dim
		self.n_hidden = n_hidden
		self.n_out = 2
		self.window_size = window_size
		self.n_in = (2+word_vec_dim) + window_size * 2 * word_vec_dim
		self.learning_rate = learning_rate
		self.L1_reg = L1_reg
		self.L2_reg = L2_reg
		self.patience_increase = 5
		self.n_epochs = 130
		self.batch_size = 20
		self.rng = np.random.RandomState(42)
		self.n_train_batches = len(train_set_x) // self.batch_size
		self.n_valid_batches = len(valid_set_x) // self.batch_size

	def shared_dataset(self, data_xy):
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
								 borrow=True)
		shared_y = theano.shared(np.asarray(data_y,
											   dtype=theano.config.floatX),
								 borrow=True)
		return shared_x, T.cast(shared_y, 'int32')

	def setDatasetsShared(self):
		shared_train_set_x, shared_train_set_y = self.shared_dataset((self.train_set_x, self.train_set_y))
		shared_valid_set_x, shared_valid_set_y = self.shared_dataset((self.valid_set_x, self.valid_set_y))
		self.train_set_x = shared_train_set_x
		self.train_set_y = shared_train_set_y
		self.valid_set_x = shared_valid_set_x
		self.valid_set_y = shared_valid_set_y

	def train(self):
		print('building the model ... ')
		# allocate symbolic variables for the data
		index = T.lscalar()  # index to a [mini]batch
		x = T.matrix('x')  # the data is presented as rasterized images
		y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
		
		classifier = FFNN(rng=self.rng, input=x, n_in=self.n_in, n_hidden=self.n_hidden,n_out=self.n_out)
		
		cost = (classifier.negative_log_likelihood(y) + self.L1_reg * classifier.L1 + self.L2_reg * classifier.L2_sqr)
		validate_model_accuracy = theano.function(
			inputs=[index],
			outputs=classifier.errors(y),
			givens={
			x: self.valid_set_x[index * self.batch_size: (index + 1) * self.batch_size],
			y: self.valid_set_y[index * self.batch_size: (index + 1) * self.batch_size]
			}
			)
		validate_model = theano.function(
			inputs=[index],
			outputs=[y, classifier.y_pred],
			givens={
				x: self.valid_set_x[index * self.batch_size: (index + 1) * self.batch_size],
				y: self.valid_set_y[index * self.batch_size: (index + 1) * self.batch_size]
				}
				)
		gparams = [T.grad(cost, param) for param in classifier.params]
		updates = [(param, param - self.learning_rate * gparam) for param, gparam in zip(classifier.params, gparams)]
		train_model = theano.function(
			inputs = [index],
			outputs = cost,
			updates = updates,
			givens = {
				x: self.train_set_x[index*self.batch_size: (index + 1) * self.batch_size],
				y: self.train_set_y[index*self.batch_size: (index + 1) * self.batch_size]
				}
				)
		print('training ... ')
		patience = 10000  # look at this many examples regardless
		patience_increase = self.patience_increase  # wait this much longer when a new best is found
		improvement_threshold = 0.995  # a relative improvement of this much is
		validation_frequency = min(self.n_train_batches, patience // 2)
		best_validation_loss = np.inf
		best_iter = 0
		test_score = 0.
		start_time = timeit.default_timer()
		epoch = 0
		done_looping = False
		while (epoch < self.n_epochs) and (not done_looping):
			epoch = epoch + 1
			for minibatch_index in range(self.n_train_batches):
				minibatch_avg_cost = train_model(minibatch_index)
				# iteration number
				iter = (epoch - 1) * self.n_train_batches + minibatch_index
				if (iter + 1) % validation_frequency == 0:
					validation_loss = [validate_model_accuracy(i) for i in range(self.n_valid_batches)]
					this_validation_loss = np.mean(validation_loss)
					validation_precision = [np.mean(precision_score(*validate_model(i),pos_label=None,average=None)) for i in range(self.n_valid_batches)]
					this_validation_precision = np.mean(validation_precision)
					validation_recall = [np.mean(recall_score(*validate_model(i),pos_label=None,average=None)) for i in range(self.n_valid_batches)]
					this_validation_recall = np.mean(validation_recall)
					validation_fscore = [np.mean(f1_score(*validate_model(i),pos_label=None,average=None)) for i in range(self.n_valid_batches)]
					this_validation_fscore = np.mean(validation_fscore)
					print('epoch %i, minibatch %i/%i, average validation precision %f, validation recall %f, validation fscore %f, and loss %f %%' %(epoch,minibatch_index + 1,self.n_train_batches,this_validation_precision * 100,this_validation_recall * 100,this_validation_fscore * 100,this_validation_loss * 100.))
					if this_validation_loss < best_validation_loss:
						if (this_validation_loss < best_validation_loss * improvement_threshold):
							patience = max(patience, iter * patience_increase)

						best_validation_loss = this_validation_loss
						best_validation_precision = this_validation_precision
						best_validation_recall = this_validation_recall
						best_validation_fscore = this_validation_fscore
						best_iter = iter
						best_states = classifier.getstate()
				if patience <= iter:
					done_looping = True
					break
		end_time = timeit.default_timer()
		print(('Optimization complete. Best validation score of %f %% obtained at iteration %i, with best vaildation recall : %f, precision: %f, fscore: %f %%') %(best_validation_loss * 100., best_iter + 1, this_validation_recall * 100., this_validation_precision *100., this_validation_fscore * 100.))
		print(('The code for file ran for %.2fm' % ((end_time - start_time) / 60.)))
		self.best_classifier = classifier
		self.best_weights = best_states

	def predictScores(self, test_set_x):
		"""
		An example of how to load a trained model and use it
		to predict labels.
		"""
		# compile a predictor function
		predict_model = theano.function(
			inputs=[self.best_classifier.input],
			outputs=self.best_classifier.p_y_given_x)
		predicted_values = predict_model(test_set_x)
		print("Predicted values for the first 10 examples in test set:")
		print(predicted_values[:10])
		return predicted_values

	def predictLabels(self,test_set_x):
		"""
		An example of how to load a trained model and use it
		to predict labels.
		"""
		# compile a predictor function
		predict_model = theano.function(
			inputs=[self.best_classifier.input],
			outputs=self.best_classifier.y_pred)
		predicted_values = predict_model(test_set_x)
		print("Predicted values for the first 10 examples in test set:")
		print(predicted_values[:10])
		return predicted_values







