import os
import sys
import json
import operator
import collections
import utils
import numpy as np
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD
import theano
import random
from sklearn.cross_validation import train_test_split
import bcubed

def getAllDocsFeatures(folder):
	x = []
	y = []
	docID = []
	docID_mentions = []
	# clusterID = []
	mentionMap = []
	allMentions = {}
	files = os.listdir(folder)
	file_count = 0
	for file_iter in files:
		print 'Getting files features for ' + file_iter
		sortedMentions = getMentionsSorted(folder, file_iter)
		allMentions.update(sortedMentions)
		# mentionPairs = returnAllPairs(sortedMentions)
		mentionPairs, mentionMap_temp = returnPairsInWindow(sortedMentions, WINDOW_SIZE)
		x_temp, y_temp = getXY(mentionPairs)
		x.extend(x_temp)
		y.extend(y_temp)
		docID.extend([file_count]*len(mentionMap_temp))
		docID_mentions.extend([file_count]*len(sortedMentions))
		print len(x_temp)
		# print len(clusterID_temp)
		# clusterID.extend(clusterID_temp)
		mentionMap.extend(mentionMap_temp)
		file_count += 1
		if file_count >= FILE_MAX:
			break
	return x, y, docID, docID_mentions, mentionMap, allMentions

def getMentionsSorted(folder, file_str):
	all_mentions = {}
	# clusterID = []
	
	# files = os.listdir(folder)
	# for file_iter in files:
	# file1 = open(folder + file_iter)
	
	file1 = open(folder+file_str)
	coref_data = json.load(file1)
	entityCount = 0
	for entity in coref_data['entities']:
		entityCount += 1
		mention_count = 0
		for mention_data in entity['mentions']:
			tempdict = {}
			# clusterID.append(entityCount)
			# tempdict['entity mention extent start'] = int(mention_data['entity mention extent start'])
			tempdict['cluster'] = entityCount
#             tempdict['mention count'] = mention_count
#             mention_count += 1
			tempdict['entity type'] = entity['entity type']
			tempdict['entity id'] = entity['entity id']
			tempdict['entity classEntity'] = entity['entity classEntity']
			tempdict['entity mention head'] = mention_data['entity mention head']
			tempdict['entity mention extent'] = mention_data['entity mention extent']
			tempdict['entity mention head start'] = int(mention_data['entity mention headStart'])
			tempdict['entity mention head end'] = int(mention_data['entity mention headEnd'])
			tempdict['entity mention extent start'] = int(mention_data['entity mention extent start'])
			tempdict['entity mention extent end'] = int(mention_data['entity mention extent end'])

			all_mentions[(tempdict['entity mention extent'], tempdict['entity mention extent start'])] = tempdict

	# print all_mentions
	# print operator.itemgetter(0)
	all_mentions_sorted = collections.OrderedDict(sorted(all_mentions.items(), key=lambda(x,y): y['entity mention extent start']))
	# print all_mentions_sorted
	return all_mentions_sorted

# sorted(statuses.iteritems(), key=lambda (x, y): y['position'])

def returnAllPairs(sorted_mentions):
	allPairs = []
	mentionMap = []
	for key1, value1 in sorted_mentions.iteritems():
		for key2, value2 in sorted_mentions.iteritems():
			if key1 != key2 and sorted_mentions[key1]['entity mention extent start'] <= sorted_mentions[key2]['entity mention extent start']:
				allPairs.append((sorted_mentions[key1], sorted_mentions[key2]))
				mentionMap.append((list(sorted_mentions.keys()).index(key1)), list(sorted_mentions.keys()).index(key2))
	return allPairs, mentionMap

def returnPairsInWindow(sorted_mentions, window_size):
	windowPairs = []
	mentionMap = []
	for key1, value1 in sorted_mentions.iteritems():
		for key2, value2 in sorted_mentions.iteritems():
			# print key1, key2
			# print list(sorted_mentions.keys()).index(key1), list(sorted_mentions.keys()).index(key2)
			if (list(sorted_mentions.keys()).index(key2) - list(sorted_mentions.keys()).index(key1) > window_size):
				break
			# print 'yoyo'
			if (list(sorted_mentions.keys()).index(key2) > list(sorted_mentions.keys()).index(key1)):
				windowPairs.append((sorted_mentions[key1], sorted_mentions[key2]))
				mentionMap.append((list(sorted_mentions.keys()).index(key1), list(sorted_mentions.keys()).index(key2)))
	return windowPairs, mentionMap

def getXY(mentionPairs):
	x = []
	y = []
	glove_word_vec_file = './baseline/data/glove/glove.6B.50d.txt'
	word_vec_dict = utils.readGloveData(glove_word_vec_file)
	for mention1, mention2 in mentionPairs:
		x_temp = []
		y_temp = [1,0]
		#if both mentions are of the same entity, y is 1
		if mention1['entity id'] == mention2['entity id']:
			y_temp = [0,1]
			# print 'yoyoyo'
		#adding word vectors to feature vectors

		x_temp.extend(utils.getWordVector(mention1['entity mention head'], word_vec_dict))
		x_temp.extend(utils.getWordVector(mention2['entity mention head'], word_vec_dict))

		#checking for head match
		if (mention1['entity mention head'] == mention2['entity mention head']):
			x_temp.extend([1])
		else:
			x_temp.extend([0])

		#checking for extent match - skipped because extent is the key in the dictionary

		# if (mention1['entity mention extent'] == mention2['entity mention extent']):
		# 	x_temp.append(1)
		# else:
		# 	x_temp.append(0)


		#checking if one mention is a substring of the other
		if (mention1['entity mention extent'] in mention2['entity mention extent']) or (mention2['entity mention extent'] in mention1['entity mention extent']):
			x_temp.extend([1])
		else:
			x_temp.extend([0])


		#checking if extent types match
		if (mention1['entity type'] == mention2['entity type']):
			x_temp.extend([1])
		else:
			x_temp.extend([0])

		#adding feature for distance between the two mentions
		
		x_temp.extend([mention1['entity mention extent start'] - mention2['entity mention extent start']])
		# print mention1
		# print mention2
		x_temp.extend([int(mention1['entity mention extent end']) - int(mention2['entity mention extent end'])])
		x_temp.extend([mention1['entity mention extent start'] - mention2['entity mention extent end']])
		x_temp.extend([mention1['entity mention extent end'] - mention2['entity mention extent start']])

	
		# distance_feature = 0.0
		# if mention1['entity mention extent start'] < mention2['entity mention extent start'] and mention1['entity mention extent end'] < mention2['entity mention extent start']:
		# 	distance_feature = mention1['entity mention extent end'] - mention2['entity mention extent start']
		# elif mention2['entity mention extent start'] < mention1['entity mention extent start'] and mention2['entity mention extent end'] < mention1['entity mention extent start']:
		# 	distance_feature = mention1['entity mention extent end'] - mention2['entity mention extent start']
		# else:
		# 	distance_feature = 0

		
		# x_temp.append(distance_feature)

		np.asarray(x_temp)
		np.asarray(y_temp)
		x.append(x_temp)
		y.append(y_temp)
	return x, y

def subsample(x, y, ratio):
	x_pos = []
	y_pos = []
	x_neg = []
	y_neg = []
	x_neg_sampled = []
	y_neg_sampled = []
	for i in range(len(y)):
		if (y[i][0] == 1):
			x_neg.append(x[i])
			y_neg.append(y[i])
		else:
			x_pos.append(x[i])
			y_pos.append(y[i])
	sample_indices = [range(len(x_neg))[i] for i in sorted(random.sample(xrange(len(range(len(x_neg)))), int(ratio*len(x_pos))))]
	for i in range(len(sample_indices)):
		x_neg_sampled.append(x_neg[i])
		y_neg_sampled.append(y_neg[i])
	x_final = x_pos
	y_final = y_pos
	x_final.extend(x_neg_sampled)
	y_final.extend(y_neg_sampled)
	return x_final, y_final

def buildModel(input_size, output_size):
	model = Sequential()

    # Two hidden layers
	model.add(Dense(80, input_dim=input_size, init='uniform', activation='tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(50, init='uniform', activation='sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(20, init='uniform', activation='sigmoid'))
	model.add(Dropout(0.5))

    # Output layer for probability
	model.add(Dense(output_size, init='uniform', activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

	return model

def getTestCluster(docID, mentionMap, y_predicted):
    cluster_alldocs = {}
    cluster_count = {}
#     cluster_count = 0
    for i in range(len(docID)):
        if docID[i] not in cluster_alldocs.keys():
            cluster_alldocs[docID[i]] = {}
            cluster_count[docID[i]] = 0
#             cluster_alldocs['cluster_size'] = 0
        mention1 = mentionMap[i][0]
        mention2 = mentionMap[i][1]
        if y_predicted[i] == 0:
            if mention1 not in cluster_alldocs[docID[i]].keys() and mention2 not in cluster_alldocs[docID[i]].keys():
                cluster_alldocs[docID[i]][mention1] = cluster_count[docID[i]]
                cluster_count[docID[i]] += 1
                cluster_alldocs[docID[i]][mention2] = cluster_count[docID[i]]
                cluster_count[docID[i]] += 1
    
            if mention1 in cluster_alldocs[docID[i]].keys() and mention2 in cluster_alldocs[docID[i]].keys():
#                 cluster_alldocs[docID[i]][mention2] = cluster_alldocs[docID[i]][mention1]
                pass
            elif mention1 in cluster_alldocs[docID[i]].keys():
#                 cluster_alldocs[docID[i]][mention2] = cluster_alldocs[docID[i]][mention1]
                cluster_alldocs[docID[i]][mention2] = cluster_count[docID[i]]
                cluster_count[docID[i]] += 1
            else:
                cluster_alldocs[docID[i]][mention1] = cluster_count[docID[i]]
                cluster_count[docID[i]] += 1
#                 cluster_alldocs[docID[i]][mention1] = cluster_alldocs[docID[i]][mention2]
                
        else:
            if mention1 not in cluster_alldocs[docID[i]] and mention2 not in cluster_alldocs[docID[i]]:
                cluster_alldocs[docID[i]][mention1] = cluster_count[docID[i]]
                cluster_alldocs[docID[i]][mention2] = cluster_count[docID[i]]
                cluster_count[docID[i]] += 1
            elif mention1 in cluster_alldocs[docID[i]].keys() and mention2 in cluster_alldocs[docID[i]].keys():
                cluster_alldocs[docID[i]][mention2] = cluster_alldocs[docID[i]][mention1]
            elif mention1 in cluster_alldocs[docID[i]].keys():
                cluster_alldocs[docID[i]][mention2] = cluster_alldocs[docID[i]][mention1]
            else:
                cluster_alldocs[docID[i]][mention1] = cluster_alldocs[docID[i]][mention2]
                
    return cluster_alldocs, cluster_count

def getGoldCluster(docID, allMentions):
    cluster_gold = {}
    cluster_count = {}
    docID_iter = 0
    for key, value in allMentions.iteritems():
        docID_temp = docID[docID_iter]
        docID_iter += 1
        if docID_temp not in cluster_gold:
            cluster_count[docID_temp] = 0
            cluster_gold[docID_temp] = {}
        cluster_gold[docID_temp][cluster_count[docID_temp]] = [allMentions[key]['cluster']]
    return cluster_gold
                
def dumpXY(x, y, xFile, yFile):
	x.dump(xFile)
	y.dump(yFile)

if __name__=="__main__":
	global FILE_MAX
	FILE_MAX = 50
	global WINDOW_SIZE
	WINDOW_SIZE = 5
	folder = './coref_output_json/'
	# sorted_mentions = getMentionsSorted(folder)
	# Pair up mentions which you want to compare
	# mentionPairs = returnAllPairs(sorted_mentions)
	# Get features: X and Y 
	x, y, docID, docID_mentions, mentionMap, allMentions = getAllDocsFeatures(folder)
	x = np.asarray(x)
	y = np.asarray(y)
	dumpXY(x, y, 'x_data.pkl', 'y_data.pkl')
	#get model according to input and output feature vectors size
	# print x
	# print y
	

	model = buildModel(len(x[0]), len(y[0]))


	#split data into train and test
	sample_rate = 0.7
	xTrain = []
	yTrain = []
	xTest = []
	yTest = []
	docID_train = []
	docID_test = []
	# clusterID_train = []
	# clusterID_test = []
	mentionMap_train = []
	mentionMap_test = []

	print 'length of x (all data) ' + str(len(x))
	print 'length of y (all data) ' + str(len(y))
	print 'negative count, positive count'
	print [sum(i) for i in zip(*y)]
	print 'length of mention pairs ' + str(len(mentionMap))
	print 'length of docID for all mention pairs ' + str(len(docID))
	# print len(clusterID)
	sample_indices = [range(len(x))[i] for i in sorted(random.sample(xrange(len(x)), int(sample_rate * len(x))))]
	# print len(sample_indices)
	# print sample_indices
	for i in range(len(x)):
	    if i in sample_indices:
	        xTrain.append(x[i])
	        yTrain.append(y[i])

	        mentionMap_train.append(mentionMap[i])
	        docID_train.append(docID[i])
	        # clusterID_train.append(clusterID[i])
	    else:
	        xTest.append(x[i])
	        yTest.append(y[i])

	        mentionMap_test.append(mentionMap[i])
	        docID_test.append(docID[i])
	        # clusterID_test.append(clusterID[i])

	# xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.30, random_state=42)

	print 'number of train examples ' + str(len(xTrain)) #, len(yTrain)
	# print [sum(i) for i in zip(*yTrain)]
	print 'number of test examples ' + str(len(xTest))#, len(yTest)
	xTrainSampled, yTrainSampled = subsample(xTrain, yTrain, 1.0)
	xTrain = xTrainSampled
	yTrain = yTrainSampled
	# print len(xTrain)
	print 'number of train examples after sampling ' + str(len(yTrain))
	print 'number of negative and positive training examples: '
	print sum(yTrain)
	# print len(xTest)
	# print len(yTest)

	# print xTest
	# print yTest
	print ''
	xTrain = np.asarray(xTrain)
	yTrain = np.asarray(yTrain)
	xTest = np.asarray(xTest)
	yTest = np.asarray(yTest)
	print type(xTrain)
	print type(yTrain)

	# for x in xTrain:
	# 	print len(x)
	# for y in yTrain:
	# 	print len(y)

	#Train model
	model.fit(xTrain, yTrain, nb_epoch = 50)

	#Test model
	performance_metrics = model.evaluate(xTest, yTest)
	y_predicted = model.predict_classes(xTest)
	print 'Length of y_predicted: ' + str(len(y_predicted))

	print 'Performance metrics for the pairwise mention predictions'
	print performance_metrics


	# print y
	# print x

	cluster_test, cluster_count = getTestCluster(docID_test, mentionMap_test, y_predicted)
	print len(cluster_test)
	print cluster_test
	for i in range(len(mentionMap_test)):
		print str(mentionMap_test[i]) + ': ' + str(y_predicted[i]) + str(yTest[i])

	yTest_class = []
	for i in range(len(yTest)):
	    if yTest[i][0] == 1:
	        yTest_class.append(0)
	    else:
	        yTest_class.append(1)
	cluster_gold, cluster_gold_count = getTestCluster(docID_test, mentionMap_test, yTest_class)


	# print len(allMentions)
	# print cluster_gold
	for key1, value1 in cluster_test.iteritems():
	    for key2, value2 in cluster_test[key1].iteritems():
	        cluster_test[key1][key2] = set([cluster_test[key1][key2]])
	for key1, value1 in cluster_gold.iteritems():
	    for key2, value2 in cluster_gold[key1].iteritems():
	        cluster_gold[key1][key2] = set([cluster_gold[key1][key2]])


	all_precision = []
	all_recall = []
	all_fscore = []
	for key, value in cluster_test.iteritems():
	    precision = bcubed.precision(cluster_test[key], cluster_gold[key])
	    recall = bcubed.recall(cluster_test[key], cluster_gold[key])
	    fscore = bcubed.fscore(precision, recall)
	    print 'precision: ' + str(precision)
	    all_precision.append(precision)
	    print 'recall: ' + str(recall)
	    all_recall.append(recall)
	    print 'fscore: ' + str(fscore)
	    all_fscore.append(fscore)
	    print ''
	print 'avg b-cubed precision: ' + str(sum(all_precision)/len(all_precision))
	print 'avg b-cubed recall: ' + str(sum(all_recall)/len(all_recall))
	print 'avg b-cubed fscore: ' + str(sum(all_fscore)/len(all_fscore))
	print 'number of negative predictions = ' + str(sum(y_predicted==0)) + ' and positive predictions = ' + str(sum(y_predicted == 1))
	#TO-DOS
	#1. Try other dimensional word embeddings