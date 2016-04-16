"""
Mention data processing

For each token in a sentence, word phrase of lengths 1,2,...,10
is created as mention candidates. For example,

All mention candidates starting with token 'DAVAO' are as follows:

[u'DAVAO,']
[u'DAVAO,', u'Philippines,']
[u'DAVAO,', u'Philippines,', u'March']
[u'DAVAO,', u'Philippines,', u'March', u'4']
[u'DAVAO,', u'Philippines,', u'March', u'4', u'(AFP)']
[u'DAVAO,', u'Philippines,', u'March', u'4', u'(AFP)', u'At']
[u'DAVAO,', u'Philippines,', u'March', u'4', u'(AFP)', u'At', u'least']
[u'DAVAO,', u'Philippines,', u'March', u'4', u'(AFP)', u'At', u'least', u'19']
[u'DAVAO,', u'Philippines,', u'March', u'4', u'(AFP)', u'At', u'least', u'19', u'people']
[u'DAVAO,', u'Philippines,', u'March', u'4', u'(AFP)', u'At', u'least', u'19', u'people', u'were']

Once candidates are identified, the code matches the actual mentions and labels
the corresponding candidate as 1 and others as 0. 

For feature processing, each mention_candidate is completed with 'PAD' tokens
up to a total length of 10 tokens. For example,

[u'DAVAO,', u'Philippines,', u'March', u'4', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD']

"""


import editdistance
import os
import json

try:
   import cPickle as pickle
except:
   import pickle

# given a token in a sentence, create phrases of length 1,2,3,...,10=max. 

def mention_candidates(idx, tokenized_sentence):
	max_length = min(len(tokenized_sentence[idx:]), 10)
	all_candidates = []
	for i in range(max_length):
		candidate = tokenized_sentence[idx:idx+1+i] + (9-i)*['PAD']
		all_candidates.append(candidate)
	return all_candidates

def extract_list_of_mentions(document_coref_data):
	list_of_mentions = []
	for entity_dic in document_coref_data['entities']:
		mentions = {}
		for men_dic in entity_dic['mentions']:
			mention_extent = men_dic['entity mention extent']
			mention_head = men_dic['entity mention head']
			start = int(men_dic['entity mention extent start'])
			end = int(men_dic['entity mention extent end'])
			list_of_mentions.append([start,end,mention_extent,mention_head])
	list_of_mentions.sort()
	return list_of_mentions


def mention_label_matcher(mention_candidate, list_of_mentions):
	if len(list_of_mentions) > 0:
		for idx, mention in enumerate(list_of_mentions):
			# exact string matching between mention_candidate and mention
			current_candidate = ' '.join(mention_candidate).strip()
			if editdistance.eval(current_candidate, mention[2].strip()) < 3L and mention[3].strip() in current_candidate:
				list_of_mentions.pop(idx)
				return [mention_candidate,1,mention[3]], list_of_mentions
	return [mention_candidate,0,None], list_of_mentions
	
def mention_document_processor(document_coref_data):
	sentences = document_coref_data['sentences']
	list_of_mentions = extract_list_of_mentions(document_coref_data)
	mention_extent_data = []
	for sent_idx, sent in enumerate(sentences):
		tokenized_sent = sent.strip().split()
		for idx in range(len(tokenized_sent)):
			all_mention_candidates = mention_candidates(idx, tokenized_sent)
			for mention_candidate in all_mention_candidates:
				mention_candidate_data ={}
				mention_candidate_data['sentence'] = sentences[sent_idx]
				mention_candidate_label, list_of_mentions = mention_label_matcher(mention_candidate, list_of_mentions)
				mention_candidate_data['mention_candidate'] = mention_candidate_label
				mention_extent_data.append(mention_candidate_data)
	return mention_extent_data



def mention_data_processor(jsonPath,num_documents=-1):
	json_files = [pos_json for pos_json in os.listdir(jsonPath) if pos_json.endswith('.json')]
	if (".DS_S.json" in json_files):
		json_files.remove(".DS_S.json") # removing the ghost json file from the list if it exists
	mention_data = []
	for js in json_files[:num_documents]:
		with open(os.path.join(jsonPath, js)) as json_file:
			#print "reading json file: " + js
			document_coref_data = json.load(json_file)
			mention_document = {}
			mention_document['document'] = document_coref_data['sentences']
			mention_document['mentions'] = mention_document_processor(document_coref_data)
			mention_data.append(mention_document)
	print('Pickling the mention data')
	mention_data_file = open('coref_data/mention_data.pkl','wb')
	pickle.dump(mention_data, mention_data_file)
	mention_data_file.close()
	return mention_data


# given a mention span, create a fixed length of 10 tokens (mention completed with
# PAD tokens), tokenize each span. 
#def featureProcessMention(mention_data,wordToVecDictFile):
#	wordVecDict = readDictData(wordToVecDictFile)
#	XX,YY=[],[]
#	for document in mention_data:
#		for mention in document['mentions']:
#			X = []
#			for mention in mention['mention_candidate']:
#				Y = mention[1]
#				for token in mention[0]:
#					X = X + isWordUpper(token) + getWordVector(token,wordVecDict)
#			XX.append(X)
#			YY.append(Y)
#	return XX,YY


		 







