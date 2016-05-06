"""
Mention data processing

(1) matches all the mentions to the sentences where 
they appear

(2) creates candidates for the mention model

For each token in a sentence, word phrases of lengths 1,2,...,10
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
import collections
# given a token in a sentence, create phrases of length 1,2,3,...,10=max. 



def extract_list_of_mentions(document_coref_data):
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






def find_mention(sentence, mention_zero):
	#for idx,sent in enumerate(sentences):
	#	if str(mention_zero[2]) in str(sent):
	#		return str(sent).find(str(mention_zero[2])), idx
	return str(sentence).find(str(mention_zero[2]))

def mention_cleaner(list_of_mentions):
	new_list = []
	for mention in list_of_mentions:
		new_list.append(mention[2:])
	return new_list

def sent_divider(length, list_of_mentions):
	if len(list_of_mentions) == 0:
		return [], []
	else:
		for idx, mention in enumerate(list_of_mentions):
			if mention[0] > length:
				return list_of_mentions[:idx], list_of_mentions[idx:]
	return list_of_mentions, [] 


def mentions_in_sents(list_of_mentions, sentences):
	"""
	given a list of documents and sentences it comes from, 
	returns a list of dictionaries with a sentence and the mentions
	within the sentence
	"""
	data = []
	lengths = [len(sent) for sent in sentences]
	list_of_mentions = prune_absent_mentions(list_of_mentions,sentences)
	mention_zero_index = find_mention(sentences[0], list_of_mentions[0])
	#if mention_zero_sent > 0:
	#	sentences = sentences[mention_zero_sent:]
	#	lengths = lengths[mention_zero_sent:]
	first_diff = list_of_mentions[0][0] - mention_zero_index 
	shifted_list_of_mentions = [(a-first_diff,b-first_diff,mention_extent,mention_head,h,t) for (a,b,mention_extent,mention_head,h,t) in list_of_mentions]
	for idx, diff in enumerate(lengths): 
		sentence_data = {}
		sentence_data['sentence'] = sentences[idx]
		sent_prev, list_of_mentions = sent_divider(diff, shifted_list_of_mentions)
		sentence_data['mentions'] = sent_prev
		if len(list_of_mentions) > 0:
			shifted_list_of_mentions = [(a-diff-2, b-diff-2, mention_extent,mention_head,h,t) for (a,b,mention_extent,mention_head,h,t) in list_of_mentions]					
		else:
			shifted_list_of_mentions = [] 
		data.append(sentence_data)
	return data 

def prune_absent_mentions(list_of_mentions, sentences):
	for mention_idx, mention in enumerate(list_of_mentions):
		mention_matched = False
		for sent in sentences:
			if str(mention[2]) in str(sent):
				mention_matched = True
		if mention_matched == False:
			list_of_mentions.pop(mention_idx)
	return list_of_mentions

def extract_list_of_relations(document_rel_data):
	list_of_relations = []
	for idx, sent_idx in enumerate(document_rel_data):
		list_of_relations = list_of_relations + document_rel_data[sent_idx]["relations"]
	return list_of_relations 

def mention_map(mention_coord, list_of_mentions ):
	for mention in list_of_mentions:
		if (mention[2],mention[3]) == mention_coord:
			return mention
	return None 

def assign_relations1(sent_with_mentions, list_of_relations):
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


def assign_relations(sent_with_mentions,list_of_relations):
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


def mention_sentence_matcher(document_coref_data, document_rel_data):
	list_of_mentions = extract_list_of_mentions(document_coref_data)
	mentions_in_sents_data = mentions_in_sents(list_of_mentions, document_coref_data['sentences'])
	list_of_relations = extract_list_of_relations(document_rel_data)
	for idx, sent_data in enumerate(mentions_in_sents_data):
		try:
			sent_data["relations"] = assign_relations(sent_data,list_of_relations)
			#sent_idx = idx + 1
			#sent_data["relations"] = document_rel_data[str(sent_idx)]["relations"]
		except:
			sent_data["relations"] = []
	return mentions_in_sents_data


def mention_meta_data_processor(coref_jsonPath, rel_jsonPath):
	json_files = [pos_json for pos_json in os.listdir(coref_jsonPath) if pos_json.endswith('.json')]
	if (".DS_S.json" in json_files):
		json_files.remove(".DS_S.json") # removing the ghost json file from the list if it exists
	meta_data = []
	problems_rel = []
	problems_coref = []
	for js in json_files:
		document_rel_data = json.load(open(os.path.join(rel_jsonPath, js)))
		document_coref_data = json.load(open(os.path.join(coref_jsonPath, js)))
		try:
			meta_data.append(mention_sentence_matcher(document_coref_data, document_rel_data))
		except:
			problems_rel.append(document_rel_data)
			problems_coref.append(document_coref_data)
			print "there was a problem with {}".format(js)
	return meta_data 


def find_avg_mention_length(coref_jsonPath):
	list_of_lengths = []
	json_files = [pos_json for pos_json in os.listdir(coref_jsonPath) if pos_json.endswith('.json')]
	if (".DS_S.json" in json_files):
		json_files.remove(".DS_S.json") # removing the ghost json file from the list if it exists
	for js in json_files:
		document_coref_data = json.load(open(os.path.join(coref_jsonPath,js)))
		list_of_mentions = extract_list_of_mentions(document_coref_data)
		list_of_lengths = list_of_lengths + [len(mention[2].strip().split()) for mention in list_of_mentions]
	counter = collections.Counter(list_of_lengths)
	return counter





def splitting_data(X,Y):
    print('Splitting the data into train, validation, test sets with ratio 8:1:1 ...')
    X_train, X_rest, y_train, y_rest = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_rest,y_rest, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test



def mention_candidates(idx, tokenized_sentence):
	max_length = min(len(tokenized_sentence[idx:]), 10)
	all_candidates = []
	for i in range(max_length):
		candidate = tokenized_sentence[idx:idx+1+i] + (9-i)*['PAD']
		all_candidates.append(candidate)
	return all_candidates



def mention_candidate_label(mention_candidate, mention_sentence_data):
	all_mentions = mention_sentence_data['mentions']

	pass





if __name__ == '__main__':
	coref_jsonPath = '../../../data/coref_data'
	rel_jsonPath = '../../../data/relation_data'
	mention_data = mention_meta_data_processor(coref_jsonPath, rel_jsonPath)
	pickle.dump(mention_data, open('../../data/mention.pkl','wb'))


