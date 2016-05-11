"""
Feature processing functions

"""
import pandas as pd
import numpy as np
import json, os
from pprint import pprint
from operator import add
import nltk
#from nltk.tag import stanford

##
## All features should be added in this file as a function
## including POS, etc. 
##

def readDictData(wordToVecDictFile):
    """
    Reads in word2vec dictionary
    """
    f = open(wordToVecDictFile, 'r')
    rawData = f.readlines()
    wordVecDict = {}
    for line in rawData:
        line = line.strip().split()
        word = line[0]
        vec = line[1:]
        wordVecDict[word] = np.array(vec, dtype=float)
    return wordVecDict

def getWordVector(word, wordVecDict):
    """
    Given a word or a PAD vector, returns the word2vec vector or the zero vector
    """
    if word != 'PAD' and word.lower() in wordVecDict:
        return list(wordVecDict[word.lower()])
    return list(np.zeros_like(wordVecDict['hi']))


def ContextTokens(tokenized_sentence, token, window_size):
    """
    Given a token in a sentence, returns (window_size) tokens before and after. 

    :param tokenized_sentence: list of tokens that make up a sentence
    :param token: the current token for which we want the context words
    :window_size: how many words before and after the current token to take

    Output: list of tokens of size 2*(window_size)
    """
    token_pos = tokenized_sentence.index(token) #position of the token in the sentence
    n_sentence = len(tokenized_sentence) 
    #print token_pos, n_sentence
    frontContext_tokens = []
    backContext_tokens = []
    for i in range(window_size):
        current_front_vector = token_pos - window_size + i
        #print "current_front_vector is %d" %current_front_vector
        
        if current_front_vector < 0 :
            frontContext_tokens.append('PAD')
        else:
            frontContext_tokens.append(tokenized_sentence[current_front_vector])
            #print "current front token is %s" %tokenized_sentence[current_front_vector]
            
        current_back_vector = token_pos + i + 1
        #print "current_back_vector is %d" %current_back_vector
        
        if current_back_vector >= n_sentence:
            backContext_tokens.append('PAD')
        else:
            backContext_tokens.append(tokenized_sentence[current_back_vector])
            #print "current back token is %s" %tokenized_sentence[current_back_vector]
        #window_size = window_size - 1
    return frontContext_tokens + backContext_tokens

# isUpper
def isWordUpper(word):
    if word[0].isupper():
        return [1]
    else:
        return [0]

# label vector given by an index
def getLabelIndex(label):
    labList = ['B-Contact-Info','B-Crime','B-FAC','B-GPE','B-Job-Title','B-LOC','B-Numeric','B-ORG','B-PER','B-Sentence','B-TIME','B-VEH','B-WEA','I-Contact-Info','I-Crime','I-FAC','I-GPE','I-Job-Title','I-LOC','I-Numeric','I-ORG','I-PER','I-Sentence','I-TIME','I-VEH','I-WEA','O']
    return labList.index(label)


def binaryFeatureProcess(DataFile,LabelFile,wordToVecDictFile,window_size):
    """
    Given data and label files, prepare training data for the NER binary classifier
    """
    rawTextData = open(DataFile, 'r').readlines()
    rawLabelData = open(LabelFile, 'r').readlines()
    wordVecDict = readDictData(wordToVecDictFile)
    XX = []
    YY = []
    ln = 1
    for line,label in zip(rawTextData,rawLabelData):
        #print "reading line %d" %ln
        #tokenized_sentence = nltk.word_tokenizer(line)
        tokenized_sentence = line.strip().split()
        #POS_tags = st.tag(tokenized_sentence)
        tokenized_y = label.strip().split()
        if len(tokenized_sentence) != len(tokenized_y):
            raise NotImplementedError
        prevLabel = 0
        for token, y in zip(tokenized_sentence,tokenized_y):
            X = isWordUpper(token) + getWordVector(token,wordVecDict)
            context = ContextTokens(tokenized_sentence,token,window_size)
            for context_token in context:
                X = X + getWordVector(context_token,wordVecDict)
            prevLabelList = [prevLabel]
            X = X + prevLabelList
            #print len(X),y,getLabelIndex(y)
            XX.append(X)
            if (y=='O'):
                yNum = 0
            else:
                yNum = 1
            YY.append(yNum)
            prevLabel = yNum
        ln += 1
    return XX, YY



def featureProcess(DataFile,LabelFile,wordToVecDictFile,window_size):
    """
    Given data and label files, returns a list of lists of length the
    number of tokens in the data. For each token, outputs a feature vector
    and a label int (corresponding to the index)
    """
    rawTextData = open(DataFile, 'r').readlines()
    rawLabelData = open(LabelFile, 'r').readlines()
    wordVecDict = readDictData(wordToVecDictFile)
    #st = StanfordPOSTagger('/Users/bolor/stanford-postagger-2015-12-09/models/english-bidirectional-distsim.tagger','/Users/bolor/stanford-postagger-2015-12-09/stanford-postagger.jar')
    # I don't think we want to hard code the dimension of X since we will add
    # more features in the future
    #feature_size = (2*window_size + 1)*word_vec_dim + 1
    XX = []
    YY = []
    #XX = np.zeros([1,feature_size])
    #YY = np.zeros([1,1])
    ln = 1
    for line,label in zip(rawTextData,rawLabelData):
        #print "reading line %d" %ln
        #tokenized_sentence = nltk.word_tokenizer(line)
        tokenized_sentence = line.strip().split()
        #POS_tags = st.tag(tokenized_sentence)
        tokenized_y = label.strip().split()
        if len(tokenized_sentence) != len(tokenized_y):
            raise NotImplementedError
        prevLabel = 'O'
        for token, y in zip(tokenized_sentence,tokenized_y):
            X = isWordUpper(token) + getWordVector(token,wordVecDict)
            context = ContextTokens(tokenized_sentence,token,window_size)
            for context_token in context:
                X = X + getWordVector(context_token,wordVecDict)
            prevLabelList = [0]*27
            prevLabelList[getLabelIndex(prevLabel)] = 1
            X = X + prevLabelList
            #print len(X),y,getLabelIndex(y)
            XX.append(X)
            YY.append(getLabelIndex(y))
            prevLabel = y
        ln += 1
    return XX, YY

def testFeatureProcess(DataFile,wordToVecDictFile,window_size):
    """
    Load the test data, get the feature
    """
    rawTextData = open(DataFile, 'r').readlines()
    wordVecDict = readDictData(wordToVecDictFile)
    XX = []
    ln = 1
    for line in rawTextData:
        #print "reading line %d" %ln
        #tokenized_sentence = nltk.word_tokenizer(line)
        tokenized_sentence = line.strip().split()
        for token in tokenized_sentence:
            X = isWordUpper(token) + getWordVector(token,wordVecDict)
            context = ContextTokens(tokenized_sentence,token,window_size)
            for context_token in context:
                X = X + getWordVector(context_token,wordVecDict)
            XX.append(X)
        ln += 1
    return XX

def featureProcessRel(relFile,wordToVecDictFile,wvecdim,pctNull):
    """
    Given relation data in json files, returns the feature set and labels
    """
    # feature data and label data
    XX = []
    YY = []    
    wordVecDict = readDictData(wordToVecDictFile)
    
    #json_files = [pos_json for pos_json in os.listdir(jsonPath) if pos_json.endswith('.json')]
    #if (".DS_S.json" in json_files):
    #    json_files.remove(".DS_S.json") # removing the ghost json file from the list if it exists
    # load the relations mention file
    data = pickle.load(open(relFile))
    passRel = 0
    totalRel = 0
    for dicts in range(0,len(data)):
        for sent in range(0,len(data[dicts])):
            sentStr = data[dicts][sent]['sentence']
            #sentStrTkn = sentStr.replace(',',' ').replace("'",' ').replace(".",' ').strip().split() #tokenized sentence
            sentStrTkn = re.sub(r"[^\w ]", " ", sentStr).strip().split()

            n_mentions = len(data[dicts][sent]['mentions'])
            # create the pairs of mentions, and later create NULL lablels for the pairs that dont have any relation
            mention_pairs = []            
            for b in itertools.permutations(range(n_mentions),2):
                mention_pairs.append(b)
            
            mention_pairs_exist = []
            for rel in range(0,len(data[dicts][sent]['relations'])):
                line1 = data[dicts][sent]['relations'][rel]['arg1_string']
                line2 = data[dicts][sent]['relations'][rel]['arg2_string']    
                if (line1 in sentStr) and (line2 in sentStr):
                    l1found = 0
                    l2found = 0
                    for mention in range(n_mentions):
                        if line1 in data[dicts][sent]['mentions'][mention]:
                            line1 = data[dicts][sent]['mentions'][mention][3]
                            mention_1 = mention
                            l1found = 1
                        if line2 in data[dicts][sent]['mentions'][mention]:
                            line2 = data[dicts][sent]['mentions'][mention][3]
                            mention_2 = mention                            
                            l2found = 1
                        if l1found and l2found:
                            mention_pairs_exist.append((mention_1,mention_2))
                            break
                            
                    sbStr = 0
                    if line1 in line2:
                        sbStr = 1
                    elif line2 in line1:
                        sbStr = 1
                    print 'dicts: '+ str(dicts) + ' sent: ' + str(sent) + ' relation: ' + str(rel) 
                    #tokenized_sentence1 = line1.replace(',',' ').replace("'",' ').replace(".",' ').strip().split()
                    #tokenized_sentence2 = line2.replace(',',' ').replace("'",' ').replace(".",' ').strip().split()
                    #tokenized_sentence1 = re.findall(r"[\w']+", line1)
                    #tokenized_sentence2 = re.findall(r"[\w']+", line2)
                    #tokenized_sentence1 = re.sub(r"[^\w ]", " ", line1).strip().split()
                    #tokenized_sentence2 = re.sub(r"[^\w ]", " ", line2).strip().split()
                    totalRel += 1
                    #if (tokenized_sentence1[0] in sentStrTkn) and (tokenized_sentence2[0] in sentStrTkn):
                    if (line1 in sentStrTkn) and (line2 in sentStrTkn):
                        passRel += 1
                        #s11 = sentStrTkn.index(tokenized_sentence1[0])
                        s11 = sentStrTkn.index(line1)
                        #s12 = sentStrTkn.index(tokenized_sentence1[len(tokenized_sentence1)-1])
                        #s12 = s11 + len(tokenized_sentence1)-1
                        #s21 = sentStrTkn.index(tokenized_sentence2[0])
                        s21 = sentStrTkn.index(line2)
                        #s22 = sentStrTkn.index(tokenized_sentence2[len(tokenized_sentence2)-1])
                        #s22 = s21 + len(tokenized_sentence2)-1
                        
                        # get the token indices for 'in between' token between two mentions
                        #sidx,eidx = getBegEnd([s11,s12],[s21,s22])
                        #gapSent = sentStrTkn[sidx:eidx+1]
                        if s11 < s21:
                            gapSent = sentStrTkn[s11:s21+1]
                        else:
                            gapSent = sentStrTkn[s21:s11+1]
                        
                        posdiff = abs(s11-s21) #int(data[dicts][sent][rel]['arg2_start']) - int(data[dicts][sent][rel]['arg1_end'])
                        
                        # add word vectors for tokens of first entity
                        tempX1 = [0]*wvecdim
                        upval1 = [0]
                        #for token in tokenized_sentence1:
                        tempX1 = map(add, tempX1, getWordVector(line1,wordVecDict))
                        if line1.isupper():
                            upval1 = isWordUpper(line1)
                        #XX.append(tempX.append(upval))
                        
                        # get a word vector which is sum of all the word vectors between two mentions 
                        tempX1X2 = [0]*wvecdim
                        for gtoken in gapSent:
                            tempX1X2 = map(add, tempX1X2, getWordVector(gtoken,wordVecDict))
                        
                        # add word vectors for tokens of second entity
                        tempX2 = [0]*wvecdim
                        upval2 = [0]
                        #for token in tokenized_sentence2:
                        tempX2 = map(add, tempX2, getWordVector(line2,wordVecDict))
                        if line2.isupper():
                            upval2 = isWordUpper(line2)
                        
                        X = [posdiff] + tempX1 + upval1 + tempX2 + upval2 + tempX1X2 + [sbStr]
                        XX.append(X)
                        #print XX
                        #print getLabelIndexRel(data[dicts][sent]['relations'][rel]['relation_type'])
                        
                        YY.append(getLabelIndexRel(data[dicts][sent]['relations'][rel]['relation_type']))
            # add some NULL relations for each sentence
            num_null = 0
            
            for pair in mention_pairs:
                if num_null > int(math.ceil(pctNull*len(mention_pairs_exist))):        
                    break
                if not pair in mention_pairs_exist:
                    line1 = data[dicts][sent]['mentions'][pair[0]][3]
                    line2 = data[dicts][sent]['mentions'][pair[1]][3]
                    
                    sbStr = 0
                    if line1 in line2:
                        sbStr = 1
                    elif line2 in line1:
                        sbStr = 1
                        
                    # check if the mention heads exist in the sentence
                    if (line1 in sentStrTkn) and (line2 in sentStrTkn):
                        #passRel += 1
                        #s11 = sentStrTkn.index(tokenized_sentence1[0])
                        s11 = sentStrTkn.index(line1)
                        #s12 = sentStrTkn.index(tokenized_sentence1[len(tokenized_sentence1)-1])
                        #s12 = s11 + len(tokenized_sentence1)-1
                        #s21 = sentStrTkn.index(tokenized_sentence2[0])
                        s21 = sentStrTkn.index(line2)
                        #s22 = sentStrTkn.index(tokenized_sentence2[len(tokenized_sentence2)-1])
                        #s22 = s21 + len(tokenized_sentence2)-1
                        
                        # get the token indices for 'in between' token between two mentions
                        #sidx,eidx = getBegEnd([s11,s12],[s21,s22])
                        #gapSent = sentStrTkn[sidx:eidx+1]
                        if s11 < s21:
                            gapSent = sentStrTkn[s11:s21+1]
                        else:
                            gapSent = sentStrTkn[s21:s11+1]
                        
                        posdiff = abs(s11-s21) #int(data[dicts][sent][rel]['arg2_start']) - int(data[dicts][sent][rel]['arg1_end'])
                        
                        # add word vectors for tokens of first entity
                        tempX1 = [0]*wvecdim
                        upval1 = [0]
                        #for token in tokenized_sentence1:
                        tempX1 = map(add, tempX1, getWordVector(line1,wordVecDict))
                        if line1.isupper():
                            upval1 = isWordUpper(line1)
                        #XX.append(tempX.append(upval))
                        
                        # get a word vector which is sum of all the word vectors between two mentions 
                        tempX1X2 = [0]*wvecdim
                        for gtoken in gapSent:
                            tempX1X2 = map(add, tempX1X2, getWordVector(gtoken,wordVecDict))
                        
                        # add word vectors for tokens of second entity
                        tempX2 = [0]*wvecdim
                        upval2 = [0]
                        #for token in tokenized_sentence2:
                        tempX2 = map(add, tempX2, getWordVector(line2,wordVecDict))
                        if line2.isupper():
                            upval2 = isWordUpper(line2)
                        
                        X = [posdiff] + tempX1 + upval1 + tempX2 + upval2 + tempX1X2 + [sbStr]
                        XX.append(X)
                        #print XX
                        #print getLabelIndexRel(data[dicts][sent]['relations'][rel]['relation_type'])
                        
                        YY.append(getLabelIndexRel(u'NONE'))

                        num_null += 1
                        

    
    print 'failed = ' + str(totalRel-passRel)
    print 'total = ' + str(totalRel)    
    return XX, YY

# label index for relation type
def getLabelIndexRel(label):
    labList = [u'PHYS', u'PART-WHOLE', u'ART', u'ORG-AFF', u'PER-SOC', u'GEN-AFF',u'NONE']
    return labList.index(label)



def POSlist(tokenized_sent):
    tagged_sent = nltk.pos_tag(tokenized_sent)
    POSlist = [tag[1] for tag in tagged_sent]
    return POSlist 

def POStagger(token_idx, tokenized_sent):
    tagged_sent = nltk.pos_tag(tokenized_sent)
    return tagged_sent[token_idx][1]


def getLabelPOS(label):
    NN_list = ["NN","NNP","NNPS","NNS","PRP","PRP$","WDT","WP","WP$"]
    if label in NN_list:
        return [1]
    else:
        return [0]

def featureProcess_mention_head(mention_data,wordVecDic,window_size):
    """
    input: mention data obtained from mention_meta_data_processor
    """
    XX = []
    YY = []
    for document_data in mention_data:
        for sent_data in document_data:
            tokenized_sent = sent_data['sentence'].strip().split()
            POS_sent = POSlist(tokenized_sent)
            labels = find_mention_head_labels(sent_data)
            if len(tokenized_sent) != len(labels):
                raise NotImplementedError
            for idx, token in enumerate(tokenized_sent):
                X = isWordUpper(token) + getWordVector(token,wordVecDic) + getLabelPOS(POS_sent[idx])
                context = ContextTokens(tokenized_sent, token, window_size)
                for context_token in context:
                    X = X + getWordVector(context_token, wordVecDic)
                XX.append(X)
                YY.append(labels[idx])
    if len(XX) != len(YY):
        raise NotImplementedError
    return XX,YY

def find_mention_head_labels(mention_sentence_data):
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

