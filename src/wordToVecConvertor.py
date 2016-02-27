import numpy as np
'''
This module creates word embeddings from the pre-trained word2vec data
'''
def readDictData(wordToVecDictFile):
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
    if word in wordVecDict:
        return wordVecDict[word]
    return np.zeros_like(wordVecDict['hi'])
    
   
def createVecFeatsLabels(dataFile,labelFile,wordToVecDictFile,conWin):
    fdata = open(dataFile, 'r')
    ldata = open(labelFile, 'r')
    
    rawTextData = fdata.readlines()
    rawLabelData = ldata.readlines()
    wordVecDict = readDictData(wordToVecDictFile)
    vecDim = len(wordVecDict['hi'])
    XX = np.zeros([1,vecDim*(2*conWin+1)])
    for line in rawTextData:
        document = line.strip('.').split()
        Xdoc = np.zeros([len(document),vecDim])
        for i in range(0, len(document)): 
            Xdoc[i] = getWordVector(document[i],wordVecDict) 
        Xdoc = getCtxtCorrVector(Xdoc,vecDim,conWin,len(document))
        XX = np.vstack((XX,Xdoc))
        
    YY = np.zeros([1,21])
    for line in rawLabelData:
        document = line.strip('.').split()
        for i in range(0, len(document)): 
            YY = np.vstack((YY,getLabelVector(document[i]))) 
    XX = np.delete(XX, (0), axis=0)
    YY = np.delete(YY, (0), axis=0)
    return XX,YY
    
def getPaddingVector(nDim):
    return np.zeros(nDim)
    
def getCtxtCorrVector(Xdoc,vecDim,conWin,docLen):
    # get the context corrected vector
    Xdoc = np.vstack((np.zeros([conWin,vecDim]),Xdoc,np.zeros([conWin,vecDim])))
    XdocCorr = np.zeros([docLen,vecDim*(2*conWin+1)])
    for i in range(conWin,docLen-conWin):
        XdocVal = Xdoc[i-conWin]
        for j in range(i-conWin+1,i+conWin+1):
            XdocVal = np.concatenate((XdocVal,Xdoc[j]))
        XdocCorr[i-conWin] = XdocVal
    return XdocCorr
    
def getLabelVector(label):
    labList = ['B-TIME','I-TIME','B-Numeric','I-Numeric','B-Contact-Info','I-Contact-Info','B-PER','I-PER','B-ORG','I-ORG','B-LOC','I-LOC','B-GPE','I-GPE','B-FAC','I-FAC','B-VEH','I-VEH','B-WEA','I-WEA','O']
    vec = np.zeros(len(labList))
    vec[labList.index(label)] = 1
    return vec
    