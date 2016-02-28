#############################
##    copyright: Hongyu    ##
##      Assign POS Tags    ##
##  POS Tags As Embedding  ##
#############################

tagset = ['PRP$', 'VBG', 'VBD', '``', 'VBN', ',', "''", 'VBP', 'WDT', 'JJ', 'WP', 'VBZ',
          'DT', '#', 'RP', '$', 'NN', 'FW', 'POS', '.', 'TO', 'PRP', 'RB', '-LRB-', ':',
          'NNS', 'NNP', 'VB', 'WRB', 'CC', 'LS', 'PDT', 'RBS', 'RBR', 'CD', 'EX', 'IN',
          'WP$', 'MD', 'NNPS', '-RRB-', 'JJS', 'JJR', 'SYM', 'UH']
#print(len(tagset))
tagsetLen = 45

def readSen():
    f = open("/Users/Ann/Documents/Stanford POS Tagger/reader-output.txt", "r")
    g = open("/Users/Ann/Documents/Stanford POS Tagger/test.txt", "w")
    sent = []
    for line in f:
        if (line == "\n"):
            sentString = " ".join(sent)
            print >> g, sentString
            sent = []
        else:
            word = line.split("\t")[0]
            sent.append(word)
    if (sent != []):
        sentString = " ".join(sent)
        print >> g, sentString
    f.close()
    g.close()

def posTagging():
    f = open("/Users/Ann/Desktop/baseline/sample-tagged.txt", "r")
    g = open("/Users/Ann/Desktop/baseline/word_pos_Embedding.txt", "w")
    # posSet = set()
    for sen in f:
        wordTagList = sen.split()
        for wordTagPair in wordTagList:
            splitList = wordTagPair.split("_")
            tag = splitList.pop()
            #posSet.add(tag)
            word = "_".join(splitList)
            tagIndex = tagset.index(tag)
            tagEmbedding = ["0"]*tagsetLen
            tagEmbedding[tagIndex] = "1"
            tagEmbeddingString = ",".join(tagEmbedding)
            print >> g, word + "\t" + tagEmbeddingString
        print >> g, "\n"
    f.close()
    g.close()

def mergeSen():
    f = open("/Users/Ann/Desktop/baseline/traindata.txt", "r")
    g = open("/Users/Ann/Desktop/baseline/input.txt", "w")
    para = ""
    for line in f:
        seq = line.split()
        sen = " ".join(seq)
        if (para == ""):
            para = sen
        else:
            para = para + " " + sen
    print >> g, para
    f.close()
    g.close()

if __name__ == "__main__":
    #mergeSen()
    # POS Tagging
    # Terminal
    # java -mx300m -classpath stanford-postagger.jar:lib/* edu.stanford.nlp.tagger.maxent.MaxentTagger
    # -model models/english-bidirectional-distsim.tagger -textFile sample-input.txt > sample-tagged.txt
    posTagging()
