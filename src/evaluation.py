#this function could calculate overall accuracy, precision,recall and fscore based on three parameters, predication, reference, verdim which is the dimension of one-hot vector,
#and its output is a list, i.e. result, the first element is the overall accuracy, i.e. result[0], the second element is a list of precision, the ith element in this vector 
#corresponds the precision of the ith NER. the thrid and fourth element are recall and fsocre whic hare similar to precision list. 
#main funtion is just for test.
def eval(predication, reference,vecdim):
    samplenum=len(predication)
    same=0
    accuracy=0.0
    for i in range(samplenum):
        if predication[i]==reference[i]:
            same+=1
    accuracy=float(same)/float(samplenum)
    ##calucate recall
    recall=[]
    total=[0]*(vecdim)
    same=[0]*(vecdim)
        
    for j in range(samplenum):
        oneposition=reference[j]
        total[oneposition]+=1
        if predication[j]==oneposition:
            same[oneposition]+=1 
    for i in range(len(total)):
        recall.append(float(same[i])/float(total[i]))
    #calcualte precision
    precision=[]
    total=[0]*(vecdim)
    same=[0]*(vecdim)
    for j in range(samplenum):
        oneposition=predication[j]
        total[oneposition]+=1
        if reference[j]==oneposition:
            same[oneposition]+=1
    fsocre=[]    
    for i in range(len(total)):
        precision.append(float(same[i])/float(total[i]))
        fsocre.append(2*precision[i]*recall[i]/(precision[i]+recall[j]))
    result=[]
    result.append(accuracy)
    result.append(precision)
    result.append(recall)
    result.append(fsocre)
    return result
if __name__ == "__main__":
    a=[1,2,3,0]
    b=a
    result=eval(a,b,4)
    print result