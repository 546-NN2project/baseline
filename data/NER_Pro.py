import os
output="/home/shawn/Desktop/traindata.txt"
root="/home/shawn/Desktop/NER_ACE_FINE"
file=open(output,'a')
num=1
for lists in os.listdir("/home/shawn/Desktop/NER_ACE_FINE"): 
    file=open(output,'a')
    path = os.path.join(root, lists)
    num+=1
    f=open(path)
    word=[]
    NER=[]
    line=f.readline()
    while line:
       
        split=line.split()
        if len(split) < 5:
            line=f.readline()
            continue
       
        word.append(split[5])
        NER.append(split[0])
        line=f.readline()
    f.close()
    for i in range(len(word)):
        if word[i]=='.':
           
            file.write(word[i]+'\n')
        else:
            file.write(word[i]+' ')
    file.close()
