from keras.models import Sequential
from keras.layers import LSTM, TimeDistributedDense, RepeatVector,Dense,Activation,BatchNormalization,Dropout,Reshape
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.cross_validation import train_test_split
def transfer(sttrlist):
    result=[]
    for i in sttrlist:
        temp=[]
        for j in range(28):
            temp.append(0.0)
        temp[int(float(i))]=1.0
        result.append(temp)
    
    return temp
def datatransfer(strlist):
    result=[]
    for i in strlist:
        temp=[]
        spl=i.split()
        for j in spl:
            temp.append(float(j))
        result.append(temp)
    return temp
trainlines=open('traindata').readlines()
labellines=open('trainlabel').readlines()
totaldata=[]
i=0
for line in trainlines:
    linesplit=line.strip().split(',')
    totaldata.append(linesplit)
totaltrain=[]

while i<len(totaldata):
    subtrain=[]
    subtrain.append(datatransfer(totaldata[i]))
    i+=1
    while i %208!=0:
        subtrain.append(datatransfer(totaldata[i]))
        i+=1
    totaltrain.append(subtrain)
###
i=0
tolabel=[]
for line in labellines:
    linesplit=line.strip().split(',')
    tolabel.append(linesplit)
totallabel=[]

while i<len(tolabel):
    subtrain=[]
    subtrain.append(transfer(tolabel[i]))
    i+=1
    while i %208!=0:
        subtrain.append(transfer(tolabel[i]))
        i+=1
    totallabel.append(subtrain)
x_train,x_rest,y_train,y_rest=train_test_split(totaltrain,totallabel,test_size=0.1,random_state=47)
x_val,x_test,y_val,y_test=train_test_split(x_rest,y_rest,test_size=0.5,random_state=47)
data_dim = 100
timesteps = 100
nb_classes = 28
model = Sequential()
class_weight={0:6548,1:119,2:284,3:81,4:18,5:16,6:26,7:21,8:741,9:213,10:106,11:154,12:111,13:28,14:28,15:37,16:1,17:1,18:12,19:15,20:1,21:3,22:35,23:3,24:3,25:1,26:2,27:33300}
sample_weight=[]
fileread=open('trainlabel')
lines=fileread.readlines()
totalsample=[]
i=0
for line in ((lines)):
    if i%208<100:
        totalsample.append(int(float(line)))
    i+=1
i=0
while i<len(totalsample):
    temp=[]
   # if totalsample[i]==0:
    #    temp.append(10)
   # elif totalsample[i]==27:
    #    temp.append(1)
   # else:
    #    temp.append(500)      
    temp.append(int(1.0/class_weight[totalsample[i]]*class_weight[27]))
    i+=1
    while i%100!=0:
       # if i%208<100
    #    if totalsample[i]==0:
     #       temp.append(10)
      #  elif totalsample[i]==27:
       #     temp.append(100)
      #  else:
       #     temp.append(100)
        temp.append(int(1.0/class_weight[totalsample[i]]*class_weight[27]))
        i+=1
    sample_weight.append(temp)
sample_weight=np.asarray(sample_weight)   
X_sample_weight,X_sample_weight_rest=train_test_split(sample_weight,test_size=0.1,random_state=47)
sample_weight_val,sample_weight_test=train_test_split(X_sample_weight_rest,test_size=0.5,random_state=47)
#X_sample_weight=np.concatenate((X_sample_weight,sample_weight_val))
#model.add(Reshape((100,100),input_shape=(10000,)))
model.add(LSTM(60,return_sequences=True,input_shape=(timesteps, data_dim)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(TimeDistributedDense(28, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',sample_weight_mode="temporal")
totaltrain=np.asarray(totaltrain)
totallabel=np.asarray(totallabel)
print len(x_val)
x_train=np.asarray(x_train)
x_test=np.asarray(x_test)
x_val=np.asarray(x_val)
y_train=np.asarray(y_train)
y_val=np.asarray(y_val)
y_test=np.asarray(y_test)
print y_train.shape
x_train=x_train[:,:100,:]
y_train=y_train[:,:100,:]
print y_train.shape
early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1) 
model.fit(x_train,y_train, callbacks=[early_stop],nb_epoch=300, sample_weight=X_sample_weight,batch_size=100,show_accuracy=True, validation_split=0.1)
with open('yaml','w') as f:
    f.write(model.to_yaml())
model.save_weights('NERmode_weights.h5',overwrite=True)

