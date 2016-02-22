#################################
#   Feedforward Network Model   #
# for Named Entity Recognition  #
#    copyright: Hongyu Gong     #
#################################

from ffnet import mlgraph, ffnet
import networkx as NX
import matplotlib.pyplot as plt

# context window size
cxtWinSize = 5
# word vector dimension
wordDim = 125
# input layer size
inputSize = cxtWinSize * wordDim
# hidden layer size
hidSize = 10
# output layer size: seven classes
outSize = 8

# Read input and output from training data
# wait to be implemented
# inputVec = 
# targetVec =

# Read input from test data
# wait to be implemented
# testInputVec = 

# specify network topology
conec = mlgraph((inputSize, hidSize, outSize), biases=False)
net = ffnet(conec)
# training process: inputVec: numpy.ndArray, target: numpy.ndArray
net.train_momentum(inputVec, targetVec, eta=0.2, momentum=0.8, maxiter=10000, disp=0)
# prediction
predProb = net.call(testInputVec)
