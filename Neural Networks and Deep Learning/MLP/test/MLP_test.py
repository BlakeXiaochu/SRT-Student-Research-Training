import sys
sys.path.append('../')
import numpy as np
from MLP import *
from imgLoader import *

trainSamples = loadImageSet(0)
trainSamples = trainSamples / 255.0
trainLables = loadLabelSet(0)
trainData = (trainSamples, trainLables)
testSamples = loadImageSet(1)
testSamples = testSamples / 255.0
testLables = loadLabelSet(1)
testData = (testSamples, testLables)

#trainning param
alpha = 0.05
batchSize = 10
epochNum = 60

model = MLP([trainSamples.shape[0], 100, 10], activateFunc = actFunc.sigmoid, regular = True, rLambda = 5.0)
model.SGD(trainData, epochNum, batchSize, alpha, testData)
