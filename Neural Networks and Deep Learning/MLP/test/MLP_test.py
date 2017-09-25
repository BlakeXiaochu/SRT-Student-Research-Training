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
alpha = 0.1
batchSize = 20
epochNum = 30

model = MLP()
model.initParams([trainSamples.shape[0], 100, 10], activateFunc = actFunction.sigmoid, lossFunc = lossFunction.crossEntropy, regular = True, momentum = True, rLambda = 5.0, miu = 0.3)
model.SGD(trainData, epochNum, batchSize, alpha, testData, monitor = True)
