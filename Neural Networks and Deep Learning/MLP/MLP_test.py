import numpy as np
from MLP import *
from imgLoader import *
import cv2

trainSamples = loadImageSet(0)
trainSamples = trainSamples / 255.0
trainLables = loadLabelSet(0)
trainData = (trainSamples, trainLables)
testSamples = loadImageSet(1)
testSamples = testSamples / 255.0
testLables = loadLabelSet(1)
testData = (testSamples, testLables)

#trainning param
alpha = 1.0
batchSize = 10
epochNum = 50

model = MLP([trainSamples.shape[0], 30, 10])
model.SGD(trainData, epochNum, batchSize, alpha, testData)
