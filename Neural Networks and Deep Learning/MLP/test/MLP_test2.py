import sys
sys.path.append('../')
import numpy as np
from MLP import *

samples = np.array([[1.0], [0.5], [0.3]])
labels = np.array([[0], [1], [0]])


model = MLP()
model.initParams([3, 5, 3], activateFunc = actFunction.sigmoid, lossFunc = lossFunction.crossEntropy)

for i in range(20):
	a = model.feedforward(samples)
	error = model.lossCompute(a, labels)
	print(error)

	model.update((samples, labels), 1.0)

model.saveModel()