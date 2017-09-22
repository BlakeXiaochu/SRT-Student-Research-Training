import sys
sys.path.append('../')
import numpy as np
from MLP import *

samples = np.array([[1.0], [0.5], [0.3]])
labels = np.array([[0], [1], [0]])


model = MLP([3, 5, 3])

for i in range(10):
	a = model.feedforward(samples)
	error = model.errCompute(a, labels)
	print(error)

	model.update((samples, labels), 3.0)