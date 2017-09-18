import numpy as np

class MLP(object):
	"""
		mutiple layers perceptron class
		layers including input layer and output layer, so neuronNums should be greater than 2.
	"""
	__slots__ = {'neuronNums', 'layerNum', 'layers', 'biases', 'weights'}
	def __init__(self, neuronNums, biases = None, weights = None):
		self.neuronNums = tuple(neuronNums)
		self.layerNum = len(neuronNums)

		if biases is None:
			self.biases = tuple([np.random.randn(i, 1) for i in self.neuronNums[1:]])
		else: 
			self.biases = biases

		if weights is None:
			self.weights = tuple([np.random.randn(i, j) for i, j in zip(self.neuronNums[:-1], self.neuronNums[1:])])
		else: 
			self.weights = weights

		#construct mutiple layers perceptron(input layer is absent)
		self.layers = tuple([layer(num, bias, weight, 'sigmoid') for num, bias, weight in zip(neuronNums[1:], self.biases, self.weights)])


class layer(object):
	"""layer class"""
	def __init__(self, neuronNum, biases, weights, activateType):
		if not isinstance(neuronNum, int):
			raise TypeError('argument 1: type int is required.')
		if not isinstance(biases, np.ndarray):
			raise TypeError('argument 2: numpy.ndarray is required.')
		if not isinstance(weights, np.ndarray):
			raise TypeError('argument 3: numpy.ndarray is required.')
		#valid activateType: linear, sigmoid, tanh, relu.
		if not isinstance(activateType, str):
			raise TypeError('argument 4: type str is required.')

		if (biases.shape[0] != neuronNum) or (weights.shape[1] != neuronNum):
			raise ValueError("weights or biases' shape is invalid.")

		self.neuronNum = neuronNum
		self.biases = biases
		self.weights = weights
		self.activateType = activateType


	#update weights
	def update(self):
		pass

	#activate neurons
	def activate(self):
		pass

