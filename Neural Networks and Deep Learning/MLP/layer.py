from funcKit import *
import numpy as np

class neurLayer(object):
	"""basic layer class(hidden layer)"""
	__slots__ = {'neuronNum', 'biases', 'weights', 'activateFunc', 'backpropFunc'}
	def __init__(self, neuronNum):
		self.neuronNum = neuronNum
		self.biases = None
		self.weights = None
		self.activateFunc = None
		self.backpropFunc = None
		# self.activateFunc = self.linear if activateType == 'linear' else\
		# 					self.sigmoid if activateType == 'sigmoid' else\
		# 					self.tanh if activateType == 'tanh' else\
		# 					self.relu if activateType == 'relu' else None
		# self.backpropFunc = self.linearPrime if activateType == 'linear' else\
		# 					self.sigmoidPrime if activateType == 'sigmoid' else\
		# 					self.tanhPrime if activateType == 'tanh' else\
		# 					self.tanhPrime if activateType == 'relu' else None


	#initialize the biases, weights and activation function
	def initParams(self, biases, weights, activateFunc):
		self.biases = biases
		self.weights = weights
		self.activateFunc = activateFunc
		self.backpropFunc = actFunction.getDerivation(activateFunc)
		
		if self.backpropFunc is None:
			raise ValueError('invalid activation type.')


	#activate neurons
	def activate(self, x):
		innerProd = np.dot(self.weights, x) + self.biases
		output = self.activateFunc(innerProd)

		return innerProd, output


	#backpropagation, this layer's error known, return the prime layer's error
	# def backprop(self, deltaIn, z, a):

	def backprop(self, deltaIn, z, a):
		deltaOut = None if (z is None) else np.dot(self.weights.T, deltaIn) * self.backpropFunc(z)
		Cb = np.mean(deltaIn, axis = 1)
		Cb.shape = (Cb.shape[0], 1)

		deltaSize = deltaIn.shape
		deltaIn.shape = list(deltaSize) + [1]
		deltaIn = np.swapaxes(deltaIn, 1, 2)

		aT = a.reshape((1, a.shape[0], -1))
		Cw = np.mean(deltaIn * aT, axis = 2)
		deltaIn.shape = deltaSize

		return deltaOut, Cb, Cw
	
	#update weights and biases
	def update(self, delta_w, delta_b):
		self.weights += delta_w
		self.biases += delta_b



class outputLayer(neurLayer):
	'''output layer class'''
	__slots__ = {'neuronNum', 'biases', 'weights', 'activateFunc', 'backpropFunc', 'lossFunc', 'lossFuncGrads'}
	def __init__(self, neuronNum):
		super(outputLayer, self).__init__(neuronNum)

	#initialize the biases, weights and activation function
	def initParams(self, biases, weights, activateFunc, lossFunc):
		self.biases = biases
		self.weights = weights
		self.activateFunc = activateFunc
		self.backpropFunc = actFunction.getDerivation(activateFunc)
		self.lossFunc = lossFunc
		self.lossFuncGrads = lossFunction.getGradient(lossFunc)
		
		if self.backpropFunc is None:
			raise ValueError('invalid activation type.')
		if self.lossFuncGrads is None:
			raise ValueError('invalid loss function type.')


	#compute loss
	def lossCompute(self, a, labels):
		return self.lossFunc(a, labels)

	#compute output layer's delta
	def deltaCompute(self, z, a, labels):
		delta = self.backpropFunc(z) * self.lossFuncGrads(a, labels)
		return delta
		