import numpy as np

class actFunc(object):
	'''
		this class contains a set of activation functions, including linear, sigmoid, tanh, relu, and their derivation respectively
	'''
	@classmethod
	def getDerivation(clsObj, func):
		funcPrime = clsObj.linearPrime if func is clsObj.linear else\
					clsObj.sigmoidPrime if func is clsObj.sigmoid else\
					clsObj.tanhPrime if func is clsObj.tanh else\
					clsObj.reluPrime if func is clsObj.relu else None
		return funcPrime

	@staticmethod
	def linear(z):
		return z

	@staticmethod
	def linearPrime(z):
		return 1

	@staticmethod
	def sigmoid(z):
		return 0.5 * (1 + np.tanh(0.5 * z))

	@staticmethod
	def sigmoidPrime(z):
		'''derivation of sigmonid'''
		a = 0.5 * (1 + np.tanh(0.5 * z))
		return a * (1.0 - a)

	@staticmethod
	def tanh(z):
		return np.tanh(z)

	@staticmethod
	def tanhPrime(z):
		a = np.tanh(z)
		return 1.0 - a**2

	@staticmethod
	def relu(z):
		z[z < 0] = 0
		return z

	@staticmethod
	def reluPrime(z):
		return np.where(z < 0, 0, 1)
