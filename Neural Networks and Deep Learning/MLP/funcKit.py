import numpy as np

class actFunction(object):
	'''
		this class contains a set of activation functions, including linear, sigmoid, tanh, relu, and their derivation respectively
	'''
	@classmethod
	def getDerivation(clsObj, func):
		funcPrime = clsObj.linearPrime if func is clsObj.linear else\
					clsObj.sigmoidPrime if func is clsObj.sigmoid else\
					clsObj.tanhPrime if func is clsObj.tanh else\
					clsObj.reluPrime if func is clsObj.relu else\
					clsObj.softmaxPrime if func is clsObj.softmax None
		return funcPrime

	@classmethod
	def funcTypeEncode(clsObj, func):
		funcType =  0 if func is clsObj.linear else\
					1 if func is clsObj.sigmoid else\
					2 if func is clsObj.tanh else\
					3 if func is clsObj.relu else\
					4 if func is clsObj.softmax else None
		return funcType

	@classmethod
	def funcTypeDecode(clsObj, funcType):
		func = 	clsObj.linear if funcType is 0 else\
				clsObj.sigmoid if funcType is 1 else\
				clsObj.tanh if funcType is 2 else\
				clsObj.relu if funcType is 3 else\
				clsObj.softmax if funcType is 4 else None
		return func

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
		return np.where(z <= 0, 0, 1)

	@staticmethod
	def softmax(z):
		zMax = np.max(a, axis = 0)
		zMax.shape = (-1, zMax.shape[0])
		#unnormalized probability
		p = np.exp(z - zMax)
		return p / (np.reshape(np.sum(p, axis = 0)), zMax.shape)

	#unfinished
	@staticmethod
	def softmaxPrime(z):
		pass


class lossFunction(object):
	'''
		this class contains a set of loss functions, including mean square, cross entropy
	'''
	@classmethod
	def getGradient(clsObj, func):
		grads = clsObj.meanSquareGrads if func is clsObj.meanSquare else\
				clsObj.crossEntropyGrads if func is clsObj.crossEntropy else None
		return grads

	@classmethod
	def funcTypeEncode(clsObj, func):
		funcType = 0 if func is clsObj.meanSquare else\
				   1 if func is clsObj.crossEntropy else None
		return funcType

	@classmethod
	def funcTypeDecode(clsObj, funcType):
		func = 	clsObj.meanSquareGrads if funcType is 0 else\
				clsObj.crossEntropyGrads if funcType is 1 else None
		return func

	@staticmethod
	def meanSquare(a, labels):
		error = np.sum((a - labels)**2 / 2.0, axis = 0)
		error = np.mean(error)
		return error

	@staticmethod
	def meanSquareGrads(a, labels):
	#gradient of mean square loss function about results 'a'
		return (a - labels)

	@staticmethod
	def crossEntropy(a, labels):
		error = np.sum((labels + 1e-8) * np.log(a + 1e-8) + (1 - labels + 1e-8) * np.log(1 - a + 1e-8), axis = 0)
		error = np.nan_to_num(error)
		error = -np.mean(error)
		return error

	@staticmethod
	def crossEntropyGrads(a, labels):
	#gradient of cross entropy loss function about results 'a'
		return (a - labels) / (a * (1 - a) + 1e-8)