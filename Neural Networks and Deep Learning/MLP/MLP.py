import numpy as np

class MLP(object):
	"""
		mutiple layers perceptron class
		layers including x layer and output layer, so neuronNums should be greater than 2.
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
			self.weights = tuple([np.random.randn(j, i) for i, j in zip(self.neuronNums[:-1], self.neuronNums[1:])])
		else: 
			self.weights = weights

		#construct mutiple layers perceptron(x layer is absent)
		self.layers = tuple([neurLayer(num, bias, weight, 'sigmoid') for num, bias, weight in zip(neuronNums[1:], self.biases, self.weights)])


	#mini-batch stochastic gradient descent
	def SGD(self, trainData, epochNum, batchSize, alpha, testData = None):
		'''
			trainData: a tuple/list of (samples, labels), in which samples and labels' type are 2-D numpy.ndarray
			epochNum: the number of trainning epochs
			batchSize: the number of trainning samples in each epoch
			alpha: learning rate
			testData: a tuple/list of (samples, labels) for testing. if provided, MLP will eavluate the testing data in each epoch and print results
		'''
		sampleNum = trainData[0].shape[0]
		testNum = testData[0].shape[0]

		for i in range(epochNum):
			#shuffle the trainning data
			randOrder = np.random.shuffle(np.arange(0, sampleNum))
			samples, labels = trainData
			samples, labels = samples[randOrder], labels[randOrder]

			#batch trainning
			for j in range(0, sampleNum - batchSize, step = batchSize):
				batch = (samples[j:j+batchSize, :].T, labels[j:j+batchSize, :].T)
				self.update(batch, alpha)

			#print trainnig progress
			if testData:
				print( 'Epoch %d: error = %.5f' % (i, self.evaluate(testData) / testNum) )
			else:
				print('Epoch %d complete...' % (i))



	def feedforward(self, x):
		for layer in self.layers:
			_, x = layer.feedforward(x)
		return x



	#apply gradient descent, and update the network weights when trainning with a batch
	#alpha is the learning rate
	def update(self, batch, alpha):
		samples, labels = batch

		#feedforward
		a = samples
		z_list = [None]
		a_list = [a]
		for layer in self.layers:
			z, a = layer.feedforward(a)
			z_list.append(z)
			a_list.append(a)


		#last layer's error
		aL, zL = a_list[-1], z_list[-1]
		delta = (aL - labels) * (self.layers[-1].sigmoidPrime(zL))

		#backprpagation
		for i in range(1, self.layerNum):
			a, z = a_list[-i - 1], z_list[-i - 1]
			layer = self.layers[-i]
			delta, Cb, Cw = layer.backprop(delta, z, a)
			layer.update(Cw, Cb, alpha)

		return			
		


	#the network will be evaluated against the test data after each epoch
	#return the number of correct classification samples 
	def evaluate(self, testData):
		samples, labels = testData
		x = self.feedforward(samples.T)
		results = np.argmax(x, axis = 0)

		return np.sum( (results - np.argmax(labels, axis = 1)) == 0 )

	#save trained model
	def saveModel(self):
		pass



class neurLayer(object):
	"""layer class"""
	def __init__(self, neuronNum, biases, weights, activateType):
		self.neuronNum = neuronNum
		self.biases = biases
		self.weights = weights
		self.activateFunc = self.linear if activateType == 'linear' else\
							self.sigmoid if activateType == 'sigmoid' else\
							self.tanh if activateType == 'tanh' else None
		self.backpropFunc = self.linearPrime if activateType == 'linear' else\
							self.sigmoidPrime if activateType == 'sigmoid' else\
							self.tanhPrime if activateType == 'tanh' else None

		if self.activateFunc is None:
			raise ValueError('invalid activation type.')

	#activation function
	def linear(self, z):
		return z

	def linearPrime(self, z):
		return np.ones_like(z)

	def sigmoid(self, z):
		return 1/(1 + np.exp(-z))

	def sigmoidPrime(self, z):
		a = self.sigmoid(z)
		return a * (1 - a)

	def tanh(self, z):
		a = np.exp(-z)
		b = np.exp(z)
		return (b - a) / (b + a)

	def tanhPrime(self, z):
		a = self.tanh(z)
		return 1 - a**2

	def relu(self, z):
		z[z < 0] = 0
		return z

	def reluPrime(self, z):
		return np.where(z < 0, 0, 1)

	#activate neurons
	def activateFunc(self, x):
		innerProd = np.dot(self.weights, x) + self.biases
		output = self.activateFunc(innerProd)

		return innerProd, output


	#backpropagation, this layer's error known, return the prime layer's error
	def backprop(self, deltaIn, z, a):
		if z is None:
			deltaOut = None
		else:
			deltaOut = np.dot(self.weights.T, deltaIn) * self.backpropFunc(z)
		Cb = np.mean(deltaIn, axis = 1)

		aT = a.T
		deltaSize, aTSize = deltaIn.shape, aT.shape
		deltaIn.shape = (deltaSize[0], 1, -1)
		aT.shape = (1, aTSize[1], -1)
		Cw = np.mean(deltaIn * a, axis = 2)
		deltaIn.shape = deltaSize
		del aT

		return deltaOut, Cb, Cw
	
	#update weights and biases
	def update(self, Cw, Cb, alpha):
		self.weights -= alpha * Cw
		self.biases -= alpha * Cb

