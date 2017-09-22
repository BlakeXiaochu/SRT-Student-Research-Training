import queue
import numpy as np
from funcKit import *
from layer import *

class MLP(object):
	"""
		mutiple layers perceptron class
		layers including x layer and output layer, so neuronNums should be greater than 2.
	"""
	__slots__ = {'neuronNums', 'activateType', 'layerNum', 'layers', 'biases', 'weights', 'lossFunc', 'regular', 'rLambda'}
	def __init__(self, neuronNums, biases = None, weights = None, activateFunc = actFunction.sigmoid, lossFunc = lossFunction.crossEntropy, regular = False, **kw):
		self.neuronNums = tuple(neuronNums)
		self.layerNum = len(neuronNums)

		#construct mutiple layers perceptron(input layer is absent)
		self.lossFunc =  lossFunc
		self.layers = tuple( [neurLayer(num) for num in neuronNums[1:-1]] + [outputLayer(neuronNums[-1])] )

		#(L2)regularization
		self.regular = regular
		self.rLambda = kw['rLambda'] if regular else None

		if biases is None:
			self.biases = tuple([np.random.randn(i, 1) for i in self.neuronNums[1:]])
		else: 
			self.biases = biases

		if weights is None:
			self.weights = tuple([np.random.randn(j, i)/np.sqrt(i) for i, j in zip(self.neuronNums[:-1], self.neuronNums[1:])])
		else: 
			self.weights = weights

		#initialize biases, weights and activation function(hidden layers)
		for i in range(self.layerNum - 2):
			self.layers[i].initParam(self.biases[i], self.weights[i], activateFunc)

		#output layer
		self.layers[-1].initParam(self.biases[-1], self.weights[-1], activateFunc, lossFunc)



	#mini-batch stochastic gradient descent
	def SGD(self, trainData, epochNum, batchSize, alpha, testData = None):
		'''
			trainData: a tuple/list of (samples, labels), in which samples and labels' type are 2-D numpy.ndarray
			epochNum: the number of trainning epochs
			batchSize: the number of trainning samples in each epoch
			alpha: learning rate
			testData: a tuple/list of (samples, labels) for testing. if provided, MLP will eavluate the testing data in each epoch and print results
		'''
		sampleNum = trainData[0].shape[1]
		testNum = testData[0].shape[1]

		#epoch
		randOrder = np.arange(0, sampleNum)
		for i in range(epochNum):
			#shuffle the trainning data
			np.random.shuffle(randOrder)
			samples, labels = trainData
			samples, labels = samples[:, randOrder], labels[:, randOrder]

			#batch trainning
			for j in range(0, sampleNum - batchSize, batchSize):
				batch = (samples[:, j:j+batchSize], labels[:, j:j+batchSize])
				self.update(batch, alpha, sampleNum)
				del batch

			del samples
			del labels

			#print trainnig progress
			if testData:
				print( 'Epoch %d: testing accuracy = %.4f' % (i,  self.evaluate(testData) / testNum) )
			else:
				print('Epoch %d complete...' % (i))



	#feedforward
	def feedforward(self, a):
		for layer in self.layers:
			_, a = layer.activate(a)
		return a


	#compute mean square error
	def lossCompute(self, a, labels):
		return self.layers[-1].lossCompute(a, labels)



	#apply gradient descent, and update the network weights when trainning with a batch
	#alpha is the learning rate
	#totalSampleNum param is used for regularization
	def update(self, batch, alpha, totalSampleNum = None):
		samples, labels = batch

		#feedforward, and save intermadiate results using LIFO queue
		a = samples
		zQ = queue.LifoQueue(maxsize = self.layerNum)
		aQ = queue.LifoQueue(maxsize = self.layerNum)
		zQ.put(None)
		aQ.put(a)
		for layer in self.layers:
			z, a = layer.activate(a)
			zQ.put(z)
			aQ.put(a)

		#output layer's delta
		zL, aL = zQ.get(), aQ.get()
		delta = self.layers[-1].deltaCompute(zL, aL, labels)

		#backprpagation
		for i in range(1, self.layerNum):
			z, a = zQ.get(), aQ.get()
			layer = self.layers[-i]
			delta, Cb, Cw = layer.backprop(delta, z, a)

			#regularization
			if self.regular:
				Cw += (self.rLambda * layer.weights / totalSampleNum)

			layer.update(Cw, Cb, alpha)

		return			
		


	#the network will be evaluated against the test data after each epoch
	#return the number of correct classification samples 
	def evaluate(self, testData):
		samples, labels = testData
		a = self.feedforward(samples)
		results = np.argmax(a, axis = 0)

		return np.sum( (results - np.argmax(labels, axis = 0)) == 0 )


	#save trained model
	def saveModel(self):
		pass
