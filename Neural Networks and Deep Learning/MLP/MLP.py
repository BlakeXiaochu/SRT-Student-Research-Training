import queue
import h5py
from sys import exit as quit
from os import remove
import numpy as np
import matplotlib.pyplot as plt
from funcKit import *
from layer import *

class MLP(object):
	"""
		mutiple layers perceptron class
		layers including x layer and output layer, so neuronNums should be greater than 2.

		Paramters:
			neuronNums: a list/tuple representing numbers of each layer's neuron.
			biases: a list/tuple containing numpy.ndarray object, which representing each layer's biases
			weights: a list/tuple containing numpy.ndarray object, which representing each layer's weights
			activateFunc: activation function type, it must be from FuncKit module(please see FuncKit.py for details)
			lossFunc: loss function type, it must be from FuncKit module(please see FuncKit.py for details)
			regular: wheather use regularization trick or not
			momentum: wheather use momentum trick or not

			**kw:
				rLamda: regularization  coefficient
				miu: momentum coefficient
	"""
	__slots__ = {'neuronNums', 'layerNum', 'layers', 'biases', 'weights', 'activateFunc', 'lossFunc', 'regular', 'rLambda', 'momentum', 'miu', 'velocity'}
	def __init__(self):
		attrs = {'neuronNums', 'layerNum', 'layers', 'biases', 'weights', 'activateFunc', 'lossFunc', 'regular', 'rLambda', 'momentum', 'miu', 'velocity'}
		for attr in attrs:
			setattr(self, attr, None)


	#initialize parameters
	def initParams(self, neuronNums, biases = None, weights = None, activateFunc = actFunction.sigmoid, lossFunc = lossFunction.crossEntropy, regular = False, momentum = False, **kw):
		self.neuronNums = tuple(neuronNums)
		self.layerNum = len(neuronNums)

		#construct mutiple layers perceptron(input layer is absent)
		self.activateFunc = activateFunc
		self.lossFunc =  lossFunc
		self.layers = tuple( [neurLayer(num) for num in neuronNums[1:-1]] + [outputLayer(neuronNums[-1])] )

		#(L2)regularization
		self.regular = regular
		self.rLambda = kw['rLambda'] if regular else None

		#momentum
		self.momentum = momentum
		self.miu = kw['miu'] if momentum else 0
		self.velocity = [0] * (self.layerNum - 1)

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
			self.layers[i].initParams(self.biases[i], self.weights[i], activateFunc)

		#output layer
		self.layers[-1].initParams(self.biases[-1], self.weights[-1], activateFunc, lossFunc)



	#mini-batch stochastic gradient descent
	def SGD(self, trainData, epochNum, batchSize, alpha, testData = None, monitor = False):
		'''
			trainData: a tuple/list of (samples, labels), in which samples and labels' type are 2-D numpy.ndarray. each column represents one sample.
			epochNum: the number of trainning epochs
			batchSize: the number of trainning samples in each epoch
			alpha: learning rate
			testData: a tuple/list of (samples, labels) for testing. if provided, MLP will eavluate the testing data in each epoch and print results
			monitor: wheather monitoring the first trainning epoch or not, used for tuning.
		'''
		sampleNum = trainData[0].shape[1]
		testNum = testData[0].shape[1]

		#monitor the first trainning process
		if monitor:
			self.monitor(trainData, batchSize, alpha)

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

		#feedforward, and save intermediate results using LIFO queue
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
		loss = self.layers[-1].lossCompute(aL, labels)
		delta = self.layers[-1].deltaCompute(zL, aL, labels)

		#backprpagation
		for i in range(1, self.layerNum):
			layer = self.layers[-i]
			z, a = zQ.get(), aQ.get()
			delta, Cb, Cw = layer.backprop(delta, z, a)

			#regularization
			if self.regular:
				Cw += (self.rLambda * layer.weights / totalSampleNum)

			self.velocity[-i] *= self.miu
			self.velocity[-i] -= alpha * Cw

			layer.update(self.velocity[-i], -alpha*Cb)

		return loss
		

	#monitoring training process in one epoch
	def monitor(self, trainData, batchSize, alpha):
		samples, labels = trainData
		sampleNum = samples.shape[1]
		results = []
		for j in range(0, sampleNum - batchSize, batchSize):
			batch = (samples[:, j:j+batchSize], labels[:, j:j+batchSize])
			loss = self.update(batch, alpha, sampleNum)
			results.append(loss)
			del batch

		#plot the results
		plt.plot(np.array(results))
		plt.xlabel('batch(s)')
		plt.ylabel('loss')
		plt.title('The First Epoch')
		plt.show()

		#continue or not
		while True: 
			cmd = input('continue?(yes/no): ')
			if cmd == 'yes':
				break
			elif cmd == 'no':
				quit('trainning end...')


	#the network will be evaluated against the test data after each epoch
	#return the number of correct classification samples 
	def evaluate(self, testData):
		samples, labels = testData
		a = self.feedforward(samples)
		results = np.argmax(a, axis = 0)

		return np.sum( (results - np.argmax(labels, axis = 0)) == 0 )


	#save trained model
	#save as hdf5 file
	def saveModel(self, path = './model.hdf5'):
		try:
			f = h5py.File(path, 'w')

			#basicParams dataset stores layerNum, the types of activation and loss function
			basicParams = f.create_dataset('basicParams', (3, ), dtype = 'i8')
			actType = actFunction.funcTypeEncode(self.activateFunc)
			lossType = lossFunction.funcTypeEncode(self.lossFunc)
			basicParams[0] = self.layerNum
			basicParams[1] = actType
			basicParams[2] = lossType

			#biases and weights groups
			biasesGrp = f.create_group('biases')
			weightsGrp = f.create_group('weights')
			for i, layer in zip(range(1, self.layerNum), self.layers):
				biasesGrp.create_dataset('b' + str(i), data = layer.biases)
				weightsGrp.create_dataset('w' + str(i), data = layer.weights)

		except Exception as e:
			f.close()
			remove(path)
			raise e
		else:
			f.close()


	#load trained model(save as hdf5 file)
	def loadModel(self, path = './model.hdf5'):
		try:
			f = h5py.File(path, 'r')

			basicParams = f['basicParams']
			self.layerNum = basicParams[0]
			self.activateFunc = actFunction.funcTypeDecode(basicParams[1])
			self.lossFunc = lossFunction.funcTypeDecode(basicParams[2])

			#biases and weights groups
			biasesGrp = f['/biases']
			weightsGrp = f['/weights']
			#uncomplete
			pass
			self.biases = f['biases'][:]
			self.weights = f['weights'][:]
		except Exception as e:
			f.close()
			raise e
		finally:
			#initialize biases, weights and activation function(hidden layers)
			for i in range(self.layerNum - 2):
				self.layers[i].initParams(self.biases[i], self.weights[i], activateFunc)
			#output layer
			self.layers[-1].initParams(self.biases[-1], self.weights[-1], activateFunc, lossFunc)