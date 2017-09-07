import sys
sys.path.append('../BinaryTree')
from BinaryTree import *
import numpy as np
from math import log

class StrongClassifier(object):
	"""
		docstring for StrongClassifier

		parameters:
			clsNum	- The number of weak classifiers(binary decision tree)
			pTree	- Trainning parameters
	"""
	__slot__ = {'clsNum', 'pTree', '_weakClsList', 'weakClsWt', 'discrete'}
	def __init__(self, clsNum, **pTree):
		if not isinstance(clsNum, int):
			print("parameter 'clsNum': type int is required")
			raise TypeError

		self.clsNum = clsNum
		self.pTree = pTree
		#weak classifier list
		self._weakClsList = None
		self.weakClsWt = None
		self.discrete = True


	#train the strong classifier through adaboost(discrete or real)
	def adaboostTrain(self, data, discrete = True):
		if self._weakClsList is not None:
			print('Strong classifier has been trained.')
			return
		
		if not isinstance(data, DataBin):
			print('DataBin object type is required.')
			raise TypeError

		if not isinstance(discrete, bool):
			print('bool type is required.')
			raise TypeError
		self.discrete = discrete

		if not data.quant:
			data.quantize()

		#train
		self._weakClsList = []
		if discrete:
			self.weakClsWt = []
		else:
			self.weakClsWt = None

		#for caculate loss function
		posCumCls = np.zeros(data.posWt.shape[0])
		negCumCls = np.zeros(data.negWt.shape[0]);
		for i in range(self.clsNum):
			binaryTree = BinaryTree(**self.pTree)
			binaryTree.train(data)

			#if discrete adaboost, classfication output is 1, -1
			if discrete:
				binaryTree.tree['hs'] = (binaryTree.tree['hs'] > 0) * 2.0 - 1

				constant = np.e**10 / (1 + np.e**10)
				alpha =  5.0 if (binaryTree.err < 1 - constant) else \
						-5.0 if (binaryTree.err > constant) else \
						0.5 * log((1 - binaryTree.err) / binaryTree.err)

				self.weakClsWt.append(alpha)
			else:
				alpha = 1.0

			self._weakClsList.append(binaryTree)

			#use trained weak classifier to classify
			posResult = binaryTree.apply(data.posSamp)
			negResult = binaryTree.apply(data.negSamp)

			#update samples weight
			posCumCls += posResult*alpha
			negCumCls += negResult*alpha
			data.posWt = np.exp(-posCumCls) / data.posWt.shape[0] / 2
			data.negWt = np.exp(negCumCls) / data.negWt.shape[0] / 2
			#data.posWt *= np.exp(-alpha * posResult)
			#data.negWt *= np.exp(alpha * negResult)

			#loss function
			loss = np.sum(data.posWt) + np.sum(data.negWt)
			print('weak classifier%d: err = %s, alpha = %s, loss = %s' % (i, format(binaryTree.err, '.3e'), format(alpha, '.3e'), format(loss, '.3e')))

			#samples weight normalization
			data.posWt /= loss
			data.negWt /= loss

			#overfit?
			if loss <= 1e-40:
				print('Stop early.')
				self.clsNum = i + 1
				break


		#apply decsion forest
		def apply(self):
			pass