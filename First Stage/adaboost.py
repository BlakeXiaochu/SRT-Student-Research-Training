import sys
sys.path.append('../BinaryTree')
from BinaryTree import *
import numpy as np
from math import log

class strongClassifier(object):
	"""
		docstring for strongClassifier

		parameters:
			clsNum	- The number of weak classifiers(binary decision tree)
			pTree	- Trainning parameters
	"""
	def __init__(self, clsNum, **pTree):
		if not isinstance(clsNum, int):
			print("parameter 'clsNum': type int is required")
			raise TypeError

		self.clsNum = clsNum
		self.pTree = pTree
		#weak classifier list
		self._weakClsList = None
		self.weakClsWt = None

	#train the strong classifier through adaboost(discrete)
	def adaboostTrain(self, data):
		if self._weakClsList is not None:
			print('Strong classifier has been trained.')
			return
		
		if not isinstance(data, DataBin):
			print('DataBin object type is required.')
			raise TypeError

		if not data.quant:
			data.quantize()

		#train
		self._weakClsList = []
		self.weakClsWt = []
		for i in range(self.clsNum):
			tree = BinaryTree(self.pTree)
			tree.train(data)
			self._weakClsList.append(tree)
			constant = np.e**10 / (1 + np.e**10)
			alpha =  5.0 if (tree.err < 1 - constant) else \
					-5.0 if (tree.err > constant) else \
					0.5 * log((1 - tree.err) / tree.err)
			self.weakClsWt.append(alpha)

			#use trained weak classifier to classify
			posResult = tree.apply(data.posSamp)
			negResult = tree.apply(data.negSamp)

			#update samples weight and normalize

