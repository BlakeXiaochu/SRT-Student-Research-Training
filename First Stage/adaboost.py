import sys
sys.path.append('../BinaryTree')
from BinaryTree import *
import numpy as np

class strongClassifier(object):
	"""
		docstring for strongClassifier

		parameters:
			clsNum	- The number of weak classifiers(binary decision tree)
			pTree	- 
	"""
	def __init__(self, clsNum, **pTree):
		if not isinstance(clsNum, int):
			print("parameter 'clsNum': type int is required")
			raise TypeError

		self.clsNum = clsNum



	#train the strong classifier through adaboost
	def adaboostTrain(data):
		pass
		