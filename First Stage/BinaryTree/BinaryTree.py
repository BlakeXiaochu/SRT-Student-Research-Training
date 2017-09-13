import os
from math import log, floor
from numpy import ctypeslib as npcl
import numpy as np
import ctypes

#Trainning parameters class, used for storing trainning parameters.
class TrainParamBin(object):
	#Limit the parameters
	__slots__ = ('nBins', 'maxDepth', 'minWeight', 'fracFtrs', 'nThreads')

	def __init__(self):
		#Default
		self.nBins = 256
		self.maxDepth = 1
		self.minWeight = 0.01
		self.fracFtrs = 1.0
		self.nThreads = 16


#Data bin class, used for storing trainning data.
class DataBin(object):
	'''
		Attributes:
			posSamp         - [NPxF] negative feature vectors(Each row represents a sample, and each column represents a feature)
			negSamp         - [NNxF] positive feature vectors(Each row represents a sample, and each column represents a feature)
			quantPosSamp	- [NPxF] quantized positive samples
			quantNegSamp	- [NNxF] quantized negative samples
			posSampInterface	   - [NPx1]quantized positive feature row vectors pointer interface(used in bestStump.c)
			negSampInterface	   - [NNx1]quantized negative feature row vectors pointer interface(used in bestStump.c)
			posWt       - [NPx1] positive samples weights
			negWt       - [NNx1] negative samples weights
			xMin       - [1xF] optional vals defining feature quantization
			xMax      - [1xF] optional vals defining feature quantization
			quant 		- Quantization flag
			nBins 		- Number of Quantization bins
	'''
	__slots__ = ('posSamp', 'negSamp', 'posWt', 'negWt', 'posSampInterface', 'negSampInterface', 'quantPosSamp', 'quantNegSamp', 'xMin', 'xMax', 'quant', 'nBins')

	def __init__(self, posSamp, negSamp, posWt = None, negWt = None, nBins = 256):
		#Check type
		if isinstance(posSamp, np.ndarray):
			self.posSamp = posSamp
		else:
			raise TypeError('argument 1: np.ndarray is required.')
		if isinstance(negSamp, np.ndarray):
			self.negSamp = negSamp
		else:
			raise TypeError('argument 2: np.ndarray is required.')

		#Check the shape of Pos and Neg
		if posSamp.shape[1] != negSamp.shape[1]:
			raise ValueError("Postive samples' shape" + str(posSamp.shape) + "does not match negtive samples'" + str(negSamp.shape))

		#Check data type
		if isinstance(posWt, np.ndarray):
			self.posWt = posWt
		elif posWt is None:
			NP = self.posSamp.shape[0]
			self.posWt = np.ones(NP, dtype = 'float64') / NP / 2
		else:
			raise TypeError("argument 'posWt': np.ndarray is required.")

		if isinstance(negWt, np.ndarray):
			self.negWt = negWt
		elif negWt is None:
			NN = self.negSamp.shape[0]
			self.negWt = np.ones(NN, dtype = 'float64') / NN / 2
		else:
			raise TypeError("argument 'negWt': np.ndarray is required.")

		#Check the sum of weights
		w = np.sum(self.posWt) + np.sum(self.negWt)
		if abs(w - 1) > 1e-3:
			self.posWt /= w
			self.negWt /= w

		if  isinstance(nBins, int):
			self.nBins = nBins
		else:
			raise TypeError("argument 'nBins': type int is required.")

		self.posSampInterface = None 
		self.negSampInterface = None
		self.quantPosSamp = None
		self.quantNegSamp = None
		self.xMin = None
		self.xMax = None
		self.quant = False


	#Construct a brand-new data bin(Deep copy).
	def deepCopy(self):
		NewDataBin = DataBin(self.posSamp.copy(), self.negSamp.copy(), self.posWt.copy(), self.negWt.copy(), self.nBins)

		if isinstance(self.quantPosSamp, np.ndarray):
			NewDataBin.quantPosSamp = self.quantPosSamp.copy()
		else: 
			NewDataBin.quantPosSamp = None

		if isinstance(self.quantNegSamp, np.ndarray):
			NewDataBin.quantNegSamp = self.quantNegSamp.copy()
		else: 
			NewDataBin.quantNegSamp = None

		if isinstance(self.xMin, np.ndarray):
			NewDataBin.xMin = self.xMin.copy()
		else: 
			NewDataBin.xMin = None

		if isinstance(self.xMax, np.ndarray):
			NewDataBin.xMax = self.xMax.copy()
		else: 
			NewDataBin.xMax = None

		NewDataBin.quant = self.quant

		#Construct 1D-Pointer array interface
		rowId = np.arange(NewDataBin.quantPosSamp.shape[0])
		NewDataBin.posSampInterface = (NewDataBin.quantPosSamp.ctypes.data + NewDataBin.quantPosSamp.strides[0] * rowId).astype('uintp')
		del(rowId)
		rowId = np.arange(NewDataBin.quantNegSamp.shape[0])
		NewDataBin.negSampInterface = (NewDataBin.quantNegSamp.ctypes.data + NewDataBin.quantNegSamp.strides[0] * rowId).astype('uintp')
		del(rowId)

		return NewDataBin


	#Shallow copy
	def __copy__(self):
		NewDataBin = DataBin(self.posSamp, self.negSamp, self.posWt, self.negWt, self.nBins)
		NewDataBin.quantPosSamp = self.quantPosSamp
		NewDataBin.quantNegSamp = self.quantNegSamp
		NewDataBin.xMin = self.xMin
		NewDataBin.xMax = self.xMax
		NewDataBin.quant = self.quant
		NewDataBin.posSampInterface = self.posSampInterface
		NewDataBin.negSampInterface = self.negSampInterface


	def __deepcopy__(self):
		return self.Copy()


	def quantize(self):
		if self.quant == True:
			return

		#find minimum values of each feature
		PosxMin = np.min(self.posSamp, axis = 0)
		NegxMin = np.min(self.negSamp, axis = 0)
		self.xMin = np.where(PosxMin < NegxMin, PosxMin, NegxMin)
		del(PosxMin)
		del(NegxMin)
		PosxMax = np.max(self.posSamp, axis = 0)
		NegxMax = np.max(self.negSamp, axis = 0)
		self.xMax = np.where(PosxMax > NegxMax, PosxMax, NegxMax)
		del(PosxMax)
		del(NegxMax)

		#quantize to 0 ~ nBins-1
		quantPosSamp = (self.posSamp - self.xMin) / (self.xMax - self.xMin) * (self.nBins - 1.0)
		self.quantPosSamp = quantPosSamp.astype('uint8')
		del(quantPosSamp)

		quantNegSamp = (self.negSamp - self.xMin) / (self.xMax - self.xMin) * (self.nBins - 1.0)
		self.quantNegSamp = quantNegSamp.astype('uint8')
		del(quantNegSamp)

		#Construct 1D-Pointer array interface
		rowId = np.arange(self.quantPosSamp.shape[0])
		self.posSampInterface = (self.quantPosSamp.ctypes.data + self.quantPosSamp.strides[0] * rowId).astype('uintp')
		del(rowId)
		rowId = np.arange(self.quantNegSamp.shape[0])
		self.negSampInterface = (self.quantNegSamp.ctypes.data + self.quantNegSamp.strides[0] * rowId).astype('uintp')
		del(rowId)

		self.quant = True

		return


	#update the samples weight
	def updateWt(self):
		pass



class BinaryTree(object):
	__slots__ = ('pTree', 'bestStumpFunc', 'applyFun', 'tree', 'err')

	#Initialize trainning parameters
	def __init__(self, **pTree):
		'''	
			pTree      - Trainning parameters
			key(str):
				nBins      - [256] maximum number of quanizaton bins (<=256)
				maxDepth   - [1] maximum depth of tree
				minWeight  - [.01] minimum sample weigth to allow split
				fracFtrs   - [1] fraction of features numbers to sample for each node split
				nThreads   - [16] max number of computational threads to use

			bestStumpFunc	- bestStump.c interface function
			applyFun		- BinaryTreeApply.c interface function

			tree 	   - Trained decision tree.
			key(str):
				fids		- the ids of features
				thrs 		- corresponding thresholds of features
				child 		- left child node of each tree node
				hs 			- misclassified samples log ratio
				weights		- the sum of trainning samples' weight of each nodes
				depth 		- depth of each tree node

			err 		- weighted mean sum of misclassified samples

		'''
		try:
			self.pTree = TrainParamBin()
			for key, value in pTree.items():
				setattr(self.pTree, key, value)

			self.bestStumpFunc = None
			self.applyFun = None
			self.tree = None
			self.err = None
		except AttributeError as e:
			raise e('No Such parameter:' + key)
		finally:
			assert self.pTree.fracFtrs <= 1

	#Load the C-function bestStump.c and construct ctype interface
	def loadBestStumpFunc(self, path = None):
		if self.bestStumpFunc is not None:
			return

		if path is None:
			path = os.path.dirname( os.path.abspath(__file__) )

		if not isinstance(path, str):
			raise TypeError("Parameter 'path' is required to be str.")

		f = npcl.load_library('bestStump', path)
		pp = npcl.ndpointer(dtype = 'uintp', ndim = 1, flags = 'C')		#2D pointer(pointer to pointer)
		double_p = npcl.ndpointer(dtype = 'float64', ndim = 1, flags = 'C')
		uint32_p = npcl.ndpointer(dtype = 'uint32', ndim = 1, flags = 'C')
		uint8_p = npcl.ndpointer(dtype = 'uint8', ndim = 1, flags = 'C')
		f.BestStump.restype = None
		f.BestStump.argtypes = [
			pp, 
			pp, 
			double_p, 
			double_p, 
			ctypes.c_int, 
			ctypes.c_int, 
			uint32_p, 
			ctypes.c_int, 
			ctypes.c_double, 
			ctypes.c_int, 
			ctypes.c_int, 
			double_p, 
			uint8_p
			]
		
		self.bestStumpFunc = f.BestStump


	#Given data, find the best stump classifier.
	#Wrap the function in C. Not recommanded to use alone.
	def bestStump(self, data, nodePosWt, nodeNegWt, stumpFtrsId = None, prior = None):
		if not prior:
			posWtSum = np.sum(posWt)
			negWtSum = np.sum(negWt)
			wtSum = posWtSum + negWtSum
			prior = posWtSum / wtSum

		if not data.quant:
			data.quantize()

		if not isinstance(stumpFtrsId, np.ndarray):
			stumpFtrsId = np.arange(data.posSamp.shape[1], dtype = 'uint32')

		if self.bestStumpFunc is None:
			self.loadBestStumpFunc()

		(NP, F) = data.posSamp.shape
		NN = data.negSamp.shape[0]
		stumpThrs = np.zeros(floor(self.pTree.fracFtrs * F), dtype = 'uint8')
		stumpErrs = np.zeros(floor(self.pTree.fracFtrs * F), dtype = 'float64')
		#Best Stump(For effiency, Coding in C)
		self.bestStumpFunc(
			data.posSampInterface,
			data.negSampInterface,
			nodePosWt,
			nodeNegWt,
			ctypes.c_int(NP),
			ctypes.c_int(NN),
			stumpFtrsId,
			ctypes.c_int( floor(self.pTree.fracFtrs * F) ),
			ctypes.c_double(prior),
			ctypes.c_int(self.pTree.nBins),
			ctypes.c_int(self.pTree.nThreads),
			stumpErrs,
			stumpThrs
			)

		return stumpErrs, stumpThrs



	def train(self, data):
		'''
			Reference to piotr dollar's Computer Vision matlab toolbox

			INPUTS
				data       - Type: DataBin. Data for training tree.

			OUTPUTS
			tree       - (dict)learned decision tree model struct with the following keys:
				fids       - [Kx1] feature ids for each node
				thrs       - [Kx1] threshold corresponding to each fid
				child      - [Kx1] index of child for each node (1-indexed)
				hs         - [Kx1] log ratio (.5*log(p/(1-p)) at each node
				weights    - [Kx1] total sample weight at each node
				depth      - [Kx1] depth of each node
			data       - data used for training tree (quantized version of input)
			err        - decision tree training error
		'''
		if not isinstance(data, DataBin):
			raise TypeError('DataBin object type is required.')

		if not data.quant:
			data.quantize()

		#Initialize arrays
		(NP, FP) = data.posSamp.shape
		(NN, FN) = data.negSamp.shape
		assert FP == FN
		F = FP

		tree = dict()
		maxNodes = 2**(self.pTree.maxDepth + 1) - 1							#Maximum number of nodes in BinaryTree
		tree['fids'] = np.zeros(maxNodes, dtype = 'uint32')
		tree['thrs'] = np.zeros(maxNodes, dtype = 'float64')   
		tree['child'] = np.zeros(maxNodes, dtype = 'uint32')
		tree['hs'] = np.zeros(maxNodes, dtype = 'float64')
		tree['weights'] = np.zeros(maxNodes, dtype = 'float64')
		tree['depth'] = np.zeros(maxNodes, dtype = 'uint32')
		errs = np.zeros(maxNodes, dtype = 'float64')

		#Train Decision tree
		curNode = 0                        #Current Node's id
		lastNode = 1					   #Last Node's id that has been yield
		nodePosWtList = [None] * maxNodes	   #an assemble of nodes' samples weight(if a sample does not reaches the node, its weight = 0)
		nodeNegWtList = [None] * maxNodes
		nodePosWtList[0] = data.posWt
		nodeNegWtList[0] = data.negWt

		while curNode < lastNode:
			nodePosWt = nodePosWtList[curNode]
			nodeNegWt = nodeNegWtList[curNode]
			nodePosWtList[curNode] = None
			nodeNegWtList[curNode] = None
			nodePosWtSum = np.sum(nodePosWt)
			nodeNegWtSum = np.sum(nodeNegWt)
			nodeWtSum = nodePosWtSum + nodeNegWtSum

			tree['weights'][curNode] = nodeWtSum
			prior = nodePosWtSum / nodeWtSum
			errs[curNode] = min(prior, 1 - prior)
			constant = np.e**8 / (1 + np.e**8)
			alpha =  4.0 if (prior > constant) else \
					-4.0 if (prior < 1 - constant) else \
					0.5 * log(prior / (1 - prior))
			tree['hs'][curNode] = alpha
			#alpha = 0.5 * log(prior / (1 - prior))
			#tree['hs'][curNode] = max(-4.0, min(4.0, alpha))

			#Node's classification is nearly pure, node's depth is out of scale, sum of node samples' weight is out of scale
			if (prior < 1e-3 or prior > 1 - 1e-3) or (tree['depth'][curNode] >= self.pTree.maxDepth) or (nodeWtSum < self.pTree.minWeight) :
				curNode += 1
				continue

			#Find best tree stump
			#wheather subsample the features or not
			if self.pTree.fracFtrs < 1:
				stumpFtrsId = np.choice(np.arange(F), floor(self.pTree.fracFtrs * F)).astype('uint32')
			else: 
				stumpFtrsId = np.arange(F, dtype = 'uint32')

			(stumpErrs, stumpThrs) = self.bestStump(data, nodePosWt/nodeWtSum, nodeNegWt/nodeWtSum, stumpFtrsId, prior)

			bestFtrsId = np.argmin(stumpErrs)
			bestThrs = stumpThrs[bestFtrsId] + 0.5
			bestFtrsId = stumpFtrsId[bestFtrsId]

			#Split node
			leftChlidPosWt = data.quantPosSamp[:, bestFtrsId] < bestThrs 		#Node's left child's positive samples' weights
			leftChlidNegWt = data.quantNegSamp[:, bestFtrsId] < bestThrs
			if (np.any(leftChlidPosWt) or np.any(leftChlidNegWt))  and  (np.any(~leftChlidPosWt) or np.any(~leftChlidNegWt)):		#Invalid stump classifier
				#Inverse quantization
				bestThrs = data.xMin[bestFtrsId] + bestThrs * (data.xMax[bestFtrsId] - data.xMin[bestFtrsId]) / (self.pTree.nBins - 1)
				nodePosWtList[lastNode] = leftChlidPosWt * nodePosWt
				nodeNegWtList[lastNode] = leftChlidNegWt * nodeNegWt
				nodePosWtList[lastNode + 1] = (~leftChlidPosWt) * nodePosWt
				nodeNegWtList[lastNode + 1] = (~leftChlidNegWt) * nodeNegWt

				tree['thrs'][curNode] = bestThrs
				tree['fids'][curNode] = bestFtrsId
				tree['child'][curNode] = lastNode
				tree['depth'][lastNode : lastNode + 2] = tree['depth'][curNode] + 1

				lastNode += 2

			curNode += 1

		#Modefy parameter 'tree':
		tree['fids'] = tree['fids'][0:lastNode].copy()
		tree['thrs'] = tree['thrs'][0:lastNode].copy()
		tree['child'] = tree['child'][0:lastNode].copy()
		tree['hs'] = tree['hs'][0:lastNode].copy()
		tree['weights'] = tree['weights'][0:lastNode].copy()
		tree['depth'] = tree['depth'][0:lastNode].copy()
		err = np.sum(errs[0:lastNode] * tree['weights'] * (tree['child'] == 0))				#Sum up the leaf nodes' error

		#return
		self.tree = tree
		self.err = err


	def loadApplyFunc(self, path = None):
		if self.applyFun is not None:
			return

		if path is None:
			path = os.path.dirname( os.path.abspath(__file__) )

		if not isinstance(path, str):
			raise TypeError("Parameter 'path' is required to be str.")

		f = npcl.load_library('BinaryTreeApply', path)
		pp = npcl.ndpointer(dtype = 'uintp', ndim = 1, flags = 'C')		#2D pointer(pointer to pointer)
		double_p = npcl.ndpointer(dtype = 'float64', ndim = 1, flags = 'C')
		uint32_p = npcl.ndpointer(dtype = 'uint32', ndim = 1, flags = 'C')
		uint8_p = npcl.ndpointer(dtype = 'uint8', ndim = 1, flags = 'C')
		f.BinaryTreeApply.restype = None
		f.BinaryTreeApply.argtypes = [
			pp,
			ctypes.c_uint32,
			uint32_p,
			double_p,
			uint32_p,
			double_p,
			ctypes.c_int,
			double_p
		]

		self.applyFun = f.BinaryTreeApply


	#apply the binary decision tree and classify the given data
	#
	def apply(self, data):
		if self.tree is None:
			raise Exception('Binary tree has not been trained.')

		if not isinstance(data, np.ndarray):
			raise TypeError('numpy.ndarray type is required.')

		if data.ndim == 1:
			data = data.copy()
			data = data.reshape(-1, 1)
		elif data.ndim > 2:
			raise ValueError('1 or 2 dimension data is required.')

		rowId = np.arange(data.shape[0])
		dataInterface = (data.ctypes.data + rowId * data.strides[0]).astype('uintp')

		if self.applyFun is None:
			self.loadApplyFunc()

		results = np.arange(data.shape[0], dtype = 'float64')

		self.applyFun(
			dataInterface, 
			ctypes.c_uint32(data.shape[0]),
			self.tree['fids'],
			self.tree['thrs'],
			self.tree['child'],
			self.tree['hs'],
			ctypes.c_int(self.pTree.nThreads),
			results
			)

		return results
