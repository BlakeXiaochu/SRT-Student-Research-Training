from math import log, floor
from numpy import ctypeslib as npcl
import numpy as np
import ctypes

#Trainning parameters class, used for storing trainning parameters.
class TrainParamBin(object):
	def __init__(self):
		#Limit the parameters
		self.__slots__ = ('nBins', 'maxDepth', 'minWeight', 'fracFtrs', 'nThreads')
		#Default
		self.nBins = 256
		self.maxDepth = 1
		self.minWeight = 0.01
		self.fracFtrs = 1.0
		self.nThreads = 16


class BinaryTree(object):
	#Initialize trainning parameters
	def __init__(self, **pTree):
		'''	
			pTree      - Trainning parameters
			dict key(type: str):
				nBins      - [256] maximum number of quanizaton bins (<=256)
				maxDepth   - [1] maximum depth of tree
				minWeight  - [.01] minimum sample weigth to allow split
				fracFtrs   - [1] fraction of features numbers to sample for each node split
				nThreads   - [16] max number of computational threads to use
		'''
		self.__slots__ = ('pTree', 'Tree', 'err')
		try:
			self.pTree = TrainParamBin()
			for key, value in pTree.items():
				setattr(self.pTree, key, value)
		except AttributeError as e:
			print('No Such parameter:', key)
			raise e
		finally:
			assert self.pTree.fracFtrs <= 1

	#Given data, find the best stump classifier.
	#Wrap the function in C
	def BestStump(self, data, PosWt, NegWt, StpFtrsId = None, prior = None):
		if not prior:
			PosWtSum = np.sum(PosWt)
			NegWtSum = np.sum(NegWt)
			WtSum = PosWtSum + NegWtSum
			prior = PosWtSum / WtSum

		if not isinstance(StpFtrsId, np.ndarray):
			StpFtrsId = np.arange(data['PosFtrsVec'].shape[1], dtype = 'uint32')

		if not isinstance(data.get('PosFtrsVecToC'), np.ndarray):
			RowId = np.arange(data['PosFtrsVec'].shape[0])
			data['PosFtrsVecToC'] = (data['PosFtrsVec'].ctypes.data + data['PosFtrsVec'].strides[0] * RowId).astype('uintp')
			del(RowId)

		if not isinstance(data.get('NegFtrsVecToC'), np.ndarray):
			RowId = np.arange(data['NegFtrsVec'].shape[0])
			data['NegFtrsVecToC'] = (data['NegFtrsVec'].ctypes.data + data['PosFtrsVec'].strides[0] * RowId).astype('uintp')
			del(RowId)

		#Load the C-function
		pp = npcl.ndpointer(dtype = 'uintp', ndim = 1, flags = 'C')		#2D pointer(pointer to pointer)
		double_p = npcl.ndpointer(dtype = 'float64', ndim = 1, flags = 'C')
		uint32_p = npcl.ndpointer(dtype = 'uint32', ndim = 1, flags = 'C')
		uint8_p = npcl.ndpointer(dtype = 'uint8', ndim = 1, flags = 'C')
		f = npcl.load_library('BestStump', '.')
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

		(NP, F) = data['PosFtrsVec'].shape
		NN = data['NegFtrsVec'].shape[0]
		StpThrs = np.zeros(floor(self.pTree.fracFtrs * F), dtype = 'uint8')
		StpErrs = np.zeros(floor(self.pTree.fracFtrs * F), dtype = 'float64')
		#Best Stump(For effiency, Coding in C)
		f.BestStump(
			data['PosFtrsVecToC'],
			data['NegFtrsVecToC'],
			PosWt,
			NegWt,
			ctypes.c_int(NP),
			ctypes.c_int(NN),
			StpFtrsId,
			ctypes.c_int( floor(self.pTree.fracFtrs * F) ),
			ctypes.c_double(prior),
			ctypes.c_int(self.pTree.nBins),
			ctypes.c_int(self.pTree.nThreads),
			StpErrs,
			StpThrs
			)

		return StpErrs, StpThrs



	def Train(self, data, **pTree):
		'''
			Reference to piotr dollar's Computer Vision matlab toolbox

			INPUTS
				data       - (dict)data for training tree.
				dict key(type: str):
					PosFtrsVec         - [NPxF] negative feature vectors(Each row represents a sample, and each column represents a feature)
					NegFtrsVec         - [NNxF] positive feature vectors(Each row represents a sample, and each column represents a feature)
					PosFtrsVecToC	   - [NPx1] positive feature row vectors pointer(used in BestStump.c)
					NegFtrsVecToC	   - [NNx1] negative feature row vectors pointer(used in BestStump.c)
					PosWt       - [NPx1] positive samples weights
					NegWt       - [NNx1] negative samples weights
					xMin       - [1xF] optional vals defining feature quantization
					xMax      - [1xF] optional vals defining feature quantization
					xType      - [] optional original data type for features

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
		#Deep copy a dict object
		def DeepCopy(a):
			if not isinstance(a, dict):
				raise TypeError
			b = dict()
			for key in a:
				#Object a has a method 'copy'
				if 'copy' in dir(a[key]):
					b[key] = a[key].copy()
				else:
					b[key] = a[key]
			return b

		#Merge data parameters
		data = DeepCopy(data)
		#default parameters
		data_dfs = {'PosFtrsVec': 'REQ', 'NegFtrsVec': 'REQ', 'PosWt': None, 'NegWt': None, 'xMin': None, 'xMax': None, 'xType': None}
		data_dfs_copy = data_dfs.copy()
		data_dfs_copy.update(data)
		data = data_dfs_copy

		#1D-Pointer array
		if not isinstance(data.get('PosFtrsVecToC'), np.ndarray):
			RowId = np.arange(data['PosFtrsVec'].shape[0])
			data['PosFtrsVecToC'] = (data['PosFtrsVec'].ctypes.data + data['PosFtrsVec'].strides[0] * RowId).astype('uintp')
			del(RowId)
		if not isinstance(data.get('NegFtrsVecToC'), np.ndarray):
			RowId = np.arange(data['NegFtrsVec'].shape[0])
			data['NegFtrsVecToC'] = (data['NegFtrsVec'].ctypes.data + data['NegFtrsVec'].strides[0] * RowId).astype('uintp')
			del(RowId)

		#Initialize arrays and data's weights
		(NP, FP) = data['PosFtrsVec'].shape
		(NN, FN) = data['NegFtrsVec'].shape
		assert FP == FN
		F = FP

		tree = dict()
		MaxNodes = (NP + FP) * 2									#Maximum number of nodes in BinaryTree
		tree['fids'] = np.zeros(MaxNodes, dtype = 'uint32')
		tree['thrs'] = np.zeros(MaxNodes, dtype = 'float64')   		#dtype??
		tree['child'] = np.zeros(MaxNodes, dtype = 'uint32')
		tree['hs'] = np.zeros(MaxNodes, dtype = 'float64')
		tree['weights'] = np.zeros(MaxNodes, dtype = 'float64')
		tree['depth'] = np.zeros(MaxNodes, dtype = 'uint32')

		if not isinstance(data['PosWt'], np.ndarray):
			data['PosWt'] = np.ones(NP, dtype = 'float64') / NP
		if not isinstance(data['NegWt'], np.ndarray):
			data['NegWt'] = np.ones(NN, dtype = 'float64') / NN
		w = np.sum(data['PosWt']) + np.sum(data['NegWt'])
		if abs(w - 1) > 1e-3:
			data['PosWt'] /= w
			data['NegWt'] /= w

		errs = np.zeros(MaxNodes, dtype = 'float64')

		#Quantize data
		if not (str(data['PosFtrsVec'].dtype) == 'uint8' and str(data['NegFtrsVec'].dtype) == 'uint8'):
			#find minimum values of each feature
			PosxMin = np.min(data['PosFtrsVec'], axis = 0)
			NegxMin = np.min(data['NegFtrsVec'], axis = 0)
			xMin = np.where(PosxMin < NegxMin, PosxMin, NegxMin)
			del(PosxMin)
			del(NegxMin)

			PosxMax = np.max(data['PosFtrsVec'], axis = 0)
			NegxMax = np.max(data['NegFtrsVec'], axis = 0)
			xMax = np.where(PosxMax > NegxMax, PosxMax, NegxMax)
			del(PosxMax)
			del(NegxMax)

			#Quantize to 0 ~ nBins-1
			data['xMin'] = xMin
			data['xMax'] = xMax
			data['xType'] = str(data['PosFtrsVec'].dtype)
			data['PosFtrsVec'] = (data['PosFtrsVec'] - xMin) / (xMax - xMin) * (self.pTree.nBins - 1)
			data['PosFtrsVec'] = data['PosFtrsVec'].astype('uint8')
			data['NegFtrsVec'] = (data['NegFtrsVec'] - xMin) / (xMax - xMin) * (self.pTree.nBins - 1)
			data['NegFtrsVec'] = data['NegFtrsVec'].astype('uint8')
			#Rearrange 1D-Pointer array 
			RowId = np.arange(data['PosFtrsVec'].shape[0])
			data['PosFtrsVecToC'] = (data['PosFtrsVec'].ctypes.data + data['PosFtrsVec'].strides[0] * RowId).astype('uintp')
			del(RowId)
			RowId = np.arange(data['NegFtrsVec'].shape[0])
			data['NegFtrsVecToC'] = (data['NegFtrsVec'].ctypes.data + data['NegFtrsVec'].strides[0] * RowId).astype('uintp')
			del(RowId)

		#Train Decision Tree
		CurNode = 0                        #Current Node's id
		LastNode = 1					   #Last Node's id that has been yield
		NodePosWtList = [[]] * MaxNodes	   #an assemble of nodes' samples weight(if a sample does not reaches the node, its weight = 0)
		NodeNegWtList = [[]] * MaxNodes
		NodePosWtList[0] = data['PosWt']
		NodeNegWtList[0] = data['NegWt']

		while CurNode < LastNode:
			NodePosWt = NodePosWtList[CurNode]
			NodeNegWt = NodeNegWtList[CurNode]
			NodePosWtList[CurNode] = []
			NodeNegWtList[CurNode] = []
			NodePosWtSum = np.sum(NodePosWt)
			NodeNegWtSum = np.sum(NodeNegWt)
			NodeWtSum = NodePosWtSum + NodeNegWtSum

			tree['weights'][CurNode] = NodeWtSum
			prior = NodePosWtSum / NodeWtSum
			errs[CurNode] = min(prior, 1 - prior)
			constant = np.e**8 / (1 + np.e**8)
			alpha =  4.0 if (prior > constant) else \
					-4.0 if (prior < 1 - constant) else \
					0.5 * log(prior / (1 - prior))
			tree['hs'][CurNode] = alpha
			#alpha = 0.5 * log(prior / (1 - prior))
			#tree['hs'][CurNode] = max(-4.0, min(4.0, alpha))

			#Node's classification is nearly pure, node's depth is out of scale, sum of node samples' weight is out of scale
			if (prior < 1e-3 or prior > 1 - 1e-3) or (tree['depth'][CurNode] >= self.pTree.maxDepth) or (NodeWtSum < self.pTree.minWeight) :
				CurNode += 1
				continue

			#Find best tree stump
			#wheather subsample the features or not
			if self.pTree.fracFtrs < 1:
				StpFtrsId = np.choice(np.arange(F), floor(self.pTree.fracFtrs * F)).astype('uint32')
			else: 
				StpFtrsId = np.arange(F, dtype = 'uint32')

			(StpErrs, StpThrs) = self.BestStump(data, NodePosWt/NodeWtSum, NodeNegWt/NodeWtSum, StpFtrsId, prior)

			BestFtrsId = np.argmin(StpErrs)
			BestThrs = StpThrs[BestFtrsId] + 0.5
			BestFtrsId = StpFtrsId[BestFtrsId]

			#Split node
			LeftCldPosWt = data['PosFtrsVec'][:, BestFtrsId] < BestThrs 		#Node's left child's positive samples' weights
			LeftCldNegWt = data['NegFtrsVec'][:, BestFtrsId] < BestThrs
			if (np.any(LeftCldPosWt) or np.any(LeftCldNegWt))  and  (np.any(~LeftCldPosWt) or np.any(~LeftCldNegWt)):		#Invalid stump classifier
				#Inverse quantization
				BestThrs = xMin[BestFtrsId] + BestThrs * (xMax[BestFtrsId] - xMin[BestFtrsId]) / (self.pTree.nBins - 1)
				NodePosWtList[LastNode] = LeftCldPosWt * NodePosWt
				NodeNegWtList[LastNode] = LeftCldNegWt * NodeNegWt
				NodePosWtList[LastNode + 1] = (~LeftCldPosWt) * NodePosWt
				NodeNegWtList[LastNode + 1] = (~LeftCldNegWt) * NodeNegWt

				tree['thrs'][CurNode] = BestThrs
				tree['fids'][CurNode] = BestFtrsId
				tree['child'][CurNode] = LastNode
				tree['depth'][LastNode : LastNode + 2] = tree['depth'][CurNode] + 1

				LastNode += 2

			CurNode += 1

		#Modefy parameter 'tree':
		tree['fids'] = tree['fids'][0:LastNode].copy()
		tree['thrs'] = tree['thrs'][0:LastNode].copy()
		tree['child'] = tree['child'][0:LastNode].copy()
		tree['hs'] = tree['hs'][0:LastNode].copy()
		tree['weights'] = tree['weights'][0:LastNode].copy()
		tree['depth'] = tree['depth'][0:LastNode].copy()
		err = np.sum(errs[0:LastNode] * tree['weights'] * (tree['child'] == 0))				#Sum up the leaf nodes' error

		#return
		self.tree = tree
		self.err = err
		return data
