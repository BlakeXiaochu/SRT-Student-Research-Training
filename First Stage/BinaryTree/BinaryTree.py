<<<<<<< HEAD
from math import log, floor
from numpy import ctypeslib as npcl
import numpy as np
import ctypes

#Trainning parameters class, used for storing trainning parameters.
class TrainParamBin(object):
	#Limit the parameters
	__slots__ = ('nBins', 'MaxDepth', 'MinWeight', 'FracFtrs', 'nThreads')

	def __init__(self):
		#Default
		self.nBins = 256
		self.MaxDepth = 1
		self.MinWeight = 0.01
		self.FracFtrs = 1.0
		self.nThreads = 16


#Data bin class, used for storing trainning data.
class DataBin(object):
	'''
		Attributes:
			PosSamp         - [NPxF] negative feature vectors(Each row represents a sample, and each column represents a feature)
			NegSamp         - [NNxF] positive feature vectors(Each row represents a sample, and each column represents a feature)
			QuantPosSamp	- [NPxF] quantized positive samples
			QuantNegSamp	- [NNxF] quantized negative samples
			PosSampITF	   - [NPx1]quantized positive feature row vectors pointer interface(used in BestStump.c)
			NegSampITF	   - [NNx1]quantized negative feature row vectors pointer interface(used in BestStump.c)
			PosWt       - [NPx1] positive samples weights
			NegWt       - [NNx1] negative samples weights
			xMin       - [1xF] optional vals defining feature quantization
			xMax      - [1xF] optional vals defining feature quantization
			Quant 		- Quantization flag
			nBins 		- Number of Quantization bins
	'''
	__slots__ = ('PosSamp', 'NegSamp', 'PosWt', 'NegWt', 'PosSampITF', 'NegSampITF', 'QuantPosSamp', 'QuantNegSamp', 'xMin', 'xMax', 'Quant', 'nBins')

	def __init__(self, PosSamp, NegSamp, PosWt = None, NegWt = None, nBins = 256):
		#Check type
		if isinstance(PosSamp, np.ndarray):
			self.PosSamp = PosSamp
		else:
			print('np.ndarray is required.')
			raise TypeError
		if isinstance(NegSamp, np.ndarray):
			self.NegSamp = NegSamp
		else:
			print('np.ndarray is required.')
			raise TypeError

		#Check the shape of Pos and Neg
		if PosSamp.shape[1] != NegSamp.shape[1]:
			print("Postive samples' shape", PosSamp.shape, "does not match negtive samples'", NegSamp.shape)
			raise ValueError

		#Check data type
		if isinstance(PosWt, np.ndarray):
			self.PosWt = PosWt
		elif PosWt is None:
			NP = self.PosSamp.shape[0]
			self.PosWt = np.ones(NP, dtype = 'float64') / NP
		else:
			print('np.ndarray is required.')
			raise TypeError

		if isinstance(NegWt, np.ndarray):
			self.NegWt = NegWt
		elif NegWt is None:
			NN = self.NegSamp.shape[0]
			self.NegWt = np.ones(NN, dtype = 'float64') / NN
		else:
			print('np.ndarray is required.')
			raise TypeError

		#Check the sum of weights
		w = np.sum(self.PosWt) + np.sum(self.NegWt)
		if abs(w - 1) > 1e-3:
			self.PosWt /= w
			self.NegWt /= w

		if  isinstance(nBins, int):
			self.nBins = nBins
		else:
			print('nBins:', 'type int is required.')
			raise TypeError

		self.PosSampITF = None 
		self.NegSampITF = None
		self.QuantPosSamp = None
		self.QuantNegSamp = None
		self.xMin = None
		self.xMax = None
		self.Quant = False


	#Construct a brand-new data bin(Deep copy).
	def Copy(self):
		NewDataBin = DataBin(self.PosSamp.copy(), self.NegSamp.copy(), self.PosWt.copy(), self.NegWt.copy(), self.nBins)

		if isinstance(self.QuantPosSamp, np.ndarray):
			NewDataBin.QuantPosSamp = self.QuantPosSamp.copy()
		else: 
			NewDataBin.QuantPosSamp = None

		if isinstance(self.QuantNegSamp, np.ndarray):
			NewDataBin.QuantNegSamp = self.QuantNegSamp.copy()
		else: 
			NewDataBin.QuantNegSamp = None

		if isinstance(self.xMin, np.ndarray):
			NewDataBin.xMin = self.xMin.copy()
		else: 
			NewDataBin.xMin = None

		if isinstance(self.xMax, np.ndarray):
			NewDataBin.xMax = self.xMax.copy()
		else: 
			NewDataBin.xMax = None

		NewDataBin.Quant = self.Quant

		#Construct 1D-Pointer array interface
		RowId = np.arange(NewDataBin.QuantPosSamp.shape[0])
		NewDataBin.PosSampITF = (NewDataBin.QuantPosSamp.ctypes.data + NewDataBin.QuantPosSamp.strides[0] * RowId).astype('uintp')
		del(RowId)
		RowId = np.arange(NewDataBin.QuantNegSamp.shape[0])
		NewDataBin.NegSampITF = (NewDataBin.QuantNegSamp.ctypes.data + NewDataBin.QuantNegSamp.strides[0] * RowId).astype('uintp')
		del(RowId)

		return NewDataBin


	#Shallow copy
	def __copy__(self):
		NewDataBin = DataBin(self.PosSamp, self.NegSamp, self.PosWt, self.NegWt, self.nBins)
		NewDataBin.QuantPosSamp = self.QuantPosSamp
		NewDataBin.QuantNegSamp = self.QuantNegSamp
		NewDataBin.xMin = self.xMin
		NewDataBin.xMax = self.xMax
		NewDataBin.Quant = self.Quant
		NewDataBin.PosSampITF = self.PosSampITF
		NewDataBin.NegSampITF = self.NegSampITF


	def __deepcopy__(self):
		return self.Copy()


	def Quantize(self):
		if self.Quant == True:
			return

		#find minimum values of each feature
		PosxMin = np.min(self.PosSamp, axis = 0)
		NegxMin = np.min(self.NegSamp, axis = 0)
		self.xMin = np.where(PosxMin < NegxMin, PosxMin, NegxMin)
		del(PosxMin)
		del(NegxMin)
		PosxMax = np.max(self.PosSamp, axis = 0)
		NegxMax = np.max(self.NegSamp, axis = 0)
		self.xMax = np.where(PosxMax > NegxMax, PosxMax, NegxMax)
		del(PosxMax)
		del(NegxMax)

		#Quantize to 0 ~ nBins-1
		QuantPosSamp = (self.PosSamp - self.xMin) / (self.xMax - self.xMin) * (self.nBins - 1)
		self.QuantPosSamp = QuantPosSamp.astype('uint8')
		del(QuantPosSamp)

		QuantNegSamp = (self.NegSamp - self.xMin) / (self.xMax - self.xMin) * (self.nBins - 1)
		self.QuantNegSamp = QuantNegSamp.astype('uint8')
		del(QuantNegSamp)

		#Construct 1D-Pointer array interface
		RowId = np.arange(self.QuantPosSamp.shape[0])
		self.PosSampITF = (self.QuantPosSamp.ctypes.data + self.QuantPosSamp.strides[0] * RowId).astype('uintp')
		del(RowId)
		RowId = np.arange(self.QuantNegSamp.shape[0])
		self.NegSampITF = (self.QuantNegSamp.ctypes.data + self.QuantNegSamp.strides[0] * RowId).astype('uintp')
		del(RowId)

		self.Quant = True

		return



class BinaryTree(object):
	__slots__ = ('pTree', 'BestStumpFunc', 'ApplyFunc', 'Tree', 'Err')

	#Initialize trainning parameters
	def __init__(self, **pTree):
		'''	
			pTree      - Trainning parameters
			dict key(type: str):
				nBins      - [256] maximum number of quanizaton bins (<=256)
				MaxDepth   - [1] maximum depth of tree
				MinWeight  - [.01] minimum sample weigth to allow split
				FracFtrs   - [1] fraction of features numbers to sample for each node split
				nThreads   - [16] max number of computational threads to use

			BestStumpFunc	- BestStump.c interface function
			ApplyFunc		- BinaryTreeApply.c interface function
		'''
		try:
			self.pTree = TrainParamBin()
			for key, value in pTree.items():
				setattr(self.pTree, key, value)

			self.BestStumpFunc = None
			self.ApplyFunc = None
			self.Tree = None
			self.Err = None
		except AttributeError as e:
			print('No Such parameter:', key)
			raise e
		finally:
			assert self.pTree.FracFtrs <= 1

	#Load the C-function BestStump.c and construct ctype interface
	def LoadBestStumpFunc(self, path = '.'):
		if self.BestStumpFunc is not None:
			return

		if not isinstance(path, str):
			print("Parameter 'path' is required to be str.")
			raise TypeError

		f = npcl.load_library('BestStump', path)
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
		
		self.BestStumpFunc = f.BestStump


	#Given data, find the best stump classifier.
	#Wrap the function in C. Not recommanded to use alone.
	def BestStump(self, data, NodePosWt, NodeNegWt, StpFtrsId = None, prior = None):
		if not isinstance(data, DataBin):
			print('DataBin object type is required.')
			raise TypeError

		if not isinstance(NodePosWt, np.ndarray):
			print('numpy.ndarray object type is required.')
			raise TypeError

		if not isinstance(NodeNegWt, np.ndarray):
			print('numpy.ndarray object type is required.')
			raise TypeError

		if not prior:
			PosWtSum = np.sum(PosWt)
			NegWtSum = np.sum(NegWt)
			WtSum = PosWtSum + NegWtSum
			prior = PosWtSum / WtSum

		if not data.Quant:
			data.Quantize()

		if not isinstance(StpFtrsId, np.ndarray):
			StpFtrsId = np.arange(data.PosSamp.shape[1], dtype = 'uint32')

		if self.BestStumpFunc is None:
			self.LoadBestStumpFunc()

		(NP, F) = data.PosSamp.shape
		NN = data.NegSamp.shape[0]
		StpThrs = np.zeros(floor(self.pTree.FracFtrs * F), dtype = 'uint8')
		StpErrs = np.zeros(floor(self.pTree.FracFtrs * F), dtype = 'float64')
		#Best Stump(For effiency, Coding in C)
		self.BestStumpFunc(
			data.PosSampITF,
			data.NegSampITF,
			NodePosWt,
			NodeNegWt,
			ctypes.c_int(NP),
			ctypes.c_int(NN),
			StpFtrsId,
			ctypes.c_int( floor(self.pTree.FracFtrs * F) ),
			ctypes.c_double(prior),
			ctypes.c_int(self.pTree.nBins),
			ctypes.c_int(self.pTree.nThreads),
			StpErrs,
			StpThrs
			)

		return StpErrs, StpThrs



	def Train(self, data):
		'''
			Reference to piotr dollar's Computer Vision matlab toolbox

			INPUTS
				data       - Type: DataBin. Data for training tree.

			OUTPUTS
			Tree       - (dict)learned decision tree model struct with the following keys:
				fids       - [Kx1] feature ids for each node
				thrs       - [Kx1] threshold corresponding to each fid
				child      - [Kx1] index of child for each node (1-indexed)
				hs         - [Kx1] log ratio (.5*log(p/(1-p)) at each node
				weights    - [Kx1] total sample weight at each node
				depth      - [Kx1] depth of each node
			data       - data used for training tree (quantized version of input)
			Err        - decision tree training error
		'''
		if not isinstance(data, DataBin):
			print('DataBin object type is required.')
			raise TypeError

		if not data.Quant:
			data.Quantize()

		#Initialize arrays
		(NP, FP) = data.PosSamp.shape
		(NN, FN) = data.NegSamp.shape
		assert FP == FN
		F = FP

		Tree = dict()
		MaxNodes = 2**(self.pTree.MaxDepth + 1) - 1							#Maximum number of nodes in BinaryTree
		Tree['fids'] = np.zeros(MaxNodes, dtype = 'uint32')
		Tree['thrs'] = np.zeros(MaxNodes, dtype = 'float64')   
		Tree['child'] = np.zeros(MaxNodes, dtype = 'uint32')
		Tree['hs'] = np.zeros(MaxNodes, dtype = 'float64')
		Tree['weights'] = np.zeros(MaxNodes, dtype = 'float64')
		Tree['depth'] = np.zeros(MaxNodes, dtype = 'uint32')
		errs = np.zeros(MaxNodes, dtype = 'float64')

		#Train Decision Tree
		CurNode = 0                        #Current Node's id
		LastNode = 1					   #Last Node's id that has been yield
		NodePosWtList = [None] * MaxNodes	   #an assemble of nodes' samples weight(if a sample does not reaches the node, its weight = 0)
		NodeNegWtList = [None] * MaxNodes
		NodePosWtList[0] = data.PosWt
		NodeNegWtList[0] = data.NegWt

		while CurNode < LastNode:
			NodePosWt = NodePosWtList[CurNode]
			NodeNegWt = NodeNegWtList[CurNode]
			NodePosWtList[CurNode] = None
			NodeNegWtList[CurNode] = None
			NodePosWtSum = np.sum(NodePosWt)
			NodeNegWtSum = np.sum(NodeNegWt)
			NodeWtSum = NodePosWtSum + NodeNegWtSum

			Tree['weights'][CurNode] = NodeWtSum
			prior = NodePosWtSum / NodeWtSum
			errs[CurNode] = min(prior, 1 - prior)
			constant = np.e**8 / (1 + np.e**8)
			alpha =  4.0 if (prior > constant) else \
					-4.0 if (prior < 1 - constant) else \
					0.5 * log(prior / (1 - prior))
			Tree['hs'][CurNode] = alpha
			#alpha = 0.5 * log(prior / (1 - prior))
			#tree['hs'][CurNode] = max(-4.0, min(4.0, alpha))

			#Node's classification is nearly pure, node's depth is out of scale, sum of node samples' weight is out of scale
			if (prior < 1e-3 or prior > 1 - 1e-3) or (Tree['depth'][CurNode] >= self.pTree.MaxDepth) or (NodeWtSum < self.pTree.MinWeight) :
				CurNode += 1
				continue

			#Find best tree stump
			#wheather subsample the features or not
			if self.pTree.FracFtrs < 1:
				StpFtrsId = np.choice(np.arange(F), floor(self.pTree.FracFtrs * F)).astype('uint32')
			else: 
				StpFtrsId = np.arange(F, dtype = 'uint32')

			(StpErrs, StpThrs) = self.BestStump(data, NodePosWt/NodeWtSum, NodeNegWt/NodeWtSum, StpFtrsId, prior)

			BestFtrsId = np.argmin(StpErrs)
			BestThrs = StpThrs[BestFtrsId] + 0.5
			BestFtrsId = StpFtrsId[BestFtrsId]

			#Split node
			LeftCldPosWt = data.QuantPosSamp[:, BestFtrsId] < BestThrs 		#Node's left child's positive samples' weights
			LeftCldNegWt = data.QuantNegSamp[:, BestFtrsId] < BestThrs
			if (np.any(LeftCldPosWt) or np.any(LeftCldNegWt))  and  (np.any(~LeftCldPosWt) or np.any(~LeftCldNegWt)):		#Invalid stump classifier
				#Inverse quantization
				BestThrs = data.xMin[BestFtrsId] + BestThrs * (data.xMax[BestFtrsId] - data.xMin[BestFtrsId]) / (self.pTree.nBins - 1)
				NodePosWtList[LastNode] = LeftCldPosWt * NodePosWt
				NodeNegWtList[LastNode] = LeftCldNegWt * NodeNegWt
				NodePosWtList[LastNode + 1] = (~LeftCldPosWt) * NodePosWt
				NodeNegWtList[LastNode + 1] = (~LeftCldNegWt) * NodeNegWt

				Tree['thrs'][CurNode] = BestThrs
				Tree['fids'][CurNode] = BestFtrsId
				Tree['child'][CurNode] = LastNode
				Tree['depth'][LastNode : LastNode + 2] = Tree['depth'][CurNode] + 1

				LastNode += 2

			CurNode += 1

		#Modefy parameter 'tree':
		Tree['fids'] = Tree['fids'][0:LastNode].copy()
		Tree['thrs'] = Tree['thrs'][0:LastNode].copy()
		Tree['child'] = Tree['child'][0:LastNode].copy()
		Tree['hs'] = Tree['hs'][0:LastNode].copy()
		Tree['weights'] = Tree['weights'][0:LastNode].copy()
		Tree['depth'] = Tree['depth'][0:LastNode].copy()
		Err = np.sum(errs[0:LastNode] * Tree['weights'] * (Tree['child'] == 0))				#Sum up the leaf nodes' error

		#return
		self.Tree = Tree
		self.Err = Err


	def LoadApplyFunc(self, path = '.'):
		if self.ApplyFunc is not None:
			return

		if not isinstance(path, str):
			print("Parameter 'path' is required to be str.")
			raise TypeError

		f = npcl.load_library('BinaryTreeApply', '.')
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

		self.ApplyFunc = f.BinaryTreeApply


	#Apply the binary decision tree and classify the given data
	#
	def Apply(self, data):
		if self.Tree is None:
			print('Binary tree has not been trained.')
			raise ValueError

		if not isinstance(data, np.ndarray):
			print('numpy.ndarray type is required.')
			raise TypeError

		if data.ndim == 1:
			data = data.copy()
			data = data.reshape(-1, 1)
		elif data.ndim > 2:
			print('2-dimension data is required.')
			raise ValueError

		RowId = np.arange(data.shape[0])
		dataITF = (data.ctypes.data + RowId * data.strides[0]).astype('uintp')

		if self.ApplyFunc is None:
			self.LoadApplyFunc()

		results = np.arange(data.shape[0], dtype = 'float64')

		self.ApplyFunc(
			dataITF, 
			ctypes.c_uint32(data.shape[0]),
			self.Tree['fids'],
			self.Tree['thrs'],
			self.Tree['child'],
			self.Tree['hs'],
			ctypes.c_int(self.pTree.nThreads),
			results
			)

		return results


=======
from math import log, floor
from numpy import ctypeslib as npcl
import numpy as np
import ctypes

#Trainning parameters class, used for storing trainning parameters.
class TrainParamBin(object):
	#Limit the parameters
	__slots__ = ('nBins', 'MaxDepth', 'MinWeight', 'FracFtrs', 'nThreads')

	def __init__(self):
		#Default
		self.nBins = 256
		self.MaxDepth = 1
		self.MinWeight = 0.01
		self.FracFtrs = 1.0
		self.nThreads = 16


#Data bin class, used for storing trainning data.
class DataBin(object):
	'''
		Attributes:
			PosSamp         - [NPxF] negative feature vectors(Each row represents a sample, and each column represents a feature)
			NegSamp         - [NNxF] positive feature vectors(Each row represents a sample, and each column represents a feature)
			QuantPosSamp	- [NPxF] quantized positive samples
			QuantNegSamp	- [NNxF] quantized negative samples
			PosSampITF	   - [NPx1]quantized positive feature row vectors pointer interface(used in BestStump.c)
			NegSampITF	   - [NNx1]quantized negative feature row vectors pointer interface(used in BestStump.c)
			PosWt       - [NPx1] positive samples weights
			NegWt       - [NNx1] negative samples weights
			xMin       - [1xF] optional vals defining feature quantization
			xMax      - [1xF] optional vals defining feature quantization
			Quant 		- Quantization flag
			nBins 		- Number of Quantization bins
	'''
	__slots__ = ('PosSamp', 'NegSamp', 'PosWt', 'NegWt', 'PosSampITF', 'NegSampITF', 'QuantPosSamp', 'QuantNegSamp', 'xMin', 'xMax', 'Quant', 'nBins')

	def __init__(self, PosSamp, NegSamp, PosWt = None, NegWt = None, nBins = 256):
		#Check type
		if isinstance(PosSamp, np.ndarray):
			self.PosSamp = PosSamp
		else:
			print('np.ndarray is required.')
			raise TypeError
		if isinstance(NegSamp, np.ndarray):
			self.NegSamp = NegSamp
		else:
			print('np.ndarray is required.')
			raise TypeError

		#Check the shape of Pos and Neg
		if PosSamp.shape[1] != NegSamp.shape[1]:
			print("Postive samples' shape", PosSamp.shape, "does not match negtive samples'", NegSamp.shape)
			raise ValueError

		#Check data type
		if isinstance(PosWt, np.ndarray):
			self.PosWt = PosWt
		elif PosWt is None:
			NP = self.PosSamp.shape[0]
			self.PosWt = np.ones(NP, dtype = 'float64') / NP
		else:
			print('np.ndarray is required.')
			raise TypeError

		if isinstance(NegWt, np.ndarray):
			self.NegWt = NegWt
		elif NegWt is None:
			NN = self.NegSamp.shape[0]
			self.NegWt = np.ones(NN, dtype = 'float64') / NN
		else:
			print('np.ndarray is required.')
			raise TypeError

		#Check the sum of weights
		w = np.sum(self.PosWt) + np.sum(self.NegWt)
		if abs(w - 1) > 1e-3:
			self.PosWt /= w
			self.NegWt /= w

		if  isinstance(nBins, int):
			self.nBins = nBins
		else:
			print('nBins:', 'type int is required.')
			raise TypeError

		self.PosSampITF = None 
		self.NegSampITF = None
		self.QuantPosSamp = None
		self.QuantNegSamp = None
		self.xMin = None
		self.xMax = None
		self.Quant = False


	#Construct a brand-new data bin(Deep copy).
	def Copy(self):
		NewDataBin = DataBin(self.PosSamp.copy(), self.NegSamp.copy(), self.PosWt.copy(), self.NegWt.copy(), self.nBins)

		if isinstance(self.QuantPosSamp, np.ndarray):
			NewDataBin.QuantPosSamp = self.QuantPosSamp.copy()
		else: 
			NewDataBin.QuantPosSamp = None

		if isinstance(self.QuantNegSamp, np.ndarray):
			NewDataBin.QuantNegSamp = self.QuantNegSamp.copy()
		else: 
			NewDataBin.QuantNegSamp = None

		if isinstance(self.xMin, np.ndarray):
			NewDataBin.xMin = self.xMin.copy()
		else: 
			NewDataBin.xMin = None

		if isinstance(self.xMax, np.ndarray):
			NewDataBin.xMax = self.xMax.copy()
		else: 
			NewDataBin.xMax = None

		NewDataBin.Quant = self.Quant

		#Construct 1D-Pointer array interface
		RowId = np.arange(NewDataBin.QuantPosSamp.shape[0])
		NewDataBin.PosSampITF = (NewDataBin.QuantPosSamp.ctypes.data + NewDataBin.QuantPosSamp.strides[0] * RowId).astype('uintp')
		del(RowId)
		RowId = np.arange(NewDataBin.QuantNegSamp.shape[0])
		NewDataBin.NegSampITF = (NewDataBin.QuantNegSamp.ctypes.data + NewDataBin.QuantNegSamp.strides[0] * RowId).astype('uintp')
		del(RowId)

		return NewDataBin


	#Shallow copy
	def __copy__(self):
		NewDataBin = DataBin(self.PosSamp, self.NegSamp, self.PosWt, self.NegWt, self.nBins)
		NewDataBin.QuantPosSamp = self.QuantPosSamp
		NewDataBin.QuantNegSamp = self.QuantNegSamp
		NewDataBin.xMin = self.xMin
		NewDataBin.xMax = self.xMax
		NewDataBin.Quant = self.Quant
		NewDataBin.PosSampITF = self.PosSampITF
		NewDataBin.NegSampITF = self.NegSampITF


	def __deepcopy__(self):
		return self.Copy()


	def Quantize(self):
		if self.Quant == True:
			return

		#find minimum values of each feature
		PosxMin = np.min(self.PosSamp, axis = 0)
		NegxMin = np.min(self.NegSamp, axis = 0)
		self.xMin = np.where(PosxMin < NegxMin, PosxMin, NegxMin)
		del(PosxMin)
		del(NegxMin)
		PosxMax = np.max(self.PosSamp, axis = 0)
		NegxMax = np.max(self.NegSamp, axis = 0)
		self.xMax = np.where(PosxMax > NegxMax, PosxMax, NegxMax)
		del(PosxMax)
		del(NegxMax)

		#Quantize to 0 ~ nBins-1
		QuantPosSamp = (self.PosSamp - self.xMin) / (self.xMax - self.xMin) * (self.nBins - 1)
		self.QuantPosSamp = QuantPosSamp.astype('uint8')
		del(QuantPosSamp)

		QuantNegSamp = (self.NegSamp - self.xMin) / (self.xMax - self.xMin) * (self.nBins - 1)
		self.QuantNegSamp = QuantNegSamp.astype('uint8')
		del(QuantNegSamp)

		#Construct 1D-Pointer array interface
		RowId = np.arange(self.QuantPosSamp.shape[0])
		self.PosSampITF = (self.QuantPosSamp.ctypes.data + self.QuantPosSamp.strides[0] * RowId).astype('uintp')
		del(RowId)
		RowId = np.arange(self.QuantNegSamp.shape[0])
		self.NegSampITF = (self.QuantNegSamp.ctypes.data + self.QuantNegSamp.strides[0] * RowId).astype('uintp')
		del(RowId)

		self.Quant = True

		return



class BinaryTree(object):
	__slots__ = ('pTree', 'BestStumpFunc', 'ApplyFunc', 'Tree', 'Err')

	#Initialize trainning parameters
	def __init__(self, **pTree):
		'''	
			pTree      - Trainning parameters
			dict key(type: str):
				nBins      - [256] maximum number of quanizaton bins (<=256)
				MaxDepth   - [1] maximum depth of tree
				MinWeight  - [.01] minimum sample weigth to allow split
				FracFtrs   - [1] fraction of features numbers to sample for each node split
				nThreads   - [16] max number of computational threads to use

			BestStumpFunc	- BestStump.c interface function
			ApplyFunc		- BinaryTreeApply.c interface function
		'''
		try:
			self.pTree = TrainParamBin()
			for key, value in pTree.items():
				setattr(self.pTree, key, value)

			self.BestStumpFunc = None
			self.ApplyFunc = None
			self.Tree = None
			self.Err = None
		except AttributeError as e:
			print('No Such parameter:', key)
			raise e
		finally:
			assert self.pTree.FracFtrs <= 1

	#Load the C-function BestStump.c and construct ctype interface
	def LoadBestStumpFunc(self, path = '.'):
		if self.BestStumpFunc is not None:
			return

		if not isinstance(path, str):
			print("Parameter 'path' is required to be str.")
			raise TypeError

		f = npcl.load_library('BestStump', path)
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
		
		self.BestStumpFunc = f.BestStump


	#Given data, find the best stump classifier.
	#Wrap the function in C. Not recommanded to use alone.
	def BestStump(self, data, NodePosWt, NodeNegWt, StpFtrsId = None, prior = None):
		if not isinstance(data, DataBin):
			print('DataBin object type is required.')
			raise TypeError

		if not isinstance(NodePosWt, np.ndarray):
			print('numpy.ndarray object type is required.')
			raise TypeError

		if not isinstance(NodeNegWt, np.ndarray):
			print('numpy.ndarray object type is required.')
			raise TypeError

		if not prior:
			PosWtSum = np.sum(PosWt)
			NegWtSum = np.sum(NegWt)
			WtSum = PosWtSum + NegWtSum
			prior = PosWtSum / WtSum

		if not data.Quant:
			data.Quantize()

		if not isinstance(StpFtrsId, np.ndarray):
			StpFtrsId = np.arange(data.PosSamp.shape[1], dtype = 'uint32')

		if self.BestStumpFunc is None:
			self.LoadBestStumpFunc()

		(NP, F) = data.PosSamp.shape
		NN = data.NegSamp.shape[0]
		StpThrs = np.zeros(floor(self.pTree.FracFtrs * F), dtype = 'uint8')
		StpErrs = np.zeros(floor(self.pTree.FracFtrs * F), dtype = 'float64')
		#Best Stump(For effiency, Coding in C)
		self.BestStumpFunc(
			data.PosSampITF,
			data.NegSampITF,
			NodePosWt,
			NodeNegWt,
			ctypes.c_int(NP),
			ctypes.c_int(NN),
			StpFtrsId,
			ctypes.c_int( floor(self.pTree.FracFtrs * F) ),
			ctypes.c_double(prior),
			ctypes.c_int(self.pTree.nBins),
			ctypes.c_int(self.pTree.nThreads),
			StpErrs,
			StpThrs
			)

		return StpErrs, StpThrs



	def Train(self, data):
		'''
			Reference to piotr dollar's Computer Vision matlab toolbox

			INPUTS
				data       - Type: DataBin. Data for training tree.

			OUTPUTS
			Tree       - (dict)learned decision tree model struct with the following keys:
				fids       - [Kx1] feature ids for each node
				thrs       - [Kx1] threshold corresponding to each fid
				child      - [Kx1] index of child for each node (1-indexed)
				hs         - [Kx1] log ratio (.5*log(p/(1-p)) at each node
				weights    - [Kx1] total sample weight at each node
				depth      - [Kx1] depth of each node
			data       - data used for training tree (quantized version of input)
			Err        - decision tree training error
		'''
		if not isinstance(data, DataBin):
			print('DataBin object type is required.')
			raise TypeError

		if not data.Quant:
			data.Quantize()

		#Initialize arrays
		(NP, FP) = data.PosSamp.shape
		(NN, FN) = data.NegSamp.shape
		assert FP == FN
		F = FP

		Tree = dict()
		MaxNodes = 2**(self.pTree.MaxDepth + 1) - 1							#Maximum number of nodes in BinaryTree
		Tree['fids'] = np.zeros(MaxNodes, dtype = 'uint32')
		Tree['thrs'] = np.zeros(MaxNodes, dtype = 'float64')   
		Tree['child'] = np.zeros(MaxNodes, dtype = 'uint32')
		Tree['hs'] = np.zeros(MaxNodes, dtype = 'float64')
		Tree['weights'] = np.zeros(MaxNodes, dtype = 'float64')
		Tree['depth'] = np.zeros(MaxNodes, dtype = 'uint32')
		errs = np.zeros(MaxNodes, dtype = 'float64')

		#Train Decision Tree
		CurNode = 0                        #Current Node's id
		LastNode = 1					   #Last Node's id that has been yield
		NodePosWtList = [None] * MaxNodes	   #an assemble of nodes' samples weight(if a sample does not reaches the node, its weight = 0)
		NodeNegWtList = [None] * MaxNodes
		NodePosWtList[0] = data.PosWt
		NodeNegWtList[0] = data.NegWt

		while CurNode < LastNode:
			NodePosWt = NodePosWtList[CurNode]
			NodeNegWt = NodeNegWtList[CurNode]
			NodePosWtList[CurNode] = None
			NodeNegWtList[CurNode] = None
			NodePosWtSum = np.sum(NodePosWt)
			NodeNegWtSum = np.sum(NodeNegWt)
			NodeWtSum = NodePosWtSum + NodeNegWtSum

			Tree['weights'][CurNode] = NodeWtSum
			prior = NodePosWtSum / NodeWtSum
			errs[CurNode] = min(prior, 1 - prior)
			constant = np.e**8 / (1 + np.e**8)
			alpha =  4.0 if (prior > constant) else \
					-4.0 if (prior < 1 - constant) else \
					0.5 * log(prior / (1 - prior))
			Tree['hs'][CurNode] = alpha
			#alpha = 0.5 * log(prior / (1 - prior))
			#tree['hs'][CurNode] = max(-4.0, min(4.0, alpha))

			#Node's classification is nearly pure, node's depth is out of scale, sum of node samples' weight is out of scale
			if (prior < 1e-3 or prior > 1 - 1e-3) or (Tree['depth'][CurNode] >= self.pTree.MaxDepth) or (NodeWtSum < self.pTree.MinWeight) :
				CurNode += 1
				continue

			#Find best tree stump
			#wheather subsample the features or not
			if self.pTree.FracFtrs < 1:
				StpFtrsId = np.choice(np.arange(F), floor(self.pTree.FracFtrs * F)).astype('uint32')
			else: 
				StpFtrsId = np.arange(F, dtype = 'uint32')

			(StpErrs, StpThrs) = self.BestStump(data, NodePosWt/NodeWtSum, NodeNegWt/NodeWtSum, StpFtrsId, prior)

			BestFtrsId = np.argmin(StpErrs)
			BestThrs = StpThrs[BestFtrsId] + 0.5
			BestFtrsId = StpFtrsId[BestFtrsId]

			#Split node
			LeftCldPosWt = data.QuantPosSamp[:, BestFtrsId] < BestThrs 		#Node's left child's positive samples' weights
			LeftCldNegWt = data.QuantNegSamp[:, BestFtrsId] < BestThrs
			if (np.any(LeftCldPosWt) or np.any(LeftCldNegWt))  and  (np.any(~LeftCldPosWt) or np.any(~LeftCldNegWt)):		#Invalid stump classifier
				#Inverse quantization
				BestThrs = data.xMin[BestFtrsId] + BestThrs * (data.xMax[BestFtrsId] - data.xMin[BestFtrsId]) / (self.pTree.nBins - 1)
				NodePosWtList[LastNode] = LeftCldPosWt * NodePosWt
				NodeNegWtList[LastNode] = LeftCldNegWt * NodeNegWt
				NodePosWtList[LastNode + 1] = (~LeftCldPosWt) * NodePosWt
				NodeNegWtList[LastNode + 1] = (~LeftCldNegWt) * NodeNegWt

				Tree['thrs'][CurNode] = BestThrs
				Tree['fids'][CurNode] = BestFtrsId
				Tree['child'][CurNode] = LastNode
				Tree['depth'][LastNode : LastNode + 2] = Tree['depth'][CurNode] + 1

				LastNode += 2

			CurNode += 1

		#Modefy parameter 'tree':
		Tree['fids'] = Tree['fids'][0:LastNode].copy()
		Tree['thrs'] = Tree['thrs'][0:LastNode].copy()
		Tree['child'] = Tree['child'][0:LastNode].copy()
		Tree['hs'] = Tree['hs'][0:LastNode].copy()
		Tree['weights'] = Tree['weights'][0:LastNode].copy()
		Tree['depth'] = Tree['depth'][0:LastNode].copy()
		Err = np.sum(errs[0:LastNode] * Tree['weights'] * (Tree['child'] == 0))				#Sum up the leaf nodes' error

		#return
		self.Tree = Tree
		self.Err = Err


	def LoadApplyFunc(self, path = '.'):
		if self.ApplyFunc is not None:
			return

		if not isinstance(path, str):
			print("Parameter 'path' is required to be str.")
			raise TypeError

		f = npcl.load_library('BinaryTreeApply', '.')
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

		self.ApplyFunc = f.BinaryTreeApply


	#Apply the binary decision tree and classify the given data
	#
	def Apply(self, data):
		if self.Tree is None:
			print('Binary tree has not been trained.')
			raise ValueError

		if not isinstance(data, np.ndarray):
			print('numpy.ndarray type is required.')
			raise TypeError

		if data.ndim == 1:
			data = data.copy()
			data = data.reshape(-1, 1)
		elif data.ndim > 2:
			print('2-dimension data is required.')
			raise ValueError

		RowId = np.arange(data.shape[0])
		dataITF = (data.ctypes.data + RowId * data.strides[0]).astype('uintp')

		if self.ApplyFunc is None:
			self.LoadApplyFunc()

		results = np.arange(data.shape[0], dtype = 'float64')

		self.ApplyFunc(
			dataITF, 
			ctypes.c_uint32(data.shape[0]),
			self.Tree['fids'],
			self.Tree['thrs'],
			self.Tree['child'],
			self.Tree['hs'],
			ctypes.c_int(self.pTree.nThreads),
			results
			)

		return results


>>>>>>> e007fcd73e53d2a35862013bf9967a1b3c41defb
