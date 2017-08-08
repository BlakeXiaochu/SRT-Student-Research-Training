def BianryTreeTrian(data, **pTree):
	'''
		Reference to piotr dollar's Computer Vision matlab toolbox

		USAGE
		[tree,data,err] = binaryTreeTrain( data, [pTree] )

		INPUTS
			data       - (dict)data for training tree.
			dict key(type: str):
				PosFtrsVec         - [NPxF] negative feature vectors
				NegFtrsVec         - [NNxF] positive feature vectors
				PosFtrsVecToC	   - [NPx1] positive feature row vectors pointer(used in BestStump.c)
				NegFtrsVecToC	   - [NNx1] negative feature row vectors pointer(used in BestStump.c)
				PosWt       - [NPx1] negative weights
				NegWt       - [NNx1] positive weights
				xMin       - [1xF] optional vals defining feature quantization
				xMax      - [1xF] optional vals defining feature quantization
				xType      - [] optional original data type for features
			pTree      - (dict)additional params (struct or name/value pairs)
			dict key(type: str):
				nBins      - [256] maximum number of quanizaton bins (<=256)
				maxDepth   - [1] maximum depth of tree
				minWeight  - [.01] minimum sample weigth to allow split
				fracFtrs   - [1] fraction of features numbers to sample for each node split
				nThreads   - [16] max number of computational threads to use

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
	from math import log, floor
	from numpy import ctypeslib as npcl
	import numpy as np
	import ctypes

	#Merge pTree parameters
	pTree = pTree.copy()
	#default parameters
	pTree_dfs = {'nBins': 256, 'maxDepth': 1, 'minWeight': 0.01, 'fracFtrs': 1, 'nThreads': 16}
	pTree = dict(pTree, **pTree_dfs)

	#Merge data parameters
	data = data.copy()
	data_dfs = {'PosFtrsVec': 'REQ', 'NegFtrsVec': 'REQ', 'PosWt': [], 'NegWt': [], 'xMin': [], 'xMax': [], 'xType':[]}
	data = dict(data, **data_dfs)

	if (not data.has_key('PosFtrsVecToC')) or (not data['PosFtrsVecToC']):
		RowId = np.arange(data['PosFtrsVec'].shape[0])
		data['PosFtrsVecToC'] = (data['PosFtrsVec'].ctypes.data + data['PosFtrsVec'].strides[0] * RowId).astype('uintp')
		del(RowId)
	if (not data.has_key('NegFtrsVecToC')) or (not data['NegFtrsVecToC']):
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
	tree['fids'] = zeros(MaxNodes, dtype = 'uint32')
	tree['thrs'] = zeros(MaxNodes, dtype = 'float64')   #dtype??
	tree['child'] = zeros(MaxNodes, dtype = 'uint32')
	tree['hs'] = zeros(MaxNodes, dtype = 'float64')
	tree['weights'] = zeros(MaxNodes, dtype = 'float64')
	tree['depth'] = zeros(MaxNodes, dtype = 'uint32')

	if not data['PosWt']:
		data['PosWt'] = ones(NP, dtype = 'float64') / (NP + NN)
	if not data['NegWt']:
		data['NegWt'] = ones(NN, dtype = 'float64') / (NN + NP)
	w = np.sum(data['PosWt']) + np.sum(data['NegWt'])
	if abds(w - 1) > 1e-3:
		data['PosWt'] /= w
		data['NegWt'] /= w

	err = zeros(MaxNodes, dtype = 'float64')

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
		data['xType'] = str(data['PosFtrsVec'])
		data['PosFtrsVec'] = (data['PosFtrsVec'] - xMin) / (xMax - xMin) * (pTree['nBins'] - 1)
		data['PosFtrsVec'] = data['PosFtrsVec'].astype('uint8')
		data['NegFtrsVec'] = (data['NegFtrsVec'] - xMin) / (xMax - xMin) * (pTree['nBins'] - 1)
		data['NegFtrsVec'] = data['NegFtrsVec'].astype('uint8')


	#Train Decision Tree
	CurNode = 0                        #Current Node's id
	LastNode = 1					   #Last Node's id that has been yield
	NodePosWtList = [[]] * MaxNodes	   #an assemble of nodes' samples weight(if a sample does not reaches the node, its weight = 0)
	NodeNegWtList = [[]] * MaxNodes
	NodePosWtList[0] = data['PosWt']
	NodeNegWtList[0] = data['NegWt']

	#Load the C-function
	pp = npcl.ndpointer(dtype = 'uintp', ndim = 1, flags = 'C')		#2D pointer(pointer to pointer)
	double_p = npcl.ndpointer(dtype = 'float64', ndim = 1, flags = 'C')
	int_p = npcl.ndpointer(dtype = 'int32', ndim = 1, flags = 'C')
	uint8_p = npcl.ndpointer(dtype = 'uint8', ndim = 1, flags = 'C')
	f = npcl.load_library('BestStump', '.')
	f.BestStump.restype = None
	f.BestStump.argtypes = [pp, pp, double_p, double_p, ctypes.c_int, ctypes.c_int, int_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, double_p, uint8_p]

	while CurNode < LastNode:
		NodePosWt = NodePosWtList[CurNode]
		NodeNegWt = NodeNegWtList[CurNode]
		NodePosWtList[CurNode] = []
		NodeNegWtList[CurNode] = []
		NodePosWtSum = np.sum(NodePosWt)
		NodeNegWtSum = np.sum(NodeNegWt)
		NodeWtSum = NodePosWt + NodePosWt

		tree['weights'][CurNode] = NodeWtSum
		prior = NodeNegWtSum / NodeWtSum
		err[CurNode] = min(prior, 1 - prior)
		alpha = 0.5 * log(prior / (1 - prior))
		tree['hs'][CurNode] = max(-4.0, min(4.0, alpha))

		#Node's classification is nearly pure, node's depth is out of scale, sum of node samples' weight is out of scale
		if (prior < 1e-3 or prior > 1 - 1e-3) or (tree['depth'][CurNode] > pTree['maxDepth']) or (NodeWtSum < pTree['minWeight']) :
			k += 1
			continue

		#Find best tree stump
		#wheather subsample the features or not
		assert pTree['fracFtrs'] <= 1
		if pTree['fracFtrs'] < 1:
			StFtrsId = np.choice(np.arange(F), floor(pTree['fracFtrs'] * F))
		else: 
			StFtrsId = np.arange(F)
		
		#Best Stump(For effiency, Coding in C)!!!!!!!!!!!!!!
		f.BestStump()					

		BestFtrsId = np.argmin(St_errs)
		BestThrs = St_thrs[BestThrs] + 0.5
		BestFtrsId = StFtrsId[BestFtrsId]

		#Split node
		LeftCldPosWt = data['PosFtrsVec'][:, BestFtrsId] < BestThrs 		#Node's left child's positive samples' weights
		LeftCldNegWt = data['NegFtrsVec'][:, BestFtrsId] < BestThrs
		if (np.any(LeftCldPosWt) or np.any(LeftCldNegWt))  or  (np.any(~LeftCldPosWt) or np.any(~LeftCldNegWt)):
			#Inverse quantization
			BestThrs = xMin[BestFtrsId] + BestThrs * (xMax[BestFtrsId] - xMin[BestFtrsId]) / (pTree['nBins'] - 1)
			NodePosWtList[LastNode] = LeftCldPosWt * data['PosWt']
			NodeNegWtList[LastNode] = LeftCldNegWt * data['NegWt']
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

		#return
		return tree, data, err







