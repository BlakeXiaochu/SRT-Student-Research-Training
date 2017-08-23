from numpy import ctypeslib as npcl
import numpy as np
import ctypes
from BinaryTreeTrain import *

data = dict()
pTree = {'nBins': 3, 'maxDepth': 1, 'minWeight': 0.01, 'fracFtrs': 1, 'nThreads': 16}

data['PosFtrsVec'] = np.array([
	[0, 2, 1],
	[2, 1, 1],
	[1, 1, 1],
	[2, 1, 1],
	[1, 2, 0],
	[2, 1, 0],
	[1, 0, 0]
	], dtype = 'uint8')
data['NegFtrsVec'] = np.array([
	[0, 0, 0],
	[1, 0, 0],
	[0, 0, 1]
	], dtype = 'uint8')

data['PosWt'] = np.ones(10, dtype = 'float64')/10
data['NegWt'] = np.ones(10, dtype = 'float64')/10
AddrIndex1 = data['PosFtrsVec'].ctypes.data + np.arange(data['PosFtrsVec'].shape[0]) * data['PosFtrsVec'].strides[0]
AddrIndex2 = data['NegFtrsVec'].ctypes.data + np.arange(data['NegFtrsVec'].shape[0]) * data['NegFtrsVec'].strides[0]
data['PosFtrsVecToC'] = AddrIndex1.astype('uintp')
data['NegFtrsVecToC'] = AddrIndex2.astype('uintp')

pp = npcl.ndpointer(dtype = 'uintp', ndim = 1, flags = 'C')		#2D pointer(pointer to pointer)
double_p = npcl.ndpointer(dtype = 'float64', ndim = 1, flags = 'C')
uint32_p = npcl.ndpointer(dtype = 'uint32', ndim = 1, flags = 'C')
uint8_p = npcl.ndpointer(dtype = 'uint8', ndim = 1, flags = 'C')
f = npcl.load_library('BestStump.dll', '.')
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
	double_p, 
	uint8_p
	]

err = np.zeros(3, dtype = 'float64')
thrs = np.zeros(3, dtype = 'uint8')

f.BestStump(
	data['PosFtrsVecToC'],
	data['NegFtrsVecToC'],
	data['PosWt'],
	data['NegWt'],
	7,
	3,
	np.arange(3, dtype = 'uint32'),
	3,
	ctypes.c_double(0.7),
	pTree['nBins'],
	err,
	thrs
	)

print(err)
print(thrs)
