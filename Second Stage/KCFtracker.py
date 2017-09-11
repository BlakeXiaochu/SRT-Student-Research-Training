class KCFtracker(object):
	import numpy as np
	from numpy.fft import fft2, ifft2
	"""
		object tracker of KCF algorithm
	"""
	__slot__ = {'alphaf', }
	def __init__(self):
		super(Tracker, self).__init__()
		self.arg = arg


	def train(self):
		pass
		

	def detect(self):
		pass


	def kernelCorrelation(self, x1, x2, method, **kw):
		if (not isinstance(x1, np.ndarray)) or (not isinstance(x2, np.ndarray)):
			print('numpy.ndarray is required.')
			raise TypeError

		if (x1.ndim != 2) or (x2.ndim ！= 2):
			print('2-dimension array is required.')
			raise ValueError

		#Gaussian kernel
		if method == 'gaussian':
			sigma = kw['sigma']
			x1Norm = np.sum(x1**2)
			x2Norm = np.sum(x2**2)
			x12DotProdf = np.sum( np.conj(fft2(x1), axes = (1, 2)) * fft2(x2, axes = (1, 2)) , axis = -1)
			x12DotProd = ifft2(x12DotProdf, axes = (1, 2))
			k = np.exp( -(x1Norm + x2Norm - 2 * x12DotProd.real) / (sigma**2) / x12DotProd.size)
			return k

		elif method == 'polynomial':
			a = kw['a']
			b = kw['b']
			x12DotProdf = np.sum( np.conj(fft2(x1), axes = (1, 2)) * fft2(x2, axes = (1, 2)) , axis = -1)
			x12DotProd = ifft2(x12DotProdf, axes = (1, 2))
			k = (x12DotProd.real + a)**b
			return k

		#wrong method type
		else:
			print('invalid kernel method.')
			raise ValueError




