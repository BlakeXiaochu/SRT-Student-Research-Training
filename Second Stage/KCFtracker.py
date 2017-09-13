import numpy as np
from numpy.fft import fft2, ifft2

class KCFtracker(object):
	"""
		object tracker of KCF algorithm
	"""
	__slot__ = {'alphaf', 'coefY', '_paddingTarget', '_cosWin'}
	def __init__(self):
		self.alphaf = None
		#coeffient controling The regression targets y
		self.coefY = 10
		self._paddingTarget = None
		self._cosWin = None


	#according to input image patch, train a regression tracker 
	def train(self, paddingRegion, regLambda, method, **kw):
		'''
			paddingRegion	- image patch(region) that contains target
			regLambda		- regularization coeffient λ for ridge regression

			method 			- kernel method
			kw 				- parameters for different kernel method. including following method and corresponding parameter(s):
								0. linear
							  	1. Gaussian: sigma (exp(-1/sigma**2*x))
							  	2. poly: a, b ((x + a)**b)
		'''
		if not isinstance(paddingRegion, np.ndarray):
			raise TypeError('argument 1: numpy.ndarray is required.')

		if not isinstance(regLambda, float):
			raise TypeError('argument 2: float is required.')

		h, w, *_ = paddingRegion.shape

		#the input patches (either raw pixels or extracted feature channels) are weighted by a cosine window
		cosWin = np.reshape(np.hamming(h), [-1, 1]) * np.reshape(np.hamming(w), [1, -1])
		if paddingRegion.ndim == 3:
			cosWin.shape = cosWin.shape + (-1,)
		self._cosWin = cosWin
		paddingRegion = paddingRegion * cosWin
		self._paddingTarget = paddingRegion


		#the regression targets y
		s = np.sqrt(h * w) / self.coefY	#spatial bandwidth s, controling The regression targets y simply follow a Gaussian function
		coord = np.mgrid[-int(h/2):int(h/2 + 0.5), -int(w/2):int(w/2 + 0.5)]
		coordX = coord[0, :, :]
		coordY = coord[1, :, :]
		y = np.exp( -0.5 * (coordX**2 + coordY**2) / (s**2) )
		y = np.roll(y, (int(h/2 + 0.5), int(w/2 + 0.5)), axis = (0, 1))

		#regression
		k = self.kernelCorrelation(paddingRegion, paddingRegion, method, **kw)
		self.alphaf = fft2(y) / (fft2(k) + regLambda)

		

	#run tracker, return relative position with the input image patch
	def detect(self, paddingRegion, method, **kw):
		'''
			
		'''
		if self.alphaf is None:
			raise Exception('tracker has not been trained.')

		#kernel correlation
		kxz = self.kernelCorrelation(paddingRegion * self._cosWin, self._paddingTarget, method, **kw)
		kf = fft2(kxz)

		#regression results in fourier domain
		ff = kf * self.alphaf
		f = np.real(ifft2(ff))
		maxPosition = np.argmax(f)

		#return relative position with the input image pacth
		h, w = paddingRegion.shape[0:2]
		x, y = (maxPosition // w, maxPosition % w)
		rePos = ((x - h) if x > h/2 else x, (y - w) if y > w/2 else y)


		return rePos



	def kernelCorrelation(self, x1, x2, method, **kw):
		if (not isinstance(x1, np.ndarray)) or (not isinstance(x2, np.ndarray)):
			raise TypeError('numpy.ndarray is required.')

		if (x1.ndim == 2) and (x2.ndim == 2):
			single = 1
		elif (x1.ndim == 3) and (x2.ndim == 3):
			single = 0
		else:
			raise ValueError('2 or 3 dimension arrays are required.')

		if x1.shape != x2.shape:
			raise ValueError('the input image pacthes do not match.')

		#linear regression
		if method == 'linear':
			if single:
				x12DotProdf = np.conj(fft2(x1)) * fft2(x2)
			else:
				x12DotProdf = np.sum( np.conj( fft2(x1, axes = (0, 1)) ) * fft2(x2, axes = (0, 1)) , axis = 2)

			x12DotProd = ifft2(x12DotProdf)
			return x12DotProd

		#Gaussian kernel
		elif method == 'gaussian':
			sigma = kw['sigma']
			x1Norm = np.sum(x1**2)
			x2Norm = np.sum(x2**2)

			if single:
				x12DotProdf = np.conj(fft2(x1)) * fft2(x2)
			else:
				x12DotProdf = np.sum( np.conj( fft2(x1, axes = (0, 1)) ) * fft2(x2, axes = (0, 1)) , axis = 2)

			x12DotProd = ifft2(x12DotProdf)
			k = np.exp( -(x1Norm + x2Norm - 2 * x12DotProd.real) / (sigma**2) / x12DotProd.size)
			return k

		#polynomial kernel
		elif method == 'poly':
			a = kw['a']
			b = kw['b']

			if single:
				x12DotProdf = np.conj(fft2(x1)) * fft2(x2)
			else:
				x12DotProdf = np.sum( np.conj( fft2(x1, axes = (0, 1)) ) * fft2(x2, axes = (0, 1)) , axis = 2)
			
			x12DotProd = ifft2(x12DotProdf)
			k = (x12DotProd.real + a)**b
			return k

		#wrong method type
		else:
			raise ValueError('invalid kernel method.')




