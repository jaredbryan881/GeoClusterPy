import numpy as np

def movingAverage(data, window):
	"""Compute the moving average of a 1D or 2D numpy array.

	Args:
		:param data: np.array
			Numpy array containing a well log curve.
		:param window: int
			Size of window for the moving average.
	Returns:
		:return avgData: np.array
			Numpy array containing averaged data.
	"""
	dims = len(data.shape)
	if dims == 1:
		avgData = np.cumsum(data, axis=0, dtype=np.float32)
		avgData[window:] = avgData[window:] - avgData[:-window]
		return avgData[window-1:] / window
	elif dims == 2:
		avgData = np.cumsum(data, axis=1, dtype=np.float32)
		avgData[:, window:] = avgData[:, window:] - avgData[:, :-window]
		return avgData[:, window-1:] / window
	else:
		raise ValueError('Data array has an unexpected number of dimensions {}'.format(dims))
