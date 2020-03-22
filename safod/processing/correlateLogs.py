import matplotlib.pyplot as plt
import numpy as np

def correlateLogs(data, labels, plot=True):
	"""Compute the correlation matrix for a given set of log data.

	Args:
		:param data: np.array
			Array containing log data
		:param logs: list
			Log names used for plotting and identification
		:param plot: Bool
			Whether or not to plot the correlation matrix

	Returns:
		:return cc_matrix: np.array
			Array containing correlation coefficients for each combination of logs
	"""
	cc_matrix = np.corrcoef(data.transpose())

	if plot:
		fig, ax = plt.subplots()
		plt.imshow(cc_matrix)
		ticks = np.arange(0,len(labels))
		ax.set_xticks(ticks)
		ax.set_xticklabels(labels, rotation=45)
		ax.set_yticks(ticks)
		ax.set_yticklabels(labels)
		plt.show()

	return cc_matrix
