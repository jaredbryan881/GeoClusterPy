import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def logsPCA(data, ncomponents):
	"""Decompose a 2D array of log data by PCA

	Args:
		:param data: np.array
			Array containing log data

	Returns:
		:return log_pca: sklearn PCA object
			PCA of the log data. Components and explained variance are attributes of this object.
		:return log_trans: np.array
			Fit and dimensionally reduced data matrix
	"""
	scaler = MinMaxScaler(feature_range=[-1, 1])
	data_rescaled = scaler.fit_transform(data)

	#data_rescaled = data
	log_pca = PCA().fit(data_rescaled)

	plt.figure()
	plt.plot(np.cumsum(log_pca.explained_variance_ratio_))
	plt.xlabel('Number of Components')
	plt.ylabel('Variance (%)') #for each component
	plt.show()
	
	pca = PCA(n_components=ncomponents)
	log_trans = pca.fit_transform(data_rescaled)
	#pca = PCA()
	#log_pca = pca.fit(data)
	#log_trans = pca.fit_transform(data)
	return log_pca, log_trans