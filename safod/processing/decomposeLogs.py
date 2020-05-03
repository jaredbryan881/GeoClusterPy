import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import MinMaxScaler

class logDecomposer:
	def __init__(self, data, ncomponents):
		# rescale data
		scaler = MinMaxScaler(feature_range=[-1,1])
		self.data = scaler.fit_transform(data)
		self.ncomponents = ncomponents

	def logsPCA(self, plot=False):
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
		log_pca = PCA().fit(self.data)
		pca = PCA(n_components=self.ncomponents)
		log_pca_trans = pca.fit_transform(self.data)

		if plot:
			plt.figure()
			plt.plot(np.cumsum(log_pca.explained_variance_ratio_))
			plt.xlabel('Number of Components')
			plt.ylabel('Variance (%)') #for each component
			plt.show()

		return log_pca, log_pca_trans

	def logsICA(self):
		"""Decompose a 2D array of log data by ICA

		Args:
			:param data: np.array
				Array containing log data

		Returns:
			:return log_pca: sklearn FastICA object
				ICA of the log data. Components and explained variance are attributes of this object.
			:return log_trans: np.array
				Fit and dimensionally reduced data matrix
		"""
		log_ica = FastICA().fit(self.data)
		ica = FastICA(n_components=self.ncomponents)
		log_ica_trans = ica.fit_transform(self.data)

		return log_ica, log_ica_trans