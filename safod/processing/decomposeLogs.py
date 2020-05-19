import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, TruncatedSVD, NMF
from sklearn.preprocessing import MinMaxScaler

class logDecomposer:
	def __init__(self, data, ncomponents, rescale=True):
		# rescale data
		if rescale:
			scaler = MinMaxScaler(feature_range=[0,1])
			self.data = scaler.fit_transform(data)
		else:
			self.data = data
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
			:return log_ica: sklearn FastICA object
				ICA of the log data.
			:return log_ica_trans: np.array
				Fit and dimensionally reduced data matrix
		"""
		log_ica = FastICA().fit(self.data)
		ica = FastICA(n_components=self.ncomponents, max_iter=1000)
		log_ica_trans = ica.fit_transform(self.data)

		return log_ica, log_ica_trans

	def logsFactorAnalysis(self):
		"""Decompose a 2D array of log data by factor analysis

			Args:
				:param data: np.array
					Array containing log data

			Returns:
				:return log_fa: sklearn FactorAnalysis object
					Factor Analysis of the log data.
				:return log_fa_trans: np.array
					Fit and dimensionally reduced data matrix
		"""
		log_fa = FactorAnalysis().fit(self.data)
		fa = FactorAnalysis(n_components=self.ncomponents)
		log_fa_trans = fa.fit_transform(self.data)

		return log_fa, log_fa_trans

	def logsTruncatedSVD(self):
		"""Decompose a 2D array of log data by truncated singular value decomposition 

			Args:
				:param data: np.array
					Array containing log data

			Returns:
				:return log_svd: sklearn TruncatedSVD object
					TruncatedSVD of the log data
				:return log_svd_trans: np.array
					Fit and dimensionally reduced data matrix
		"""
		log_svd = FactorAnalysis().fit(self.data)
		svd = TruncatedSVD(n_components=self.ncomponents)
		log_svd_trans = svd.fit_transform(self.data)

		return log_svd, log_svd_trans

	def logsNMF(self):
		"""Decompose a 2D array of log data by nonnegative matrix factorization

			Args:
				:param data: np.array
					Array containing log data

			Returns:
				:return log_nmf: sklearn TruncatedSVD object
					NMF of the log data
				:return log_nmf_trans: np.array
					Fit and dimensionally reduced data matrix
		"""
		log_nmf = NMF().fit(self.data)
		nmf = NMF(n_components=self.ncomponents)
		log_nmf_trans = nmf.fit_transform(self.data)

		return log_nmf, log_nmf_trans