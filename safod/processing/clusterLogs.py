from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class clusterLogs:
	def __init__(self, data, logs):
		self.data = data
		self.logs = logs

	def stackLogs(self):
		logArr = [self.data[log] for log in self.logs]
		stackedLogs = np.stack(logArr, axis=1)

		return stackedLogs

	def dbscan_cluster(self, data, eps, minSamples):
		db = DBSCAN(eps=eps, min_samples=minSamples).fit(data)

		clusterStats = self.dbscan_cluster_stats(db)
		return db, clusterStats
			
	def dbscan_cluster_stats(self, db):
		clusterStats = {}
		# labels
		labels = db.labels_
		clusterStats['labels'] = labels
		# unique labels
		clusterStats['uniqueLabels'] = set(labels)
		# number of clusters
		clusterStats['n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
		# number of noise points
		clusterStats['n_noise'] = list(labels).count(-1)
		# core samples mask
		clusterStats['coreSamplesMask'] = np.zeros_like(labels, dtype=bool)
		clusterStats['coreSamplesMask'][db.core_sample_indices_] = True

		print('Number of clusters: {}'.format(clusterStats['n_clusters']))
		print('Number of noise points: {}'.format(clusterStats['n_noise']))

		return clusterStats


def normalize(data, axis):
	# Set all dataset ranges to [-1, 1]
	for i in range(data.shape[axis]):
		# get largest absolute value of an element
		n_factor = np.max((abs(np.min(data[:, i])), abs(np.max(data[:, i]))))
		data[:, i] /= n_factor
	return data
