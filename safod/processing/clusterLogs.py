from sklearn.cluster import DBSCAN
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
import hdbscan
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class clusterLogs:
    def __init__(self, data, logs):
        """Initializes log curve clustering by setting data and logs.
        
        Args:
            :param data: dict
                Dictionary containing log data and associated log names.
            :param logs: list
                List containing log names to cluster.
        """
        self.data = data
        self.logs = logs

    def stackLogs(self):
        """Stack multiple log curves into a multidimensional numpy array.
        
        Returns:
            :return stackedLogs: np.array
                Numpy array containing stacked log data.
        """
        logArr = [self.data[log] for log in self.logs]
        stackedLogs = np.stack(logArr, axis=1)

        return stackedLogs

    def dbscan_cluster(self, data, eps, minSamples):
        """Cluster a dataset using the DBSCAN algorithm.
	    
        Args:
            :param data: np.array
                Data containing log data in n dimensions.
            :param eps: float
                Maximum distance between two points to be considered as in the same neighborhood.
            :param minSamples: int
                Minimum number of points needed to define a cluster.
              
        Returns:
            :return db: DBSCAN object
                DBSCAN clustering object.
            :return clusterStats: dict
                Dictionary containing clustering statistics.
        """
        db = DBSCAN(eps=eps, min_samples=minSamples).fit(data)

        clusterStats = self.dbscan_cluster_stats(db)
        return db, clusterStats

    def dbscan_cluster_stats(self, db):
        """Create dictionary for DBSCAN clustering statistics.
        
        Args: 
            :param db: DBSCAN object
                DBSCAN clustering object.
        
        Returns:
            :return clusterStats: dict
                Dictionary containing clustering statistics.
        """
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

        return clusterStats
        
    def knGraph(self, data, k):
        """Construct a k-nearest neighbors graph to inform choice of DBSCAN eps param.
        
        Args:
            :param data: np.array
                numpy array containing normalized log data
            :param k: int
                number of neighbors.
        """
        nbrs = NearestNeighbors(k).fit(data)
        distances, indices = nbrs.kneighbors(data)
        distanceDec = sorted(distances[:, k-1], reverse=True)
        
        plt.plot(list(range(1, data.shape[0]+1)), distanceDec)
        plt.title('K-Nearest-Neighbors')
        plt.show()
        
    def hdbscan_cluster(self, data, minSamples, gen_mst):
        """Cluster a dataset using the HDBSCAN algorithm.
        
        Args:
            :param data: np.array
                numpy array containing log data.
            :param minSamples: int
                number of samples needed to define a unique cluster.
            :param gen_mst: bool
                whether to generate the minimum spanning tree or not.
                
        Returns:
            :return hdb: HDBSCAN object
                HDBSCAN clustering object
        """
        hdb = hdbscan.HDBSCAN(min_cluster_size=minSamples, gen_min_span_tree=gen_mst).fit(data)
        clusterStats = self.hdbscan_cluster_stats(hdb)
        
        return hdb, clusterStats
        
    def hdbscan_cluster_stats(self, hdb):
        """Create dictionary for HDBSCAN clustering statistics.
        
        Args: 
            :param hdb: HDBSCAN object
                HDBSCAN clustering object.
        
        Returns:
            :return clusterStats: dict
                Dictionary containing clustering statistics.
        """
        clusterStats = {}
        # labels
        labels = hdb.labels_
        clusterStats['labels'] = labels
        # unique labels
        clusterStats['uniqueLabels'] = set(labels)
        # probability of cluster membership
        clusterStats['probabilities'] = hdb.probabilities_
        # cluster persistence over different distance scales
        clusterStats['persistence'] = hdb.cluster_persistence_
        # outlier score
        clusterStats['outliers'] = hdb.outlier_scores_
        
        return clusterStats


def normalize(data, axis):
    """Set all dataset ranges to [-1, 1].
	
    Args:
        :param data: np.array
            numpy array containing log data.
        :param axis: int
            axis to normalize over.
            
    Returns:
        :return data: np.array
            numpy array containing normalized log data.
        :return nfList: list
            list of floats containing the factor used to normalize the data"""
    for i in range(data.shape[axis]):
    	# get largest absolute value of an element
    	n_factor = np.max((abs(np.min(data[:, i])), abs(np.max(data[:, i]))))
    	data[:, i] /= n_factor
    	
    return data

