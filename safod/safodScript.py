import argparse
import numpy as np
import copy
import matplotlib.pyplot as plt

from logIO.Borehole import BoreholeIO

# clean and process data
from processing.cleanLogs import cleanLog
from processing.clusterLogs import clusterLogs
from processing.clusterLogs import normalize
from processing.correlateLogs import correlateLogs
from processing.decomposeLogs import logsPCA
from tools.smoothLogs import movingAverage

# Plotting utilities
from tools.plotLogs import logPlotter
from tools.plotLogs import dbscanClusterPlotter
from tools.plotLogs import hdbscanClusterPlotter
from tools.plotLogs import kmeansClusterPlotter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inFile', type=str, default="../data/safod_data_trimmed.h5")
    parser.add_argument('-o', '--outFile', required=False, type=str)
    parser.add_argument('-s', '--skipHead', required=False, type=int)
    args = parser.parse_args()

    # read well log data from hdf5
    safodIO = BoreholeIO(args.inFile)
    
    data = safodIO.readHDF5()
    
    # average well log data using moving average
    avgData = {}
    for key in list(data.keys()):
        avgData[key] = movingAverage(data[key], window=14)
    print(list(data.keys()))

    logs = ['m2rx', 'Vp', 'Density', 'Gamma', 'SP', 'Porosity', 'Caliper', 'Thorium', 'Uranium', 'm2r1', 'Potassium']
    logs = list(data.keys())
    logs.remove('Depth_m')
    logs.remove('Depth_ft')
    units = ['Ohm-m', 'km/s', 'g/cm^3', 'counts?', 'Who knows', 'Who knows', 'm', 'Counts', 'Counts', 'Ohm-m', 'Counts']
    units=[i for i in range(len(logs))]
    if len(logs) <= 4:
        logPlot = logPlotter(data, 
                             logs=logs,
                             units=units)
	
    # initialize clusterer and prepare data for clustering.
    clusterer = clusterLogs(data, 
                            logs=logs)
    stackedLogs = clusterer.stackLogs()
    stackedLogsNorm = copy.deepcopy(stackedLogs)
    stackedLogsNorm = normalize(stackedLogsNorm, axis=1)

    log_pca, log_trans = logsPCA(stackedLogs, 4)
    pcLab = ["PC #{}".format(i) for i in range(4)]
    for i in range(log_trans.shape[1]):
    	plt.plot(data['Depth_m'], log_trans[:, i], label=pcLab[i])
    plt.legend()
    plt.show()

    catMat = np.concatenate((stackedLogs, log_trans), axis=1)
    catLab = np.concatenate((logs, pcLab))
    corr_matrix = correlateLogs(catMat, catLab, plot=True)

    # -------------------------------
    # -- cluster logs using kmeans --
    # -------------------------------
    km = clusterer.kmeans_cluster(stackedLogsNorm, n_clusters=3) # replace with stackedLogsNorm
    kmClusterPlot = kmeansClusterPlotter(data=stackedLogs,
                                         km_labels=km.labels_,
                                         n_clusters=3,
                                         logs=logs,
                                         units=units)
    kmClusterPlot.cluster2DepthPlotter(depth=data['Depth_m'])
    
    # -------------------------------
    # -- cluster logs using DBSCAN --
    # -------------------------------
    # create knn-graph to find knee-point. This will inform our choice of epsilon for DBSCAN
    clusterer.knGraph(stackedLogsNorm, 10)
    db, clusterStats = clusterer.dbscan_cluster(stackedLogsNorm, eps=0.025, minSamples=25)
    # plot clusters
    dbscanClusterPlot = dbscanClusterPlotter(data=stackedLogs, 
                                             clusterStats=clusterStats,
                                             logs=logs,
                                             units=units,
                                             nonCore=True)
    # reference clusters back to depth                                         
    dbscanClusterPlot.cluster2DepthPlotter(depth=data['Depth_m'])


    # --------------------------------
    # -- cluster logs using HDBSCAN --
    # --------------------------------
    hdb, clusterStats = clusterer.hdbscan_cluster(stackedLogsNorm, minSamples=25, gen_mst=True)
    #hdb.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=50, edge_linewidth=2)
    # plot clusters
    hdbscanClusterPlot = hdbscanClusterPlotter(data=stackedLogs,
                                               clusterStats=clusterStats,
                                               logs=logs,
                                               units=units,
                                               hideOutliers=1.0)
    hdbscanClusterPlot.cluster2DepthPlotter(depth=data['Depth_m'])
    clusterProps = [np.average(hdb.exemplars_[i], axis=0)*nfList[i] for i in range(len(hdb.exemplars_))]


if __name__ == '__main__':
    main()

