import argparse
import numpy as np
import copy
import matplotlib.pyplot as plt

from logIO.Borehole import BoreholeIO

from processing.cleanLogs import cleanLog
from processing.clusterLogs import clusterLogs
from processing.clusterLogs import normalize

from tools.smoothLogs import movingAverage
from tools.plotLogs import logPlotter
from tools.plotLogs import dbscanClusterPlotter
from tools.plotLogs import hdbscanClusterPlotter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inFile', required=True, type=str)
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
	
    logPlot = logPlotter(data, 
                         logs=['m2rx', 'Vp', 'Density', 'Depth_m'],
                         units = ['Ohm-m', 'km/s', 'g/cm^3', 'm'])

    # initialize clusterer and prepare data for clustering.
    clusterer = clusterLogs(data, 
                            logs=['m2rx', 'Vp', 'Density'])
    stackedLogs = clusterer.stackLogs()
    stackedLogsNorm = copy.deepcopy(stackedLogs)
    stackedLogsNorm = normalize(stackedLogsNorm, axis=1)
    
    # create knn-graph to find knee-point. This will inform our choice of epsilon for DBSCAN
    clusterer.knGraph(stackedLogsNorm, 5)
    # cluster logs using DBSCAN algorithm
    db, clusterStats = clusterer.dbscan_cluster(stackedLogsNorm, eps=0.025, minSamples=25)
	# plot clusters
    dbscanClusterPlot = dbscanClusterPlotter(data=stackedLogs, 
                                             clusterStats=clusterStats,
                                             logs=['m2rx', 'Vp', 'Density'],
                                             units=['Ohm-m', 'km/s', 'g/cm^3'],
                                             nonCore=True)
    # reference clusters back to depth       
    for c in clusterStats['uniqueLabels']:
        d = data['Depth_m'][np.where(clusterStats['labels'] == c)[0]]
        cl = np.ones(d.shape[0]) * c
        plt.scatter(d, cl)
    plt.xlabel('Depth (m)')
    plt.ylabel('Cluster')
    plt.show()
    
    # cluster logs using HDBSCAN algorithm
    hdb, clusterStats = clusterer.hdbscan_cluster(stackedLogsNorm, minSamples=25, gen_mst=True)

    # plot clusters
    hdbscanClusterPlot = hdbscanClusterPlotter(data=stackedLogs,
                                               clusterStats=clusterStats,
                                               logs=['m2rx', 'Vp', 'Density'],
                                               units=['Ohm-m', 'km/s', 'g/cm^3'],
                                               hideOutliers=1.0)

if __name__ == '__main__':
    main()