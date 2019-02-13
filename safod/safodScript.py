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
    dbscanClusterPlot.cluster2DepthPlotter(depth=data['Depth_m'])
    
    # cluster logs using HDBSCAN algorithm
    hdb, clusterStats = clusterer.hdbscan_cluster(stackedLogsNorm, minSamples=25, gen_mst=True)
    #hdb.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=50, edge_linewidth=2)
    # plot clusters
    hdbscanClusterPlot = hdbscanClusterPlotter(data=stackedLogs,
                                               clusterStats=clusterStats,
                                               logs=['m2rx', 'Vp', 'Density'],
                                               units=['Ohm-m', 'km/s', 'g/cm^3'],
                                               hideOutliers=1.0)
    hdbscanClusterPlot.cluster2DepthPlotter(depth=data['Depth_m'])

if __name__ == '__main__':
    main()

