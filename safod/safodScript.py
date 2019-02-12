import argparse

from logIO.Borehole import BoreholeIO

from processing.cleanLogs import cleanLog
from processing.clusterLogs import clusterLogs
from processing.clusterLogs import normalize

from tools.smoothLogs import movingAverage
from tools.plotLogs import logPlotter
from tools.plotLogs import clusterPlotter

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--inFile', required=True, type=str)
	parser.add_argument('-o', '--outFile', required=False, type=str)
	parser.add_argument('-s', '--skipHead', required=False, type=int)
	args = parser.parse_args()

	safodIO = BoreholeIO(args.inFile)
	data = safodIO.readHDF5()
	avgData = {}
	for key in list(data.keys()):
		avgData[key] = movingAverage(data[key], window=14)
	print(list(data.keys()))
	
	logPlot = logPlotter(data, 
	                     logs=['m2rx', 'Vp', 'Density', 'Depth_m'],
	                     units = ['Ohm-m', 'km/s', 'g/cm^3', 'm'])
	

	clusterer = clusterLogs(data, 
	                        logs=['m2rx', 'Vp', 'Density'])
	stackedLogs = clusterer.stackLogs()
	stackedLogs = normalize(stackedLogs, axis=1)
	db, clusterStats = clusterer.dbscan_cluster(stackedLogs, eps=0.0275, minSamples=25)
	
	clusterPlot = clusterPlotter(data=stackedLogs, 
	                             clusterStats=clusterStats,
	                             logs=['m2rx', 'Vp', 'Density'],
	                             units=['Ohm-m', 'km/s', 'g/cm^3'],
	                             nonCore=False)


if __name__ == '__main__':
	main()
