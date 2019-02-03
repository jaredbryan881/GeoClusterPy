import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse
import h5py

from logIO.Borehole import BoreholeIO
from processing.Logs import processLog
from tools.smoothLogs import movingAverage
from tools.plotLogs import logPlotter

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--inFile', required=True, type=str)
	parser.add_argument('-o', '--outFile', required=False, type=str)
	parser.add_argument('-s', '--skipHead', required=False, type=int)
	args = parser.parse_args()

	safodIO = BoreholeIO(args.inFile)
	data = safodIO.readHDF5()
	print(list(data.keys()))

	plotter = logPlotter(data, 
                         logs=['m2rx', 'Vp', 'Density', 'Depth_m'],
                         units = ['Ohm-m', 'km/s', 'g/cm^3', 'm'])


if __name__ == '__main__':
	main()
