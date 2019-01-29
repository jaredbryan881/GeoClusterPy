import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas
import h5py


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--inFile', required=False, default='safod_data.csv', type=str)
	parser.add_argument('-o', '--outFile', required=False, default='safod_data.h5', type=str)
	parser.add_argument('-s', '--skipHead', required=False, default=79, type=int)
	args = parser.parse_args()

	logs = {'Uranium (PPM)': ['Uranium', 'ppm'],
	        'Caliper (in)': ['Caliper', 'in'],
	        'ZDL Density (g/cm^3)': ['ZDLDen', 'g/cm^3'],
	        'Vp/Vs': ['VpVs', ''],
	        'Thorium (PPM)': ['Thorium', 'ppm'],
	        'Spontaneous Potential (mV)': ['SP', 'mV'],
	        'ZDL correction (g/cm^3)': ['ZDLCorr', 'g/cm^3'],
	        'Depth (ft)': ['Depth_ft', 'ft'],
	        'Depth (m)': ['Depth', 'm'],
	        'Vs (km/s)': ['Vs', 'km/s'],
	        'Vp (km/s)': ['Vp', 'km/s'],
	        'Porosity (pu)': ['Porosity', 'pu'],
	        'Gamma ray (GAPI)': ['Gamma', 'GAPI'],
	        'Potassium (%)': ['Potassium', '%'],
	       }

	safod = boreHole(args.inFile)

	data = safod.readHDF5('safod_data.h5')

	cleanVp = safod.cleanData(data, 'Vp')
	avgVp = safod.movingAverage(cleanVp, num=14, dims=2)

	cleanVs = safod.cleanData(data, 'Vs')
	avgVs = safod.movingAverage(cleanVs, num=14, dims=2)

	cleanPoro = safod.cleanData(data, 'Porosity')
	avgPoro = safod.movingAverage(cleanPoro, num=14, dims=2)

	#avgVpTrim, avgVsTrim, avgPoroTrim = safod.depthRangeTrim([cleanVp, cleanVs, cleanPoro])
	trimVp, trimVs, trimPoro = safod.depthRangeTrim([cleanVp, cleanVs, cleanPoro])
	print(trimVp.shape)
	print(trimVs.shape)
	print(trimPoro.shape)
	for i in range(trimVp.shape[0]):
		if trimVp[i] != trimPoro[i]:
			print(i)
	print(list(data.keys()))
	# TODO: the data is not of the correct shape because certain values are not present. For example,
	# the porosity is missing a value that is present in the other logs. You need to clean this better
	# and get rid of the values where there is data missing.


class boreHole():
	def __init__(self, inFile):
		self.inFile = inFile

	def readText(self, head):
		""""Read geophysical well logs from a .las file.

		Args:
			:param head: int
				Number of header lines to skip before reading in data.

		Returns:
			:return dataDict: dict
				Dictionary containing well log data.
		"""
		# read data into pandas dataframe
		csvdf = pandas.read_csv(self.inFile, header=head, sep=',')

		# pass keys to list
		csvKeys = list(csvdf.keys())
		# pass data to np array
		csvData = csvdf.values

		dataDict = {}
		for i, key in enumerate(csvKeys):
			dataDict[key] = csvData[:, i]

		return dataDict

	def writeHDF5(self, outFile, data, dsetNames):
		"""Write geophysical well logs to hdf5.

		Args:
			:param outFile: str
				Filename of output file.
			:param data: dict
				Dictionary containing a collection of well logs.
			:param dsets: dict
				Dictionary containing keys from data dict and corresponding dataset name.
		"""
		hf = h5py.File(outFile, 'w')
		for i, dsetName in enumerate(dsetNames):
			dset = hf.create_dataset(dsetNames[dsetName][0], data=data[dsetName])
			dset.attrs[dsetNames[dsetName][0]] = dsetNames[dsetName][1]
		hf.close()

	def readHDF5(self, inFile):
		"""Read well log data from hdf5.

		Args:
			:param inFile: str
				Filename of input file.
		Returns:
			:return data: np.array
				Numpy array containing well log data.
			:return keys: list
				List containing dataset names.
		"""
		hf = h5py.File(inFile, 'r')
		data = {}
		for key in list(hf.keys()):
			data[key] = hf[key][:]
		hf.close()

		return data

	def cleanData(self, data, log):
		"""Remove NaNs from well logs and set depth range of clean data.

		Args:
			:param data: dict
				Dictionary containing well log data.
			:param log: str
				Name of the well log to clean.
		Returns:
			:return cleanData: np.array
				Depths at valid log entries (axis 0) and cleaned well log data (axis 1).
		"""
		cleanLog = data[log][~np.isnan(data[log])]
		cleanDepth = data['Depth'][~np.isnan(data[log])]
		cleanData = np.stack((cleanDepth, cleanLog))
		return cleanData

	def movingAverage(self, data, num, dims):
		"""Compute the moving average of a 1D or 2D numpy array.

		Args:
			:param data: np.array
				Numpy array containing a well log curve.
			:param num: int
				Size of window for the moving average.
		Returns:
			:return avgData: np.array
				Numpy array containing averaged data.
		"""
		dims = len(data.shape)
		if dims == 1:
			avgData = np.cumsum(data, axis=0, dtype=np.float32)
			avgData[num:] = avgData[num:] - avgData[:-num]
			return avgData[num-1:] / num
		elif dims == 2:
			avgData = np.cumsum(data, axis=1, dtype=np.float32)
			avgData[:, num:] = avgData[:, num:] - avgData[:, :-num]
			return avgData[:, num-1:] / num
		else:
			raise ValueError('Data array has an unexpected number of dimensions {}'.format(dims))

	def depthRangeTrim(self, logs):
		"""Trim a collection of well logs to a uniform depth range.

		Args:
			:param logs: list
				Collection of well logs.
		Returns:
			:return trimmedLogs: list
				Collection of trimmed well logs.
		"""
		startVal = np.amax([log[0, 0] for log in logs])
		print(startVal)
		endVal = np.amin([log[0, -1] for log in logs])
		print(endVal)
		trimmedLogs = []
		for log in logs:
			#print(log[0][(log[0]>startVal) & (log[0]<endVal)])
			#print(log[0])
			trimmedLogs.append(log[0][(log[0]>startVal) & (log[0]<endVal)])
		return trimmedLogs


if __name__ == '__main__':
	main()
