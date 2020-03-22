import h5py
import pandas

class BoreholeIO:
	def __init__(self, inPath):
		"""Initializes the borehole model by reading the respective well logs.

		Arguments:
			:param inPath: str
				Path to file containing well logs.
		"""
		self.inPath = inPath

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
		csvdf = pandas.read_csv(self.inPath, header=head, sep=',')

		# pass keys to list
		csvKeys = list(csvdf.keys())
		# pass data to np array
		csvData = csvdf.values

		dataDict = {}
		for i, key in enumerate(csvKeys):
			dataDict[key] = csvData[:, i]

		return dataDict

	def readHDF5(self):
		"""Read well log data from hdf5.

		Returns:
			:return data: dict
				Dictionary containing well log data and corresponding log names.
		"""
		hf = h5py.File(self.inPath, 'r')
		keys = list(hf.keys())
		data = {}
		for key in keys:
			data[key] = hf[key][:]
		hf.close()

		return data

	def readExcel(self, sheetname, skiprows):
		"""Read well log data from xlsx
		
		Args:
			:param sheetname: str
				Name of the excel sheet to read.

		Returns:
			:return exdf
		"""
		exdf = pandas.read_excel(self.inPath, sheet_name=sheetname, skiprows=skiprows)
		data = {item[0]:item[1] for item in list(exdf.items())}

		#keys = list(exdf.keys())
		#data = [exdf[key] for key in keys]

		return data

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
			# save units as a dataset attribute
			dset.attrs[dsetNames[dsetName][0]] = dsetNames[dsetName][1]
		hf.close()
