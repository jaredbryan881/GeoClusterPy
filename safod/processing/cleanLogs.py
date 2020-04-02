import numpy as np

class cleanLog:
	def __init__(self, data):
		"""Initialize well log processing and cleaning utilities.

		Args:
			:param data: dict
				Dictionary containing well log data and log names.
		"""
		self.data = data

	def removeNaN(self, log, depth='Depth_m'):
		"""Remove NaNs from well logs and set depth range of clean data.

		Args:
			:param log: str
				Name of the well log to clean.
			:param depth: str
				Key for depth log to use.
		Returns:
			:return cleanData: np.array
				Depths at valid log entries (axis 0) and cleaned well log data (axis 1).
		"""
		# keep only data with no NaNs
		cleanLog = self.data[log][~np.isnan(self.data[log])] 
		# use above NaN values to adjust depth log
		cleanDepth = self.data[depth][~np.isnan(self.data[log])]

		cleanData = np.stack((cleanDepth, cleanLog))

		return cleanData

	def depthRangeTrim(self, logs):
		"""Trim a collection of well logs to a uniform depth range.

		Args:
			:param logs: list
				Collection of well log names.
			:param depth: str
				Key for depth log to use.
		Returns:
			:return trimmedLogs: list
				Collection of trimmed well logs.
		"""
		# assume these logs use the same depth log
		# require that the depth log is in the 0th position of the log list for each log in logs
		startIndex = np.amax([log[0, 0] for log in logs])
		endIndex = np.amin([log[0, -1] for log in logs])

		# repackage trimmed logs into a new list
		trimmedLogs = []
		for log in logs:
			trimmedLogs.append(log[0][(log[0]>startIndex) & (log[0]<endIndex)])

		return trimmedLogs

def match_depths(log, cur_depth, match_depth):
	"""Interpolate log values from those related to cur_depth to a target match_depth.
	
	Args:
		:param log: np.array
			Data for a given geophysical log
		:param cur_depth: np.array
			Depth values corresponding to log
		:param match_depth: np.array
			Target depth values

	Returns:
		:return matched_logs: np.array
			Interpolated log values to match_depth
	"""
	return np.interp(match_depth, cur_depth, log)


