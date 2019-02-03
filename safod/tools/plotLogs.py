import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class logPlotter:
	def __init__(self, data, logs, units=[]):
		"""Initialize log plotter by setting data and logs to plot.

		Args:
			:param data: dict
				Dictionary containing well log data and corresponding labels.
			:param logs: list
				List containing names of well logs to plot.
			:param units: list
				List of strings containing units of plotted quantites.
		"""
		self.data = data
		self.logs = logs
		self.units = units

		# set dimension of the plot based on the number of logs provided
		plotDim = len(self.logs)

		# set missing values of units to None
		for i in range(plotDim):
			try:
				val = self.units[i]
			except:
				self.units.append(None)

		if plotDim == 1:
			self.plot1D()
		elif plotDim == 2:
			self.plot2D()
		elif plotDim == 3:
			self.plot3D()
		elif plotDim == 4:
			self.plot4D()
		else:
			raise ValueError("Plot dimension specified is not viewable by mere mortals.")

	def plot1D(self):
		"""Plot a single quantity horizontally."""
		plt.plot(self.data[self.logs[0]])

		# add x label
		if self.units[0] is not None:
			plt.xlabel(self.logs[0] + '(' + self.units[0] + ')')
		else:
			plt.xlabel(self.logs[0])
		plt.show()

	def plot2D(self):
		"""Plot two quantities."""
		plt.plot(self.data[self.logs[0]], self.data[self.logs[1]])

		# add x label
		if self.units[0] is not None:
			plt.xlabel(self.logs[0] + '(' + self.units[0] + ')')
		else:
			plt.xlabel(self.logs[0])
		# add y label
		if self.units[1] is not None:
			plt.xlabel(self.logs[1] + '(' + self.units[1] + ')')
		else:
			plt.xlabel(self.logs[1])
		plt.show()

	def plot3D(self):
		"""Plot three quantities."""
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(self.data[self.logs[0]], 
                   self.data[self.logs[1]], 
                   self.data[self.logs[2]], 
                   s=1)
		# add x label
		if self.units[0] is not None:
			ax.set_xlabel(self.logs[0] + '(' + self.units[0] + ')')
		else:
			ax.set_xlabel(self.logs[0])
		# add y label
		if self.units[1] is not None:
			ax.set_ylabel(self.logs[1] + '(' + self.units[1] + ')')
		else:
			ax.set_ylabel(self.logs[1])
		# add z label
		if self.units[2] is not None:
			ax.set_zlabel(self.logs[2] + '(' + self.units[2] + ')')
		else:
			ax.set_zlabel(self.logs[2])
		plt.show()
		
		

	def plot4D(self):
		"""Plot three quantities with a fourth quantity defining the color map."""
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		p = ax.scatter(self.data[self.logs[0]], 
                       self.data[self.logs[1]], 
                       self.data[self.logs[2]], 
                       s=1, 
                       c=self.data[self.logs[3]],
                       cmap='coolwarm')

		# add x label
		if self.units[0] is not None:
			ax.set_xlabel(self.logs[0] + '(' + self.units[0] + ')')
		else:
			ax.set_xlabel(self.logs[0])
		# add y label
		if self.units[1] is not None:
			ax.set_ylabel(self.logs[1] + '(' + self.units[1] + ')')
		else:
			ax.set_ylabel(self.logs[1])
		# add z label
		if self.units[2] is not None:
			ax.set_zlabel(self.logs[2] + '(' + self.units[2] + ')')
		else:
			ax.set_zlabel(self.logs[2])
		# label colormap
		cbar = fig.colorbar(p)
		if self.units[3] is not None:
			cbar.set_label(self.logs[3] + '(' + self.units[3] + ')')
		else:
			cbar.set_label(self.logs[3])
		plt.show()




