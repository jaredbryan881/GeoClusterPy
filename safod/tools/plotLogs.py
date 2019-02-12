import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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
        """Plot a single continuous quantity."""
        plt.plot(self.data[self.logs[0]])

        # add x label
        if self.units[0] is not None:
            plt.xlabel(self.logs[0] + '(' + self.units[0] + ')')
        else:
            plt.xlabel(self.logs[0])
        plt.show()

    def plot2D(self):
        """Plot two quantities as a scatter plot."""
        plt.scatter(self.data[self.logs[0]], self.data[self.logs[1]], s=2)

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
        """Plot three quantities as a scatter plot."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.data[self.logs[0]], 
                   self.data[self.logs[1]], 
                   self.data[self.logs[2]], 
                   s=2)
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
        """Plot three quantities as a scatter plot with a fourth quantity defining the color map."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(self.data[self.logs[0]], 
                       self.data[self.logs[1]], 
                       self.data[self.logs[2]], 
                       s=5, 
                       c=self.data[self.logs[3]],
                       edgecolors='k',
                       linewidth=0.3,
                       cmap='Spectral')

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


class clusterPlotter:
    def __init__(self, data, clusterStats, logs, units, nonCore=True):
        """Initialize the cluster plotter by setting data and cluster statistics.

        Args:
            :param data: np.array
                2D numpy array containing log data.
            :param clusterStats: dict
                Number of core and accessory clustered points, noise points, and labels.
            :param logs: list
                List of log names.
            :param units: list
                List of log units.
            :param nonCore: bool
                Whether to plot noncore cluster points (including noise).
        """
        self.data = data
        self.clusterStats = clusterStats
        self.logs = logs
        self.units = units
        self.nonCore = nonCore

        # set cluster colors
        colorSpace = np.linspace(0, 1, len(self.clusterStats['uniqueLabels']))
        self.colors = [plt.cm.Spectral(l) for l in colorSpace]
		
        plotDim = self.data.shape[1]
		
        if plotDim == 2:
            self.plotClusters2D()
        elif plotDim == 3:
            self.plotClusters3D()
		    
    def plotClusters2D(self):
        """Plot clustered points in 2 dimensions."""
        fig = plt.figure()

        for k, col in zip(self.clusterStats['uniqueLabels'], self.colors):
            # remove black, use for noise instead
            if k == -1:
                col = [0, 0, 0, 1]

            # set group the point belongs to
            classMemberMask = (self.clusterStats['labels'] == k)

            # plot core cluster values
            xyz = self.data[classMemberMask & self.clusterStats['coreSamplesMask']]
            plt.scatter(xyz[:, 0], xyz[:, 1], facecolors=col, edgecolors='k', linewidth=0.3, s=3)

            # plot non core cluster values
            if self.nonCore is True:
                xyz = self.data[classMemberMask & ~self.clusterStats['coreSamplesMask']]
                plt.scatter(xyz[:, 0], xyz[:, 1], facecolors=col, edgecolors='k', linewidth=0.3, s=3)

        # add axis labels and units
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
	    
    def plotClusters3D(self):
        """Plot clustered points in 3 dimensions."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for k, col in zip(self.clusterStats['uniqueLabels'], self.colors):
            # remove black, use for noise instead
            if k == -1:
                col = [0, 0, 0, 1]

            # set group the point belongs to
            classMemberMask = (self.clusterStats['labels'] == k)
            
            # plot core cluster values
            xyz = self.data[classMemberMask & self.clusterStats['coreSamplesMask']]
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], facecolors=col, edgecolors='k', linewidth=0.3, s=3)

            # plot non core cluster values
            if self.nonCore is True:
                xyz = self.data[classMemberMask & ~self.clusterStats['coreSamplesMask']]
                ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], facecolors=col, edgecolors='k', linewidth=0.3, s=3)

        # add axis labels and units	
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


