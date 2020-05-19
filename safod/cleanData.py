from logIO.Borehole import BoreholeIO
from processing.cleanLogs import match_depths

import matplotlib.pyplot as plt
import numpy as np

# read/write
safod_geophysData_inPath = "/home/jared/GeoClusterPy/data/SAFOD_data_total.xlsx"
safod_geophysData_outPath = "/home/jared/GeoClusterPy/data/SAFOD_data_cleaned.h5"
safodIO = BoreholeIO(safod_geophysData_inPath)
# read most geophysical data
gp_data = safodIO.readExcel("Geophys. Log Data", 1)
# read resistivity data
res_data = safodIO.readExcel("Geophys. Resistivity", 0)

# These are the logs we keep
gp_logs = np.array((['Caliper (in)', 'Caliper', 'in'], 
				    ['Gamma ray (GAPI)', 'Gamma', 'GAPI'],
				    ['Porosity (pu)', 'Porosity', 'pu'], 
				    ['Spontaneous Potential (mV)', 'SP', 'mV'], 
				    ['ZDL Density (g/cm^3)', 'Density', 'g/cm^3'], 
				    ['Potassium (%)', 'Potassium', '%'],
				    ['Thorium (PPM)', 'Thorium', 'ppm'],
				    ['Uranium (PPM)', 'Uranium', 'ppm'],
				    ['Vp (km/s)', 'Vp', 'km/s'],
				    ['Vs (km/s)', 'Vs', 'km/s'],
				    ['Vp/Vs', 'VpVs', 'nondim']))
res_logs = np.array((['M2R1', 'm2r1', 'ohms'],
					 ['M2R2', 'm2r2', 'ohms'],
					 ['M2R3', 'm2r3', 'ohms'],
					 ['M2R6', 'm2r6', 'ohms'],
					 ['M2R9', 'm2r9', 'ohms'],
					 ['M2RX', 'm2rx', 'ohms']))

# Rename logs
for namePair in gp_logs:
	gp_data[namePair[1]] = gp_data.pop(namePair[0])
for namePair in res_logs:
	res_data[namePair[1]] = res_data.pop(namePair[0])

# define longest complete depth interval
dmin = 3041.5
dmax = 3954.4
newDepthInds = np.where(np.logical_and(gp_data['Depth (m)']<dmax, gp_data['Depth (m)']>dmin))[0]
newDepth = gp_data['Depth (m)'][newDepthInds].values

# interpolate resistivity logs to this depth interval and sampling
joinedLogs = {}
joinedLogs['Depth_m']=newDepth
joinedLogs['Depth_ft']=newDepth*3.28084
for log in res_logs:
	joinedLogs[log[1]] = match_depths(np.log10(np.abs(res_data[log[1]])), res_data['depth(m)'], newDepth)

# move over limited gp_logs
for log in gp_logs:
	joinedLogs[log[1]] = gp_data[log[1]][newDepthInds].values

# remove NaNs across all logs
# TODO: Find a better way to locate nans across all logs and remove rows
nanInds=[]
for log in list(joinedLogs.keys()):
	nanLocs=np.where(np.isnan(joinedLogs[log])==True)[0]
	if nanLocs.size!=0:
		nanInds.append(nanLocs[0])
nanInds=np.unique(nanInds)
for log in list(joinedLogs.keys()):
	joinedLogs[log] = np.delete(joinedLogs[log], nanInds, None)

# write data to hdf5
units = np.concatenate((['m', 'ft'], gp_logs[:,2], res_logs[:,2]))
dsetNames = list(joinedLogs.keys())
safodIO.writeHDF5(safod_geophysData_outPath, joinedLogs, dsetNames, units=units)