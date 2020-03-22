import argparse
import matplotlib.pyplot as plt
from logIO.Borehole import BoreholeIO
from processing.correlateLogs import correlateLogs
from processing.clusterLogs import clusterLogs
import numpy as np

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--inFile', type=str, default="../data/safod_data_trimmed.h5")
	parser.add_argument('-s', '--sheetname', type=str, nargs='+', default="Mag. Susc.")
	parser.add_argument('-sr', '--skiprows', nargs='+', default=0)
	parser.add_argument('-o', '--outFile', required=False, type=str)
	args = parser.parse_args()

	safodIO = BoreholeIO(args.inFile)
	data = {}
	for (s, sheet) in enumerate(args.sheetname):
		data[sheet] = safodIO.readExcel(sheetname=sheet, skiprows=[args.skiprows[s]])

	mag_susc = data["Mag. Susc."]
	xrf_majors = data["XRF Majors Save"]
	print(list(xrf_majors.keys()))

	ms = mag_susc["Mass Susceptibility (K. Verosub, all), 10-8 m3/kg"]
	SiO2 = xrf_majors["SiO2"]
	TiO2 = xrf_majors["TiO2"]
	Al2O3 = xrf_majors["Al2O3"]
	FeO = xrf_majors["FeO*"]
	MnO = xrf_majors["MnO"]
	MgO = xrf_majors["MgO"]

	clusterer = clusterLogs(xrf_majors, logs=list(xrf_majors.keys())[1:15])
	stackedLogs = clusterer.stackLogs()
	corr_matrix = correlateLogs(stackedLogs, list(xrf_majors.keys())[1:15], plot=True)

	plt.plot(mag_susc["Sample Depth, ft"]/3.2808, ms/np.max(ms))
	plt.plot(xrf_majors["Core Depth (m)"], SiO2/np.max(SiO2))
	plt.show()
	quit()

if __name__ == "__main__":
	main()