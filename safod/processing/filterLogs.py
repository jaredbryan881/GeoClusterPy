import pycwt
import pywt
import numpy as np

def icwt(W, sj, dt, dj, wav):
	return dj*np.sqrt(dt)/0.771*(np.pi**0.25) * np.sum(np.real(W.transpose())/(sj**(0.5)),axis=1)

def cwtFilter(y, threshold, dt, dj=1/12):
	# define parameters for cwt
	wavelet=pycwt.wavelet._check_parameter_wavelet('morlet')
	s0=2*dt/wavelet.flambda()
	yshape = y.shape[0]
	J=np.int(np.round(np.log2(yshape*dt/s0)/dj))

	# take cwt
	W, sj, freq, coi, _, _ = pycwt.cwt(y, dt, dj, s0, J, 'morlet')

	# filter low amplitude values from cwt
	W[np.where(np.abs(W)<threshold*np.max(np.abs(W)))] = 0.0

	# inverse cwt
	datarec = icwt(W, sj, dt, dj, 'morlet')

	return datarec

def dwtFilter(y, threshold, levels, wav):
	# define wavelet
	wavelet=pywt.Wavelet(wav)

	# decompose with dwt
	coeffs = pywt.wavedec(y, wav, level=levels)

	# filter low amplitudes from dwt
	for i in range(1, len(coeffs)):
		coeffs[i] = pywt.threshold(coeffs[i], threshold*np.max(coeffs[i]))

	# inverse dwt
	datarec = pywt.waverec(coeffs, wavelet)

	return datarec
