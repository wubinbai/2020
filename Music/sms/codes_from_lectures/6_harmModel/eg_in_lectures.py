import sys
import os

#sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))




from scipy.signal import get_window
#from software.models import utilFunctions as UF
#from software.models import dftModel as DFT

import utilFunctions as UF
import dftModel as DFT

def TWM_Errors(pfreq, pmag, f0c):
	"""
	Two-way mismatch algorithm for f0 detection (by Beauchamp&Maher)
	[better to use the C version of this function: UF_C.twm]
	pfreq, pmag: peak frequencies in Hz and magnitudes,
	f0c: frequencies of f0 candidates
	returns f0, f0Error: fundamental frequency detected and its error
	"""

	p = 0.5                                          # weighting by frequency value
	q = 1.4                                          # weighting related to magnitude of peaks
	r = 0.5                                          # scaling related to magnitude of peaks
	rho = 0.33                                       # weighting of MP error
	Amax = max(pmag)                                 # maximum peak magnitude
	maxnpeaks = 10                                   # maximum number of peaks used
	harmonic = np.matrix(f0c)
	ErrorPM = np.zeros(harmonic.size)                # initialize PM errors
	MaxNPM = min(maxnpeaks, pfreq.size)
	for i in range(0, MaxNPM) :                      # predicted to measured mismatch error
		difmatrixPM = harmonic.T * np.ones(pfreq.size)
		difmatrixPM = abs(difmatrixPM - np.ones((harmonic.size, 1))*pfreq)
		FreqDistance = np.amin(difmatrixPM, axis=1)    # minimum along rows
		peakloc = np.argmin(difmatrixPM, axis=1)
		Ponddif = np.array(FreqDistance) * (np.array(harmonic.T)**(-p))
		PeakMag = pmag[peakloc]
		MagFactor = 10**((PeakMag-Amax)/20)
		ErrorPM = ErrorPM + (Ponddif + MagFactor*(q*Ponddif-r)).T
		harmonic = harmonic+f0c

	ErrorMP = np.zeros(harmonic.size)                # initialize MP errors
	MaxNMP = min(maxnpeaks, pfreq.size)
	for i in range(0, f0c.size) :                    # measured to predicted mismatch error
		nharm = np.round(pfreq[:MaxNMP]/f0c[i])
		nharm = (nharm>=1)*nharm + (nharm<1)
		FreqDistance = abs(pfreq[:MaxNMP] - nharm*f0c[i])
		Ponddif = FreqDistance * (pfreq[:MaxNMP]**(-p))
		PeakMag = pmag[:MaxNMP]
		MagFactor = 10**((PeakMag-Amax)/20)
		ErrorMP[i] = sum(MagFactor * (Ponddif + MagFactor*(q*Ponddif-r)))

	Error = (ErrorPM[0]/MaxNPM) + (rho*ErrorMP/MaxNMP)  # total error
	#f0index = np.argmin(Error)                       # get the smallest error
	#f0 = f0c[f0index]                                # f0 with the smallest error
	return Error
        

(fs, x) = UF.wavread('/home/wb/Downloads/A4_resample.wav')

N = 1024
M = 601
t = -60
minf0 = 50
maxf0 = 2000
hN = N//2
hM = (M+1)//2
w = get_window('hamming',M)
start = int(.8*fs)
x1 = x[start:start+M]
mX, pX = DFT.dftAnal(x1, w, N)
mX = mX[:-1]
pX = pX[:-1]
ploc = UF.peakDetection(mX, t)
iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc) 
ipfreq = fs * iploc / N
f0c = np.argwhere((ipfreq>minf0) & (ipfreq<maxf0))[:,0]
f0cf = ipfreq[f0c]
f0Errors = TWM_Errors(ipfreq, ipmag, f0cf)
freqaxis = fs*np.arange(N/2)/float(N)
plt.plot(freqaxis, mX)
plt.plot(ipfreq, ipmag, marker='x', linestyle='') 



