# this file should be within the workspace dir within sms-tools dir

import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))

from scipy.signal import get_window
import utilFunctions as UF
import dftModel as DFT
import stft as STFT
import harmonicModel as HM
import sineModel as SM

#(fs, x) = UF.wavread('/home/wb/Downloads/A4_resample.wav')

N = 2048
t = -60
minf0 = 50
maxf0 = 800
w = get_window('blackman',1001)
f0et = 1
H = 1000


# use this line to retrieve f0: pass in x as audio array, fs as sampling rate == 44100

#f0 = HM.f0Detection(x, fs, w, N, H, t, minf0, maxf0, f0et)
#plt.figure()
#plt.plot(f0)
#start = int(.8*fs)
#x1 = x[start:start+M]
#mX, pX = DFT.dftAnal(x1, w, N)
#mX = mX[:-1]
#pX = pX[:-1]
#ploc = UF.peakDetection(mX, t)
#iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc) 
#ipfreq = fs * iploc / N
#f0c = np.argwhere((ipfreq>minf0) & (ipfreq<maxf0))[:,0]
#f0cf = ipfreq[f0c]
#f0Errors = TWM_Errors(ipfreq, ipmag, f0cf)
#freqaxis = fs*np.arange(N/2)/float(N)
#plt.plot(freqaxis, mX)
#plt.plot(ipfreq, ipmag, marker='x', linestyle='') 



