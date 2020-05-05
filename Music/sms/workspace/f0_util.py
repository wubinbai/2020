import matplotlib.pyplot as plt
import numpy as np
import time, os, sys
import math

from scipy.signal import hamming, triang, blackman
import sys, os, functools, time

#sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import utilFunctions as UF
import dftModel as DF
import stft as STFT
import sineModel as SM
import harmonicModel as HM


def visualize_f0(filename):
    (fs, x) = UF.wavread(filename)
    w = np.blackman(1501)
    N = 2048
    t = -90
    minf0 = 20
    maxf0 = 2000
    f0et = 1
    maxnpeaksTwm = 4
    H = 128
    # select 
    x1 = x[int(0.5*fs):int(0.8*fs)]
    mX, pX = STFT.stftAnal(x, w, N, H)
    f0 = HM.f0Detection(x, fs, w, N, H, t, minf0, maxf0, f0et)
    f0 = UF.cleaningTrack(f0, 5)
    yf0 = UF.sinewaveSynth(f0, .8, H, fs)
    f0[f0==0] = np.nan
    maxplotfreq = 800.0
    numFrames = int(mX[:,0].size)
    frmTime = H*np.arange(numFrames)/float(fs)                             
    binFreq = fs*np.arange(N*maxplotfreq/fs)/N                        
    plt.figure()
    plt.plot(f0)
    plt.title(filename[-9:-4])
    plt.show()
def util_get_f0_from_audio(y,fs=22050):
    #(fs, x) = UF.wavread(filename)
    x = y
    w = np.blackman(1501)
    N = 2048
    t = -90
    minf0 = 20
    maxf0 = 2000
    f0et = 1
    maxnpeaksTwm = 4
    H = 128
    # select 
    x1 = x[int(0.5*fs):int(0.8*fs)]
    mX, pX = STFT.stftAnal(x, w, N, H)
    f0 = HM.f0Detection(x, fs, w, N, H, t, minf0, maxf0, f0et)
    f0 = UF.cleaningTrack(f0, 5)
    yf0 = UF.sinewaveSynth(f0, .8, H, fs)
    f0[f0==0] = np.nan
    maxplotfreq = 800.0
    numFrames = int(mX[:,0].size)
    frmTime = H*np.arange(numFrames)/float(fs)                             
    binFreq = fs*np.arange(N*maxplotfreq/fs)/N                        
    #plt.figure()
    #plt.plot(f0)
    #plt.title(filename[-9:-4])
    #plt.show()
    return f0

