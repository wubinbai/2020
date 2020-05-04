import numpy as np
import librosa
import matplotlib.pyplot as plt


def plot_pitch(y,N=2048,H=512,sr=44100,maxplotfreq=800):
    '''
    N: fft size
    H: hop length
    maxplotfreq: max Hz plot
    '''
    pitches,mags = librosa.piptrack(y)
    numFrames = int(mags[0,:].size)
    frmTime = H*np.arange(numFrames)/float(sr)
    binFreq = sr*np.arange(N*maxplotfreq/sr)/N
    plt.figure()
    plt.pcolormesh(frmTime, binFreq, pitches[:int(N*maxplotfreq/sr+1),:])
    plt.figure()
    plt.pcolormesh(frmTime, binFreq, mags[:int(N*maxplotfreq/sr+1),:])

    return pitches, mags
