import matplotlib.pyplot as plt
import numpy as np
import time, os, sys
import math

#sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import utilFunctions as UF
import dftModel as DF

(fs, x) = UF.wavread('/home/wb/Downloads/resample_A4.wav')
audio_length = x.shape[0]
seg_length = audio_length//4
pins = [seg_length*(i+1) for i in range(3)]
w = np.hamming(1001)
N = 2048*2
#pin = 5000+115000
for pin in pins:
    x1 = x[pin:pin+w.size]
    #x1 = x[pin:pin+w.size]
    mX, pX = DF.dftAnal(x1, w, N)

    plt.figure( figsize=(9.5, 7))
    plt.subplot(311)
    plt.plot(np.arange(pin, pin+w.size)/float(fs), x1, 'b', lw=1.5)
    plt.axis([pin/float(fs), (pin+w.size)/float(fs), min(x1), max(x1)])
    plt.title('x (oboe-A4.wav)')

    plt.subplot(3,1,2)
    plt.plot(fs*np.arange(mX.size)/float(N), mX, 'r', lw=1.5)
    plt.axis([0,3900,-140,max(mX)])
    plt.grid()
    plt.title ('mX')

    plt.subplot(3,1,3)
    plt.plot(fs*np.arange(pX.size)/float(N), pX, 'c', lw=1.5)
    plt.axis([0,3900,-18,14])
    plt.title ('pX')

    plt.tight_layout()
    plt.savefig('oboe-spectrum.png')
    plt.show()
