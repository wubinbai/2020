import matplotlib.pyplot as plt
import numpy as np
import time, os, sys
import math

#sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import utilFunctions as UF
import dftModel as DF

def visualize_three_segment(filename):
    (fs, x) = UF.wavread(filename)
    audio_length = x.shape[0]
    seg_length = audio_length//4
    pins = [seg_length*(i+1) for i in range(3)]
    w = np.hamming(1001*2)
    N = 2048*2
    #pin = 5000+115000
    plt.figure( figsize=(9.5, 7))
    for ind in range(len(pins)):
        pin = pins[ind]
        x1 = x[pin:pin+w.size]
        #x1 = x[pin:pin+w.size]
        mX, pX = DF.dftAnal(x1, w, N)

        plt.subplot(3,2,(ind*2+1))
        plt.plot(np.arange(pin, pin+w.size)/float(fs), x1, 'b*', lw=1.5)
        plt.axis([pin/float(fs), (pin+w.size)/float(fs), min(x1), max(x1)])
        plt.title(filename[-9:-4])

        plt.subplot(3,2,(ind*2+2))
        plt.plot(fs*np.arange(mX.size)/float(N), mX, 'r', lw=1.5)
        plt.axis([0,3900,-140,max(mX)])
        plt.grid()
        plt.title ('mX'+filename[-9:-4])

        #plt.subplot(3,1,3)
        #plt.plot(fs*np.arange(pX.size)/float(N), pX, 'c', lw=1.5)
        #plt.axis([0,3900,-18,14])
        #plt.title ('pX')

        plt.tight_layout()
        #plt.savefig('oboe-spectrum.png')
        plt.show()
