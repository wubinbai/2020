import librosa
import librosa.display as disp

import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def save_mel(path,filenames,destination):
    assert len(filenames) == 4
    print('note: len(samples) must be 4, otherwise, modify this function...')
    plt.cla()
    plt.clf()
    temp = filenames[0][:3]+'_'+filenames[0][6:-4]
    stringfigname = destination + temp + '.png'
    filenames.sort()
    for index in range(len(filenames)):
            fname = filenames[index]
            y, sr = librosa.load(path+fname)
            melspec = librosa.feature.melspectrogram(y,sr)
            S_dB = librosa.power_to_db(melspec,ref=np.max)
            plt.subplot(2,2,index+1)
            disp.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)#fmax=8000)
            plt.colorbar(format='%+2.0f dB')
            plt.title(stringfigname[19:19+8])
            plt.tight_layout()
            
    plt.savefig(stringfigname)


fs = os.listdir('../curated')
l = [s[:3] for s in fs]
short = [int(v) for v in l]
big = max(short)
small = min(short)
for index in tqdm(range(small,big+1)):
    process = []
    for f in fs:
        if int(f[:3]) == index:
            process.append(f)
    print('processing: ', process)
    destination = '../ana_curated/'
    path = '../curated/'
    save_mel(path,process,destination)
    print('ok...!')

