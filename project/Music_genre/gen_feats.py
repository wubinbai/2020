import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv
import keras
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
'''
#  * Blues
#  * Classical
#  * Country
#  * Disco
#  * Hiphop
#  * Jazz
#  * Metal
#  * Pop
#  * Reggae
#  * Rock
cmap = plt.get_cmap('inferno')

plt.figure(figsize=(10,10))
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)     
    for filename in os.listdir(f'./MIR/genres/{g}'):
        songname = f'./MIR/genres/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=5)
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')
        plt.clf()
'''


header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate std_chroma_stft std_rmse std_spec_cent std_spec_bw std_rolloff std_zcr max_rmse min_rmse max_spec_cent min_spec_cent max_spec_bw min_spec_bw max_rolloff min_rolloff max_zcr min_zcr'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

print('Start computing feats and writing to csv!!!')
file = open('data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in tqdm(genres):
    for filename in os.listdir(f'./MIR/genres/{g}'):
        songname = f'./MIR/genres/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        #bug
        rmse = librosa.feature.rms(y)
        vsc1 = np.percentile(spec_cent,30)
        vsc2 = np.percentile(spec_cent,50)
        vsc3 = np.percentile(spec_cent,70)
        bw1 = np.percentile(spec_bw,30)
        bw2 = np.percentile(spec_bw,50)
        bw3 = np.percentile(spec_bw,70)
        ro1 = np.percentile(rolloff,30)
        ro2 = np.percentile(rolloff,50)
        ro3 = np.percentile(rolloff,70)
        zcr1 = np.percentile(zcr,30)
        zcr2 = np.percentile(zcr,50)
        zcr3 = np.percentile(zcr,70)
        rms1 = np.percentile(rmse,30)
        rms2 = np.percentile(rmse,50)
        rms3 = np.percentile(rmse,70)

        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} {np.std(chroma_stft)} {np.std(rmse)} {np.std(spec_cent)} {np.std(spec_bw)} {np.std(rolloff)} {np.std(zcr)} {np.max(rmse)} {np.min(rmse)} {np.max(spec_cent)} {np.min(spec_cent)} {np.max(spec_bw)} {np.min(spec_bw)} {np.max(rolloff)} {np.min(rolloff)} {np.max(zcr)} {np.min(zcr)} {vsc1} {vsc2} {vsc3} {bw1} {bw2} {bw3} {ro1} {ro2} {ro3} {zcr1} {zcr2} {zcr3} {rms1} {rms2} {rms3}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())



print('FINISH WRITING TO CSV!!!')


