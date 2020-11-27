import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
import librosa
df = pd.read_csv('train.txt',sep='\s+')
cwd = os.getcwd()
root_path = Path(cwd)
fake_mono = 0
fake_dual = 0
fake_same = 0
fake_sr = []
fake_mono_sr = []
fake_dual_sr = []
rms_ratios = []
for fwav in tqdm(df.Audio_Name[:500]):
    f = root_path/fwav
    #print(f)
    y, sr = librosa.load(f,sr=None,mono=False)
    fake_sr.append(sr)
    #print(y.shape)
    if len(y.shape) == 1:
        fake_mono += 1
        fake_mono_sr.append(sr)
        rms = librosa.feature.rms(y)[0]
    elif len(y.shape) == 2:
        fake_dual += 1
        fake_dual_sr.append(sr)
        rms = librosa.feature.rms(y[0,:])[0]
        if all(y[0,:] == y [1,:]):
            fake_same += 1
    a = np.unique(rms).shape[0]
    b = rms.shape[0]
    rms_ratio = a/b
    rms_ratios.append(rms_ratio)

fake_mono_sr = np.array(fake_mono_sr)
fake_dual_sr = np.array(fake_dual_sr)
#########
real_mono = 0
real_dual = 0
real_sr = []
real_same = 0
real_mono_sr = []
real_dual_sr = []
for fwav in tqdm(df.Audio_Name[500:]):
    f = root_path/fwav
    #print(f)
    y, sr = librosa.load(f,sr=None,mono=False)
    real_sr.append(sr)
    #print(y.shape)
    if len(y.shape) == 1:
        real_mono += 1
        real_mono_sr.append(sr)
        rms = librosa.feature.rms(y)[0]
    elif len(y.shape) == 2:
        real_dual += 1
        real_dual_sr.append(sr)
        rms = librosa.feature.rms(y[0,:])[0]
        if all(y[0,:] == y [1,:]):
            real_same += 1
    a = np.unique(rms).shape[0]
    b = rms.shape[0]
    rms_ratio = a/b
    rms_ratios.append(rms_ratio)

real_mono_sr = np.array(real_mono_sr)
real_dual_sr = np.array(real_dual_sr)
