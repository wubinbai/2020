import librosa
import os
from tqdm import tqdm
import shutil

path = './test'

fs = os.listdir(path)
all_durations = []
for f in tqdm(fs):
    fp = './test/' + f
    y,sr = librosa.load(fp,res_type='kaiser_fast')
    duration = librosa.get_duration(y)
    DES = './10s_test'
    if duration <= 10:
        ORI = fp
        shutil.move(ORI,DES)

import librosa
import os
from tqdm import tqdm
import shutil

path = './10s_test/'
fs = os.listdir(path)
all_durations = []
for f in tqdm(fs):
    fp = path + f
    y,sr = librosa.load(fp,res_type='kaiser_fast')
    duration = librosa.get_duration(y)
    DES = './5s_test'
    if duration <= 5:
        ORI = fp
        shutil.move(ORI,DES)

path = './5s_test/'
fs = os.listdir(path)
all_durations = []
for f in tqdm(fs):
    fp = path + f
    y,sr = librosa.load(fp,res_type='kaiser_fast')
    duration = librosa.get_duration(y)
    DES = './2s_test'
    if duration <= 2:
        ORI = fp
        shutil.move(ORI,DES)


