import librosa
import os
from tqdm import tqdm
import shutil
os.mkdir('10s')
os.mkdir('5s')
os.mkdir('2s')

path = './curated/'

fs = os.listdir(path)
all_durations = []
for f in tqdm(fs):
    fp = path + f
    y,sr = librosa.load(fp,res_type='kaiser_fast')
    duration = librosa.get_duration(y)
    DES = './10s'
    if duration <= 10:
        ORI = fp
        shutil.move(ORI,DES)

import librosa
import os
from tqdm import tqdm
import shutil

path = './10s/'
fs = os.listdir(path)
all_durations = []
for f in tqdm(fs):
    fp = path + f
    y,sr = librosa.load(fp,res_type='kaiser_fast')
    duration = librosa.get_duration(y)
    DES = './5s'
    if duration <= 5:
        ORI = fp
        shutil.move(ORI,DES)

path = './5s/'
fs = os.listdir(path)
all_durations = []
for f in tqdm(fs):
    fp = path + f
    y,sr = librosa.load(fp,res_type='kaiser_fast')
    duration = librosa.get_duration(y)
    DES = './2s'
    if duration <= 2:
        ORI = fp
        shutil.move(ORI,DES)


