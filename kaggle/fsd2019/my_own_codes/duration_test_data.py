from preprocess import *
import librosa
from tqdm import tqdm


def get_duration(test):
    ts = []
    for fname in tqdm(test.fname[:50]):
        fpath = '../data/test/' + fname
        y, sr = librosa.load(fpath)
        duration = librosa.get_duration(y,sr)
        ts.append(duration)
    print('the duration for each sample is calculated and is returned as ts')
    return ts

ts = get_duration(test)

