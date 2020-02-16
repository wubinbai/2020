from preprocess import *
from tqdm import tqdm

import librosa
import os
PREFIX = '../'
dir_80 = PREFIX+'80_classes'

def get_t_info():
    os.chdir(dir_80)
    for i in ('train_curated','train_noisy'):
        os.chdir(i)
        now = i[6:] # now = 'curated' or 'noisy'
        for sub in eval('arr_classes_'+now):
            # go into every class dir
            os.chdir(sub)
            # do whatever you want
            fs = os.listdir()
            ts_one_class = []
            for f in tqdm(fs):
                y, sr = librosa.load(i)
                t = librosa.get_duration(y,sr)
                ts_one_class.append(t)
            assert len(ts_one_class) == len(fs)

        # After exploring one of curated/noisy path, go out.
        os.chdir('../')
    print('Everything is good for whatever experiment you have done on curated and noisy dirs!')
