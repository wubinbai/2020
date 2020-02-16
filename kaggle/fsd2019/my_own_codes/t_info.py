from preprocess import *
from tqdm import tqdm

import librosa
import os
PREFIX = '../'
dir_80 = PREFIX+'80_classes'
import numpy as np

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
            ts_i = []
            print('for running all samples, modify to fs[:]')
            for f in tqdm(fs[:10]):
                y, sr = librosa.load(f)
                t = librosa.get_duration(y,sr)
                ts_i.append(t)
            t_mean = np.mean(ts_i)
            t_std = np.std(ts_i)
            t_min = np.min(ts_i)
            t_max = np.max(ts_i)
            num_files = len(fs)
            # after analysis, make a statistics dir that stores info
            os.mkdir('statistics')
            stats = np.array([ts_i,t_mean,t_std,t_min,t_max,num_files])
            np.save('./statistics/stats.npy',stats)

        # After exploring one of curated/noisy path, go out.
        os.chdir('../')
    print('Everything is good for whatever experiment you have done on curated and noisy dirs!')
