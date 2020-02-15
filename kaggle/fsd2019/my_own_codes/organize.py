from preprocess import *
from tqdm import tqdm
import shutil


def org():
    import os
    PREFIX = '../'
    dir_80 = PREFIX+'80_classes'
    print('Start making 80_classes directories, if FileExists Error, rm the whole 80_classes directory')
    os.mkdir(dir_80)
    os.chdir(dir_80)
    os.mkdir('train_curated')
    os.mkdir('train_noisy')
    #
    for i in ('train_curated','train_noisy'):
        os.chdir(i)
        now = i[6:] # now = 'curated' or 'noisy'
        # already have this line in preprocess: arr_classes, y0 = get_arr_classes(eval(now))
        for sub in eval('arr_classes_'+now):
            os.mkdir(sub)
        os.chdir('../')
    print('finish making 80_classes directories!')
    print('Start moving .wav files into 80_classes... ')
    for i in ('train_curated','train_noisy'):
        os.chdir(i)
        now = i[6:] # now = 'curated' or 'noisy'
        f = now+'_m'
        for i in tqdm(eval(f).values):
            ori = i[0]
            des = i[2]
            ORI = '../../' + TOT + '/train_' + now + '/' + ori
            DES = './' + des
            shutil.copy(ORI,DES)
        os.chdir('../')
