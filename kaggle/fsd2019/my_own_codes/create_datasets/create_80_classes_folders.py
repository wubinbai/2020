import pandas as pd
#TOT = 'input'
TOT = 'data'

#CURATED = '../' + TOT + '/audio_train/'
#NOISY = CURATED
#TEST = '../' + TOT + '/test/'

CURATED = '../' + TOT + '/train_curated/'
NOISY = '../' + TOT + '/train_noisy/'
TEST = '../' + TOT + '/test/'



curated = pd.read_csv('../' + TOT + '/train_curated.csv')
noisy = pd.read_csv('../' + TOT + '/train_noisy.csv')
test = pd.read_csv('../' + TOT + '/sample_submission.csv')

curated_paths = CURATED + curated['fname']
noisy_paths = NOISY + noisy['fname']
test_paths = TEST + test['fname']


def get_arr_classes(df):
    res = []
    for row in df.labels.values:
        res.append(row.split(',')[0])
    res_df = pd.DataFrame(res)
    s = res_df = res_df[0] # Series.unique()
    res_df = res_df.unique()
    return res_df, s


arr_classes_curated, s_curated = get_arr_classes(curated)
arr_classes_noisy, s_noisy = get_arr_classes(noisy)
#arr_classes_test = get_arr_classes(test)

# modified
curated_m = curated.copy()
noisy_m = noisy.copy()

curated_m['y0'] = s_curated
noisy_m['y0'] = s_noisy

#from preprocess import *
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
