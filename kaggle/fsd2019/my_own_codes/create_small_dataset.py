from preprocess import *
from tqdm import tqdm
import shutil


def cp_small_dataset():
    import os
    PREFIX = '../'
    dir_80 = PREFIX+'80_classes'
    print('First you need to manually create small_datasets folder in /home/wb/temp/small_datasets/curated, as well as /home/wb/temp/small_datasets/noisy')
    DESTINATION = '/home/wb/temp/small_datasets/'
    print('Start copying small datasets')
    n = 4
    print('each class: ', n, 'samples')
    os.chdir(dir_80)
    for i in ('train_curated','train_noisy'):
        os.chdir(i)
        now = i[6:] # now = 'curated' or 'noisy'
        # already have this line in preprocess: arr_classes, y0 = get_arr_classes(eval(now))
        temp =  eval('arr_classes_'+now)
        temp.sort()
        for index in range(len(temp)):#eval('arr_classes_'+now):
            #os.mkdir(sub)
            os.chdir(temp[index])
            samples = os.listdir()
            for i in range(n):
                source = samples[i]
                destination = DESTINATION
                shutil.copy(source, destination+now+'/'+str(index+100)+'_'+str(i)+'_'+temp[index]+'.wav')
            os.chdir('../')
        os.chdir('../')
    print('finish copying small_datasets!')
