from preprocess import *

def org():
    import os
    PREFIX = '../'
    dir_80 = PREFIX+'80_classes'
    os.mkdir(dir_80)
    os.chdir(dir_80)
    os.mkdir('train_curated','train_noisy')
    #
    for i in ('train_curated','train_noisy'):
        os.chdir(i)
        now = i[6:] # now = 'curated' or 'noisy'
        # already have this line in preprocess: arr_classes, y0 = get_arr_classes(eval(now))
        for sub in eval('arr_classes_'+now):
            os.mkdir(sub)
        os.chdir('../')
