from disp_melspectro.py import *
import os

PATH_IN = '../80_classes'

for one in ('train_curated','train_noisy'):
    os.chdir(one)
    class_dir = os.listdir()
    for sub in class_dir:
        os.chdir(sub)
        files = os.listdir()
        # only 1 file
        for f in [files[0]]:

            -> cp files to dest0
            -> save fig to dest1

        go back
    go back

