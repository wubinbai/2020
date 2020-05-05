import librosa
from f0_util import *

meows = librosa.util.find_files('/home/wb/baidunetdiskdownload/guitar')   
samples_dir = meows

#all_files = librosa.util.find_files(samples_dir,recurse=True)
all_files = meows
#print(all_files)
for f in all_files:
    if 'g1E_resample' in f:
        print(f)
        visualize_f0(f)
