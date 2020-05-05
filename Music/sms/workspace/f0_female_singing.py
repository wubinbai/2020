import librosa
from f0_util import *

meows = librosa.util.find_files('/home/wb/5050/processed_50_50/Female_singing')   
samples_dir = meows

#all_files = librosa.util.find_files(samples_dir,recurse=True)
all_files = meows
#print(all_files)
for f in all_files[10+10:20+10]:
    print(f)
    visualize_f0(f)
