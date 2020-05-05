import librosa
from spec_util import *

bass_guitars = librosa.util.find_files('/home/wb/5050/processed_50_50/Electric_guitar')   
samples_dir = bass_guitars

#all_files = librosa.util.find_files(samples_dir,recurse=True)
all_files = bass_guitars
#print(all_files)
for f in all_files[10:20]:
    visualize_three_segment(f)
