import librosa
from spec_util import *

samples_dir = '/home/wb/2020/Music/my_fsd_file'
all_files = librosa.util.find_files(samples_dir,recurse=True)
#print(all_files)
for f in all_files:
    visualize_three_segment(f)
