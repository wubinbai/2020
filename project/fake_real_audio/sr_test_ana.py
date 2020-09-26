import os
import librosa
from tqdm import tqdm

dirs =os.listdir('../eda/test')
paths = []
for d in dirs:
    path = '../eda/test/' + d
    print(path)
    paths.append(path)
srs = [[] for k in range(len(paths))]
for i in tqdm(range(len(paths))):
    p = paths[i]
    fs = librosa.util.find_files(p)
    print(len(fs))
    for f in fs:
        y, sr = librosa.load(f,sr=None,mono=True)
        srs[i].append(sr)







