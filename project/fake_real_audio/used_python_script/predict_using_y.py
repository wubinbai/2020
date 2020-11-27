import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
import librosa
df = pd.read_csv('train.txt',sep='\s+')
cwd = os.getcwd()
root_path = Path(cwd)
fake_mono = 0
fake_dual = 0
fake_same = 0
fake_sr = []
fake_mono_sr = []
fake_dual_sr = []
ratios = []
for fwav in tqdm(df.Audio_Name[:500]):
    f = root_path/fwav
    #print(f)
    y, sr = librosa.load(f,sr=None,mono=True)
    fake_sr.append(sr)
    
    a = np.unique(y).shape[0]
    b = y.shape[0]
    ratio = a/b
    ratios.append(ratio)

fake_mono_sr = np.array(fake_mono_sr)
fake_dual_sr = np.array(fake_dual_sr)
#########
real_mono = 0
real_dual = 0
real_sr = []
real_same = 0
real_mono_sr = []
real_dual_sr = []
for fwav in tqdm(df.Audio_Name[500:]):
    f = root_path/fwav
    #print(f)
    y, sr = librosa.load(f,sr=None,mono=True)
    real_sr.append(sr)
    a = np.unique(y).shape[0]
    b = y.shape[0]
    ratio = a/b
    ratios.append(ratio)

real_mono_sr = np.array(real_mono_sr)
real_dual_sr = np.array(real_dual_sr)


In [4]: cmp_d = {'ratios':ratios,'isfaked':df.I
   ...: s_Faked}

In [5]: cmp_df = pd.DataFrame(cmp_d)

