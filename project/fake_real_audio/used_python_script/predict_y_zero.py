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
yzeros = []
ys = []

### defs
def get_zero(y):
    count = 0
    for i in range(0,len(y)-1000,60):
        if all(y[i:i+1000] == 0):
            count += 1
    return count

###

for fwav in tqdm(df.Audio_Name[:500]):
    f = root_path/fwav
    #print(f)
    y, sr = librosa.load(f,sr=None,mono=True,duration=10*3)
    y, _ = librosa.effects.trim(y)
    ys.append(y)
    fake_sr.append(sr)
    #yzero = (y == 0).sum()/len(y)
    yzero = get_zero(y)
    yzeros.append(yzero)
    

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
    y, sr = librosa.load(f,sr=None,mono=True,duration=10*3)
    y, _ = librosa.effects.trim(y)
    ys.append(y)
    real_sr.append(sr)
    #yzero = (y==0).sum()/len(y)
    yzero = get_zero(y)
    yzeros.append(yzero)
real_mono_sr = np.array(real_mono_sr)
real_dual_sr = np.array(real_dual_sr)


cmp_d = {'yzeros':yzeros,'isfaked':df.Is_Faked}
cmp_df = pd.DataFrame(cmp_d)
temp_index_not_fake  = df[df['Is_Faked'] == 0].index
temp_index_fake = df[df['Is_Faked']==1].index[:]
for i in range(len(temp_index_not_fake)):
    index = temp_index_not_fake[i]
    print(df.Is_Faked[index],yzeros[index])
for i in range(len(temp_index_fake)):
    index = temp_index_fake[i]
    print(df.Is_Faked[index],yzeros[index])

