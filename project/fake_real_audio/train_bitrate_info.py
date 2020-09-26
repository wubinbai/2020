import librosa
from tqdm import tqdm
import os
import shutil
from tqdm import tqdm
import json
df = pdrc('../train/train.txt',sep='\s+')

with open('../txt/soxi_train.txt') as f:
    data = f.readlines()

names = []
bitrates = []
for i in range(0,len(data)-1,10):
    name = data[1+i][:][-14:][:-2]
    names.append(name)
    bitrate = data[7+i][-6:][:-1]
    bitrates.append(bitrate)
dict_bitrates = dict()
for i in range(len(names)):
    dict_bitrates[names[i]] = bitrates[i]
with open('../dict_bitrates.json','w') as f:
    json.dump(dict_bitrates,f)

arr_bitrates = np.array(bitrates)
seq_bitrates = []
for audio_name in df.Audio_Name:
    seq_bitrates.append(dict_bitrates[audio_name])
df['bitrates'] = seq_bitrates
df.to_csv('../new_df.csv',index=False)
########
##### create folders and move wav into corresponding bitrate folder

os.makedirs('../eda/',exist_ok=True)
os.makedirs('../eda/train',exist_ok=True)
os.makedirs('../eda/train/128k',exist_ok=True)
os.makedirs('../eda/train/256k',exist_ok=True)
#os.makedirs('../eda/train/258k',exist_ok=True)
#os.makedir`s('../eda/train/259k',exist_ok=True)
os.makedirs('../eda/train/384k',exist_ok=True)
os.makedirs('../eda/train/512k',exist_ok=True)
os.makedirs('../eda/train/768k',exist_ok=True)
os.makedirs('../eda/train/1_06M',exist_ok=True)


for a,b in tqdm(dict_bitrates.items()):
    #print(a,b)
    src = '../train/' + a
    if not b.endswith('M'):
        dst = '../eda/train/' + b[1:]
    else:
        dst = '../eda/train/' + '1_06M'
    #print(src,dst)
    shutil.copy(src,dst)

path_256k = '../eda/train/256k'
fs_256k = librosa.util.find_files(path_256k)
for f in tqdm(fs_256k):
    f_tail = f.split('/')[-1]
    isfake = df.loc[df.loc[df['Audio_Name'] == f_tail].index[0],'Is_Faked']
    if isfake:
        move_to_256k_fake(f)
def move_to_256k_fake(f):
    os.makedirs('../eda/train/256k/fake',exist_ok=True)
    src = f
    dst = '../eda/train/256k/fake'
    shutil.move(src,dst)


