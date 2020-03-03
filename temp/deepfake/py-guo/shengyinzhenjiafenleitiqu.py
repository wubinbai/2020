#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
def show_path(dir_path):
    with open(dir_path+'/metadata.json','r') as fp:
        data = json.load(fp)
    return data


# In[2]:


# 获取所有真视频的名称
def get_real_mp4(part,data):
    #part=0
    #data = show_path(f'../data/videos/dfdc_train_part_{part}')
    total = len(data.keys())
    lst1 = []
    for k in list(data.keys()):
        #print(k,data[k])
        if len(data[k])==2:
            lst1.append(k)
        #print(len(data[k]))
    print(f'part:{part},total:{total},real:{len(lst1)}')
    return lst1


# In[ ]:


lst2 = []
for k in list(data.keys()):
    if len(data[k])==3:
        if data[k]['original']=='aayrffkzxn.mp4':
            lst2.append(k)
    #print(len(data[k]))
print(len(lst2))


# In[5]:


part = 0
data = show_path(f'../data/videos/dfdc_train_part_{part}')
real_list = get_real_mp4(part,data)


# In[6]:


import librosa
import librosa.display
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import IPython
import IPython.display
import PIL
def read_audio(conf, pathname, trim_long_data):
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    # trim silence
    if 0 < len(y): # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y) # trim, top_db=default(60)
    # make it unified length to conf.samples
    if len(y) > conf.samples: # long enough
        if trim_long_data:
            print('long enough')
            y = y[0:0+conf.samples]
    else: # pad blank
        padding = conf.samples - len(y)    # add padding at both ends
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')
    return y

def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(audio, 
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=conf.n_fft,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

def show_melspectrogram(conf, mels, title='Log-frequency power spectrogram'):
    librosa.display.specshow(mels, x_axis='time', y_axis='mel', 
                             sr=conf.sampling_rate, hop_length=conf.hop_length,
                            fmin=conf.fmin, fmax=conf.fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()
    #plt.savefig("fake.png")

def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):
    x = read_audio(conf, pathname, trim_long_data)
    mels = audio_to_melspectrogram(conf, x)
    if debug_display:
        IPython.display.display(IPython.display.Audio(x, rate=conf.sampling_rate))
        show_melspectrogram(conf, mels)
    return mels


class conf:
    # Preprocessing settings
    sampling_rate = 44100
    duration = 2
    hop_length = 347*duration # to make time steps 128
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    samples = sampling_rate * duration


# In[19]:


list3 = []
for real in real_list:
    lst2 = []
    lst2.append(real)
    set1 = set()
    std = np.sum(np.std(read_as_melspectrogram(conf,f'/home/dl/deepfake/data/videos/dfdc_train_part_{part}/{real}', trim_long_data=False, debug_display=False)))
    set1.add(std)
    for k in list(data.keys()):
        if len(data[k])==3:
            if data[k]['original']==real:
                std = np.sum(np.std(read_as_melspectrogram(conf,f'/home/dl/deepfake/data/videos/dfdc_train_part_{part}/{k}', trim_long_data=False, debug_display=False)))
                if std not in set1:
                    lst2.append(k)
                    set1.add(std)
        #print(len(data[k]))
    list3.append(lst2)


# In[20]:


print(list3)


# In[22]:


lst4 = []
for i in list3:
    if len(i)>1:
        lst4.append(i)


# In[23]:


print(lst4)


# In[ ]:




