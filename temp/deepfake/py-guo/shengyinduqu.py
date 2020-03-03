#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa


# In[2]:


path = '/home/dl/deepfake/data/dfdc_train_part_0/hthuorhwdn.mp4'
data, _ = librosa.load(path, 44100)


# In[3]:


print(data)


# In[17]:


from tqdm import tqdm
import numpy as np


# In[8]:


def worker_cgf(file_path):
    result = []
    for path in tqdm(file_path):
        print(path)
        data, _ = librosa.load(path, 44100)

        result.append(get_global_feat(data, num_steps=128))
    print(result)
    return result


# In[21]:


def get_global_feat(x,num_steps):
    stride = len(x)/num_steps
    ts = []
    # 分为128 段 ，每次1.5 段  去每次的 10 分
    for s in range(num_steps):
        i = s * stride
        wl = max(0,int(i - stride/2))
        wr = int(i + 1.5*stride)
        local_x = x[wl:wr]
        percent_feat = np.percentile(local_x, [0, 1, 25, 30, 50, 60, 75, 99, 100]).tolist()
        range_feat = local_x.max()-local_x.min()
        ts.append([np.mean(local_x),np.std(local_x),range_feat]+percent_feat)
        #print(wl,wr)
    ts = np.array(ts)
    print(ts.shape)
    assert ts.shape == (128,12),(len(x),ts.shape)
    return ts


# In[22]:


def create_global_feat():
    file_path=['/home/dl/deepfake/data/dfdc_train_part_0/hthuorhwdn.mp4','/home/dl/deepfake/data/dfdc_train_part_0/tjdxwdoumt.mp4']
    results = []
    res = worker_cgf(file_path)             
    results.append(res)

    results = np.concatenate([res for res in results],axis=0)
    print(results)
    np.save('gfeat', np.array(results))


# In[23]:


create_global_feat()


# In[2]:


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

# example
x = read_as_melspectrogram(conf,'/home/dl/deepfake/data/videos/dfdc_train_part_0/htorvhbcae.mp4', trim_long_data=False, debug_display=True)


# In[3]:


fake = read_as_melspectrogram(conf,'/home/dl/deepfake/data/videos/dfdc_train_part_0/dhjnjkzuhq.mp4', trim_long_data=False, debug_display=True)


# In[6]:


real = read_as_melspectrogram(conf,'/home/dl/deepfake/data/videos/dfdc_train_part_0/dzlpfpszyv.mp4', trim_long_data=False, debug_display=True)


# In[7]:


real2 = read_as_melspectrogram(conf,'/home/dl/deepfake/data/videos/dfdc_train_part_0/ueobjeflda.mp4', trim_long_data=False, debug_display=True)


# In[19]:


print(fake.shape)


# In[8]:


np.sum(np.std(fake,axis=0))


# In[9]:


np.sum(np.std(real,axis=0))


# In[10]:


np.sum(np.std(real2,axis=0))


# In[31]:


np.sum(np.std(read_as_melspectrogram(conf,'/home/dl/deepfake/data/videos/dfdc_train_part_0/fckxaqjbxk.mp4',trim_long_data=False),axis=0))


# In[53]:


np.sum(np.std(read_as_melspectrogram(conf,'/home/dl/deepfake/data/videos/dfdc_train_part_1/fyiymoftcx.mp4',trim_long_data=False),axis=0))


# In[64]:



np.sum(np.std(read_as_melspectrogram(conf,'/home/dl/deepfake/data/videos/dfdc_train_part_0/wclvkepakb.mp4',trim_long_data=False),axis=0))


# In[65]:


# dfdc_train_part_1/gshruegrcb.mp4 # dfdc_train_part_1/yietrwuncf.mp4
# dfdc_train_part_1/ujslmfluwe.mp4   dfdc_train_part_1/oznyyhvzxk.mp4
import json
def show_path(dir_path):
    with open(dir_path+'/metadata.json','r') as fp:
        data = json.load(fp)
    return data


# In[66]:


print(len("{'label': 'REAL', 'split': 'train'}"))


# In[73]:


part=0
data = show_path(f'../data/videos/dfdc_train_part_{part}')
print(len(data.keys()))
lst1 = []
for k in list(data.keys()):
    #print(k,data[k])
    if len(data[k])==2:
        lst1.append(k)
    #print(len(data[k]))
print(len(lst1))


# In[107]:


lst2 = []
for k in list(data.keys()):
    if len(data[k])==3:
        if data[k]['original']=='aayrffkzxn.mp4':
            lst2.append(k)
    #print(len(data[k]))
print(len(lst2))


# In[108]:


np.sum(np.std(read_as_melspectrogram(conf,'/home/dl/deepfake/data/videos/dfdc_train_part_0/aayrffkzxn.mp4',trim_long_data=False),axis=0))


# In[109]:


set1 = set()
for i in lst2:
    std = np.sum(np.std(read_as_melspectrogram(conf,f'/home/dl/deepfake/data/videos/dfdc_train_part_0/{i}',trim_long_data=False),axis=0))
    print(i,std)
    set1.add(std)
print(set1)


# In[102]:


real = read_as_melspectrogram(conf,'/home/dl/deepfake/data/videos/dfdc_train_part_0/cythdgxpdi.mp4', trim_long_data=False, debug_display=True)


# In[104]:


real = read_as_melspectrogram(conf,'/home/dl/deepfake/data/videos/dfdc_train_part_0/hpwdlgshqg.mp4', trim_long_data=False, debug_display=True)
# j假


# In[106]:


from PIL import Image
from PIL import ImageChops 

def compare_images(path_one, path_two, diff_save_location):
    """
    比较图片，如果有不同则生成展示不同的图片
 
    @参数一: path_one: 第一张图片的路径
    @参数二: path_two: 第二张图片的路径
    @参数三: diff_save_location: 不同图的保存路径
    """
    image_one = Image.open(path_one)
    image_two = Image.open(path_two)
    try: 
        diff = ImageChops.difference(image_one, image_two)
 

        if diff.getbbox() is None:
        # 图片间没有任何不同则直接退出
            print("【+】We are the same!")
        else:
            diff.save(diff_save_location)
    except ValueError as e:
        text = ("表示图片大小和box对应的宽度不一致，参考API说明：Pastes another image into this image."
                "The box argument is either a 2-tuple giving the upper left corner, a 4-tuple defining the left, upper, "
                "right, and lower pixel coordinate, or None (same as (0, 0)). If a 4-tuple is given, the size of the pasted "
                "image must match the size of the region.使用2纬的box避免上述问题")
        print("【{0}】{1}".format(e,text))

 
if __name__ == '__main__':
    compare_images('fake.png',
                   'real.png',
                   '我们不一样.png')


# In[1]:


import librosa
y, sr = librosa.load('/home/dl/deepfake/data/videos/dfdc_train_part_0/hpwdlgshqg.mp4')


# In[ ]:




