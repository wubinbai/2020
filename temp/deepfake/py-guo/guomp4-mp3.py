#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np 
import pandas as pd 
import subprocess
import glob
import os
from pathlib import Path
import shutil
from zipfile import ZipFile


# In[27]:


# 创建文件夹
output_format = 'mp3'  # can also use aac, wav, etc

output_dir = Path(f"{output_format}s")
#Path(output_dir).mkdir(exist_ok=True, parents=True)


# In[34]:


dir_path='./data'
for i in range(50):
    dfdc = f'dfdc_train_part_{i}'
    list_of_files = glob.glob('./data/dfdc_train_part_*/*.mp4') 
    


# In[31]:


list_of_files = glob.glob('./data/dfdc_train_part_*/*.mp4') 
list_of_files


# In[7]:


for file in list_of_files:
    command = f"ffmpeg -i {file} -ab 192000 -ac 2 -ar 44100 -vn {output_dir/file[-14:-4]}.{output_format}"
    subprocess.call(command, shell=True)


# In[8]:


import IPython.display as ipd  # To play sound in the notebook
fname = './mp3s/owxbbpjpch.mp3'   # Hi-hat
ipd.Audio(fname)


# In[9]:


ffmpeg -i input.mp4 -r 1 -q:v 2 -f image2 pic-%03d.jpeg


# In[21]:


# 创建文件夹
output_format = 'ffe-jpeg'  # can also use aac, wav, etc

output_dir = Path(f"{output_format}s")
Path(output_dir).mkdir(exist_ok=True, parents=True)


# In[25]:


list_of_files = glob.glob('./data/dfdc_train_part_0/*.mp4') 


# In[26]:


for file in list_of_files:
    Path(f'{output_dir/file[-14:-4]}').mkdir(exist_ok=True, parents=True)
    command = f"ffmpeg -i ./data/dfdc_train_part_0/owxbbpjpch.mp4  -r 1 -q:v 2 -f image2 {output_dir/file[-14:-4]}/pic-%03d.jpeg"
    subprocess.call(command, shell=True)


# In[35]:


import librosa


# In[ ]:




