#!/usr/bin/env python
# coding: utf-8

# In[6]:


from IPython.display import HTML
from base64 import b64encode
TRAIN_SAMPLE_FOLDER = 'train_sample_videos'
def play_video(video_file, subset=TRAIN_SAMPLE_FOLDER):
    '''
    Display video
    param: video_file - the name of the video file to display
    param: subset - the folder where the video file is located (can be TRAIN_SAMPLE_FOLDER or TEST_Folder)
    '''
    video_url = open(video_file,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(video_url).decode()
    return HTML("""<video width=500 controls><source src="%s" type="video/mp4"></video>""" % data_url)


# In[7]:


from IPython.core.display import Image, display
def show_img(img_file):
    return display(Image(img_file, width=290, unconfined=True))


# In[2]:


import json
def show_path(dir_path):
    with open(dir_path+'/metadata.json','r') as fp:
        data = json.load(fp)
    return data


# 视频名称

# In[5]:


# 请输入 part 

part = 0

data = show_path(f'../data/videos/dfdc_train_part_{part}')
print(len(data.keys()))
for k in list(data.keys()):
    print(k,data[k])


# 播放视频

# In[10]:


# part
part = 0

# mp4
mp4_file = 'lmjzfqizks.mp4'

#
#
play_video(f'../data/dfdc_train_part_{part}/{mp4_file}')


# 展示图片

# In[11]:


# part
part = 0

# mp4
img_file = 'frmzkdhkzw'

# img id     0 - 299
img_id = 1

#
#
show_img(f'../data/detect_result/dfdc_train_part_{part}/{img_file}/{img_id}_0.png')


# In[ ]:




