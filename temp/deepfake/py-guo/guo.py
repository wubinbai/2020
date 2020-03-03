#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython.display import HTML
from base64 import b64encode
TRAIN_SAMPLE_FOLDER = 'train_sample_videos'
def play_video(video_file):
    video_url = open(video_file,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(video_url).decode()
    return HTML("""<video width=500 controls><source src="%s" type="video/mp4"></video>""" % data_url)


# In[3]:


dir_path = './data/dfdc_train_part_0/'


# In[ ]:





# In[4]:


import json
with open(dir_path+'metadata.json','r') as fp:
    data = json.load(fp)
print(type(data))
for k in list(data.keys())[:5]:
    print(k,data[k])


# In[5]:


play_video(dir_path+'owxbbpjpch.mp4')


# In[6]:


play_video(dir_path+'wynotylpnm.mp4')


# In[2]:


pa = './data/detect_result/dfdc_train_part_20/fflfnwokgs/240_0.png'
import cv2 as cv


# In[4]:


from IPython.core.display import Image, display
display(Image('./data/detect_result/dfdc_train_part_20/fflfnwokgs/240_0.png', width=290, unconfined=True))


# In[ ]:




