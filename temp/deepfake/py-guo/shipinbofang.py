#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


img_path='./aaqaifqrwn.mp4'
play_video(img_path)


# In[ ]:




