#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# # Parameters

# In[2]:


# path = "/home/dl/deepfake/data/detect_result_20/"
path = "/home/dl/deepfake/data/detect_result_new/"
version = '0219'


# In[17]:


get_ipython().run_cell_magic('time', '', '# 整体数据集，不含无文件的数据\nall_png_pathList = [os.path.join(dirname, filename) for dirname, _, filenames in os.walk(path) for filename in filenames]\nlen(all_png_pathList)\nall_png_pathDf = [all_png_pathList[i].split("/")[3:] for i in range(len(all_png_pathList))]\ndf = pd.DataFrame(all_png_pathDf)\ncol_list = [\'deepfake\', \'data\', \'detect_result\', \'dfdc_train_part\', \'video\', \'png\']\ndf.columns = col_list\nprint(df.shape)\ndf.head()')


# In[18]:


get_ipython().run_cell_magic('time', '', '# 无文件的数据\n\n# 获得一级文件list\nfile_list0 =  sorted(os.listdir(path))\n# 获得二级文件夹名称列表，对应了file_list0的一级目录\nfile_list1 = [sorted(os.listdir(path+i)) for i in file_list0]\nall_folder_pathList = [[file_list0[i], file_list1[i][j]] for i in range(len(file_list0)) for j in range(len(file_list1[i]))]\nlen(all_folder_pathList)\nfolder_df = pd.DataFrame(all_folder_pathList)\nfolder_df.shape\nfolder_df.columns = [\'dfdc_train_part\', \'video\']\nnon_png = folder_df[~folder_df[\'video\'].isin(df[\'video\'])]\nfor col in [\'deepfake\', \'data\', \'detect_result\']:\n    non_png[col] = df[col]\nnon_png[\'png\'] = np.nan\nnon_png = non_png[col_list]\nnon_png.to_csv("df_groupby_number_of_png_0_{}.csv".format(version))\nprint(non_png.shape)\nnon_png.head()')


# In[5]:


# 合并数据
df = pd.concat([df, non_png])
df.shape


# In[6]:


df['_0'] = df['png'].apply(lambda x: x.split("_")[-1].replace("png", "") if type(x)==str else x)
df.head()


# In[7]:


df['notHave1'] = 0
df['notHave1'].loc[df['_0']=='0.'] = 1
df.head()


# In[8]:


df.drop(['data', 'deepfake', 'detect_result'], axis=1).to_csv("all_png_path_data_byLeo_{}.csv".format(version))


# In[9]:


# 文件名后缀统计
df['_0'].value_counts()


# In[10]:


# 各视频中抽取结果是_0的数量对比
df_groupby_png = pd.DataFrame(df.groupby(['detect_result', 'dfdc_train_part', 'video'])['notHave1'].sum().reset_index())
df_groupby_png.head()


# In[11]:


# 抽取图片数量汇总
df_groupby_png["notHave1"].value_counts(normalize=True)
df_groupby_png["notHave1"].value_counts()


# In[12]:


# 各视频中抽取结果是_0的数量在10以下的数据
df_groupby_png10 = df_groupby_png[df_groupby_png['notHave1']<=10]
df_groupby_png10.to_csv('df_groupby_number_of_png_0-10_{}.csv'.format(version))
df_groupby_png10.shape


# In[13]:


# 各视频中抽取结果是_0的数量在5以下的数据
df_groupby_png5 = df_groupby_png[df_groupby_png['notHave1']<=5]
df_groupby_png5.to_csv('df_groupby_number_of_png_0-5_{}.csv'.format(version))
df_groupby_png5.shape


# In[14]:


# 各视频中抽取结果是_0的数量为0的数据
df_groupby_png0 = df_groupby_png[df_groupby_png['notHave1']==0]
df_groupby_png0.to_csv('df_groupby_number_of_png_0_{}.csv'.format(version))
df_groupby_png0.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ##  -------------------------分割线-----------------------------
# 以下代码无需使用

# In[ ]:


# 以下代码无需使用


# In[ ]:





# In[27]:


# 获得一级文件list
file_list0 =  sorted(os.listdir(path))
file_list0[:3]


# In[28]:


len(file_list0)


# In[29]:


file_list1 = []
for i in file_list0:
    temp_file_list = sorted(os.listdir(path+i))
    
len(temp_file_list)


# In[30]:


# 获得二级文件夹名称列表，对应了file_list0的一级目录
file_list1 = [sorted(os.listdir(path+i)) for i in file_list0]
len(file_list1[-1])


# In[73]:


# 二级文件总数
n = 0
for i in range(len(file_list0)):
#     print(len(file_list0))
    for j in range(len(file_list1[i])):
#         print(len(file_list1[i]))
        n += 1
print(n)

file_list2 = [sorted(os.listdir(path+i+"/"+j)) for i in file_list0]for i in range(len(file_list0)):
#     print(len(file_list0))
    for j in range(len(file_list1[i])):
        file_list2import os
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
#         print(os.path.join(dirname, filename))
# In[77]:


get_ipython().run_cell_magic('time', '', "all_folder_pathList = [[file_list0[i], file_list1[i][j]] for i in range(len(file_list0)) for j in range(len(file_list1[i]))]\nlen(all_folder_pathList)\nfolder_df = pd.DataFrame(all_folder_pathList)\nfolder_df.shape\nfolder_df.columns = ['dfdc_train_part', 'video']\nnon_png = folder_df[~folder_df['video'].isin(df['video'])]\nfor col in ['deepfake', 'data', 'detect_result']:\n    non_png[col] = df[col]\nnon_png.head()")


# In[89]:


folder_df.columns = ['dfdc_train_part', 'video']
folder_df.head()


# In[94]:


non_png = folder_df[~folder_df['video'].isin(df['video'])]
len(non_png)


# In[99]:


for col in ['deepfake', 'data', 'detect_result']:
    non_png[col] = df[col]
non_png.head()


# In[32]:


get_ipython().run_cell_magic('time', '', 'all_png_pathList = [os.path.join(dirname, filename) for dirname, _, filenames in os.walk(path) for filename in filenames]\nlen(all_png_pathList)\nall_png_pathDf = [all_png_pathList[i].split("/")[3:] for i in range(len(all_png_pathList))]\ndf = pd.DataFrame(all_png_pathDf)\ncol_list = [\'deepfake\', \'data\', \'detect_result\', \'dfdc_train_part\', \'video\', \'png\']\ndf.columns = col_list')

for i in range(10):
    print(all_png_pathList[i])all_png_pathDf = [all_png_pathList[i].split("/")[3:] for i in range(len(all_png_pathList))]
len(all_png_pathDf)for i in range(10):
    print(all_png_pathDf[i])import pandas as pd
df = pd.DataFrame(all_png_pathDf)df.head()col_list = ['deepfake', 'data', 'detect_result', 'dfdc_train_part', 'video', 'png']
df.columns = col_listdf.columns = col_list
# In[41]:


len(df)


# In[42]:


df.head()


# In[43]:


df.to_csv("all_png_pathDf0217byLeo.csv")


# In[44]:


df['_0'] = df['png'].apply(lambda x: x.split("_")[-1].replace(".png", ""))


# In[45]:


df.head()


# In[46]:


df[df["_0"]=="1"]


# In[47]:


df["_0"].value_counts()


# In[48]:


df.loc[(df["_0"]!="0") & (df["_0"]!="1")]


# In[50]:


df_notHave1 = df[df["_0"]=="0"]


# In[51]:


df_notHave1.shape


# In[52]:


df.columns


# In[53]:


# 各视频中抽取结果是_0的数量对比
df_groupby_png = pd.DataFrame(df_notHave1.groupby(['detect_result', 'dfdc_train_part', 'video'])['png'].count().reset_index())
df_groupby_png


# In[54]:


# 抽取图片数量汇总
df_groupby_png["png"].value_counts(normalize=True)
df_groupby_png["png"].value_counts()


# In[55]:


# 各视频中抽取结果是_0的数量在10以下的数据
df_groupby_png10 = df_groupby_png[df_groupby_png['png']<=10]
df_groupby_png10.shape


# In[ ]:




