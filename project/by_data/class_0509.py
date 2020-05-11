#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 22:34:47 2020

"""


#基本信息
import numpy as np
import pandas as pd
from pandas import Series,DataFrame

#股票数据的读取 pip install pandas_datareader
import pandas_datareader as pdr

#可视化
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime


#1.获取数据
alibaba = pdr.get_data_yahoo("BABA")

#2.保存数据
alibaba.to_csv('BABA.csv')
#alibaba.to_csv('BABA.csv',index=None)



#3.读取数据
alibaba_df = pd.read_csv("BABA.csv")




#4.数据类型
print(alibaba_df.shape)
print(len(alibaba_df))



#5.读取前10行
print(alibaba_df.head())


#5.读取后10行
print(alibaba_df.tail())

#6.读取表格名称
print(alibaba_df.columns)

#7.表格情况
print(alibaba_df.info())




#8.画折线图
alibaba_df["Adj Close"].plot(legend=True)
alibaba_df["Volume"].plot(legend=True)




#9.复制数据
a_copy = alibaba_df.copy()
a_copy['Low'] = 1


b_copy = a_copy
b_copy['Low'] = 2




#10.数据处理

mid = alibaba_df['Open']>83
open_number = alibaba_df[alibaba_df['Open']>83]
print(open_number.shape)


open_number = alibaba_df[(alibaba_df['Open']>85)&(alibaba_df['Open']<100)]
print(open_number.shape)


#11.数据处理

alibaba_df['Open1'] = alibaba_df['Open'].map(lambda x:x+1)


all_sum = alibaba_df['Open'].sum()
all_mean = alibaba_df['Open'].mean()
all_std = alibaba_df['Open'].std()
all_max = alibaba_df['Open'].max()
all_min = alibaba_df['Open'].min()




#12.日期处理 2015/03/01

alibaba_df['date'] = alibaba_df['Date'].map(lambda x : datetime.strptime(x,'%Y-%m-%d'))


alibaba_df['day'] = alibaba_df['date'].dt.day
alibaba_df['week'] = alibaba_df['date'].dt.weekday
alibaba_df['month'] = alibaba_df['date'].dt.month
alibaba_df['year'] = alibaba_df['date'].dt.year






#13.以目标索引
data_year = alibaba_df.groupby(['year'])['Low'].sum().reset_index()
data_year = data_year.rename(columns={"Low": "Low_y_sum"})



data_week = alibaba_df.groupby(['year','week'])['Low'].sum().reset_index()
data_week = data_week.rename(columns={"Low": "Low_w_sum"})







#14.数据拼接
alibaba_df1 = pd.merge(alibaba_df,data_year,on='year',how='left')

alibaba_df2 = pd.merge(alibaba_df,data_week,on=['year','week'],how='left')




data_year_con1 = pd.concat([data_year,data_week],axis=1)
data_year_con0 = pd.concat([data_year,data_week],axis=0)






#15.数据填充
data_year_con1 = data_year_con1.fillna(0)


















