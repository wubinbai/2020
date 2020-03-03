import numpy as np 
import pandas as pd 
import os, sys, random
import numpy as np
import pandas as pd
import cv2
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch import Tensor

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import _utils

from random import choice

from skimage import io
from PIL import Image

import glob

#from torchsummary import summary
import logging

import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models
# from tqdm.notebook import tqdm
from tqdm import tqdm
from pretrainedmodels import se_resnext50_32x4d, se_resnet152
from sklearn.utils import shuffle
from apex import amp

import random


# def make_splits(df, ratio = ):
#     real = df[df['label'] == 'REAL']
#     fake = df[df['label'] == 'FAKE']
#     real_train = real.sample(frac = ratio)
#     real_val = real[~real.index.isin(real_train.index)]
#     fake_train = fake[fake['original'].isin(real_train.index)]
#     fake_val = fake[fake['original'].isin(real_val.index)]
#     return fake_train, pd.concat([real_val, fake_val])

def make_split_part(df, ratio = 0.8):
    real = df[df['label'] == 'REAL']
    real_train = real.sample(frac = ratio)
    real_val = real[~real.index.isin(real_train.index)]
    train = df[df['original'].isin(real_train.index) | df.index.isin(real_train.index)]
    val = df[df['original'].isin(real_val.index) | df.index.isin(real_val.index)]
    return train, val

def make_splits(k = 50, ratio = 0.8):
    train_list = []
    val_list = []
    for i in range(k):
        Metadata_Path = f'./metadatas/{i+1}metadata.json'
        df = pd.read_json(Metadata_Path).transpose()
        df['path'] = f'dfdc_train_part_{i}'
        train_df, val_df = make_split_part(df, ratio = ratio)
        train_list.append(train_df)
        val_list.append(val_df)
    train_df, val_df = pd.concat(train_list), pd.concat(val_list)
    return train_df, val_df


input_size = (224, 224)

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])




class VideoTrainDataset(Dataset):
    def __init__(self, df, file_path, fake_ratio = 1):
        self.df = df
        self.real_df = self.df[self.df['label'] == 'REAL']
        self.file_path = file_path
        self.fake_ratio = fake_ratio
    def __getitem__(self, index):
        '''
        training set 的dataloader 返回tuple，里边包含两个路径
        1.fake视频的路径
        2.fake对应的real视频的路径
        '''
        row = self.real_df.iloc[index]
        fake = self.df[self.df['original'] == row.name]
        fake = fake.sample(n = self.fake_ratio) if len(fake) > self.fake_ratio else fake
        fake_path = [f"{self.file_path}/{fake.iloc[i]['path']}/{fake.iloc[i].name.split('.')[0]}" for i in range(len(fake))]

        return f"{self.file_path}/{row['path']}/{row.name.split('.')[0]}", fake_path
    
    def __len__(self):
        return len(self.real_df )


class VideoValDataset(Dataset):
    def __init__(self, df, file_path):
        self.df = df
        self.file_path = file_path
        
    def __getitem__(self, index):
        '''
        返回一个tuple
        第一个元素返回对应是视频的路径
        第二个是对应的label
        '''
        row = self.df.iloc[index]
        return (f"{self.file_path}/{row['path']}/{row.name.split('.')[0]}", 1 if row['label'] == 'FAKE' else 0)
    
    def __len__(self):
        return len(self.df)


# def SampleVideo(path, label, K = 1):
#     '''
#     every time sample K frames
#     first sample from [0:L//k]
#     second sample from [L//K, 2*L//K]
#     .....
#     .....
#     The last sample from [(K - 1)L//K, end]

#     根据视频的路径，找到对应的文件夹，从文件夹中随机抽取K张图片，如果不足K张，就全部抽取
#     '''
#     faces = glob.glob(f'{path}/*')
#     if faces == []:
#         return []
#     if len(faces) <= K:
#         return [(transform(Image.open(i)), label) for i in faces]
#     interval = len(faces)//K if K != 1 else len(faces)
#     return [(transform(Image.open(choice(faces[i:i+interval if i + 2*interval <= len(faces) else len(faces)]))), label) for i in range(0, K*interval, interval)]      


def SampleVideo(path, label, K = 1):
    faces = glob.glob(f'{path}/*')
    if faces == []:
        return []
    if len(faces) > K:
        faces = random.sample(faces, K)
    return [(transform(Image.open(i)), label) for i in faces]



def collate_train_fn(batches):
    '''
    training set 根据视频路径，采样视频对应文件夹下的图片，
    '''
    b = []
    for i in batches:
        b += SampleVideo(i[0], 0, K = len(i[1]) if len(i[1]) > 0 else 1)
        for j in i[1]:
            b += SampleVideo(j, 1)
    return _utils.collate.default_collate(b)


def collate_val_fn(batches):
    '''
    training set 根据视频路径，采样视频对应文件夹下的图片，
    '''
    b = []
    for i in batches:
        b += SampleVideo(i[0], i[1], K = 2)
    return _utils.collate.default_collate(b)






def create_data_loaders(file_path, train_df, val_df, batch_size, num_workers, collate_fn = None):
    '''
    创建dataloader，
    training set 用 collate_train_fn
    validation set 用 collate_val_fn
    '''
    logging.info(f"training size is {len(train_df[train_df['label'] == 'REAL']), len(train_df[train_df['label'] == 'FAKE'])}, validation size is {len(val_df[val_df['label'] == 'REAL']), len(val_df[val_df['label'] == 'FAKE'])}")
    train_set = VideoTrainDataset(train_df, file_path)
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers=num_workers, pin_memory=True, collate_fn = collate_train_fn)
    val_set = VideoValDataset(val_df, file_path)
    val_loader = DataLoader(val_set, batch_size = batch_size*2, shuffle = False, num_workers=num_workers, pin_memory=True, collate_fn = collate_val_fn) 
    return train_loader, val_loader


def get_My_se_resnext50_32x4d(prtrained = True):
    if prtrained == True:
        model = se_resnext50_32x4d()
        model.last_linear = nn.Linear(model.last_linear.in_features, 1)
    else:
        model = se_resnext50_32x4d(num_classes= 1, pretrained=None)
    return model



def evaluate(net, data_loader, device):
    '''
    计算验证集误差
    '''
    net.train(False)
    
    bce_loss = 0
    total_size = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            total_size += len(data[0])
            #put data to GPU
            x = data[0].to(gpu)
            label = data[1].to(gpu).float()

            y_pred = net(x)
            y_pred = y_pred.squeeze()

            bce_loss += F.binary_cross_entropy_with_logits(y_pred, label, reduction = 'sum').item()
        else:
            bce_loss /= total_size
        
    return bce_loss

def evaluate_batch(net, epoch, best_loss, bce_loss, total_size, optimizer, history):
    '''
    计算验证集测试集的loss，并保存checkpoint
    '''
    bce_loss /= total_size
    val_bce = evaluate(net, val_loader, device = gpu)

    history['train_bce'].append(bce_loss)
    history['val_bce'].append(val_bce)

    checkpoint = {
        'best_loss':best_loss,
        'epoch': epoch+1,
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(checkpoint, 'se_resnext50_32x4d_running_checkpoint.pth')
    if val_bce <= best_loss:
        best_loss = val_bce
        shutil.copy('se_resnext50_32x4d_running_checkpoint.pth', 'se_resnext50_32x4d_checkpoint.pth')
            

    logging.info(f"Epoch: {epoch+1}, train BCE:{bce_loss}, val BCE {val_bce}")
    return bce_loss, val_bce


def train(net, dataloader:tuple, epochs, optimizer, scheduler, checkpoint = None):
    '''
    checkpoint is a dict
    '''
    global gpu
    history = {'train_bce':[], 'val_bce':[]}
    train_loader, val_loader = dataloader

    if checkpoint is None:
        start_epoch = 0
        best_loss = float('inf')
    else:
        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    criterion = nn.BCEWithLogitsLoss(reduction = 'sum')
    
    for epoch in range(start_epoch, epochs):    
        bce_loss = 0
        net.train(True)

        total_size = 0
        val_bce = None
        temp = []
        for batch_idx, data in enumerate(tqdm(train_loader)):
            #put data to GPU
            total_size += len(data[0])
            x = data[0].to(gpu)
            label = data[1].to(gpu).float()

            # zero the parameter gradients
            optimizer.zero_grad()

            #forward 
            y_pred = net(x)
            y_pred = y_pred.squeeze()
            
            #calculate loss
            loss = criterion(y_pred, label)

            #backward,calculate gradients
            loss.backward()
            #backward,update parameter
            optimizer.step()
            
            #the learning rate should be update after optimizer's update 
            #change the learning rate, because using One cycle pollicy,the learning rate should be update per mini-batch
            # scheduler.step()

            bce_loss += loss.item()
            # if batch_idx != 0 and (batch_idx % length) == 0:
            #     temp.append((bce_loss, total_size))
            #     evaluate_batch(net, epoch, best_loss, bce_loss, total_size, optimizer, history)
            #     total_size = 0
            #     bce_loss = 0
        else:
            train_bce, val_bce = evaluate_batch(net, epoch, best_loss, bce_loss, total_size, optimizer, history)
        scheduler.step()

        # train_bce = sum([i[0] for i in temp])/sum([i[1] for i in temp])
        print(f"Epoch: {epoch+1}, train BCE:{train_bce}, val BCE {val_bce}")           
    print("Finished")
    return history



if __name__ == '__main__':
    log_file = 'se_resnext50_32x4d.log'
    if os.path.exists(log_file):
        os.remove(log_file)
    logging.basicConfig(filename=log_file, level = logging.INFO)

    # os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Faces_Path = './../data/detect_result_new'

    train_loader, val_loader = create_data_loaders(Faces_Path, *make_splits(k = 50), batch_size = 100 , num_workers = 2)
    
    max_lr = 5e-4
    epochs = 40
    logging.info(f"The maximum lr is {max_lr}")
    
    wd = 0.
    model = get_My_se_resnext50_32x4d().to(gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, weight_decay=wd)
    # checkpoint = torch.load('./se_resnext50_32x4d_running_checkpoint.pth')
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=epochs)
    history = train(model, (train_loader, val_loader), epochs, optimizer, scheduler)

    logging.info(history)
