import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv
import keras
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
'''
#  * Blues
#  * Classical
#  * Country
#  * Disco
#  * Hiphop
#  * Jazz
#  * Metal
#  * Pop
#  * Reggae
#  * Rock
cmap = plt.get_cmap('inferno')

plt.figure(figsize=(10,10))
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)     
    for filename in os.listdir(f'./MIR/genres/{g}'):
        songname = f'./MIR/genres/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=5)
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')
        plt.clf()
'''


header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate std_chroma_stft std_rmse std_spec_cent std_spec_bw std_rolloff std_zcr max_rmse min_rmse max_spec_cent min_spec_cent max_spec_bw min_spec_bw max_rolloff min_rolloff max_zcr min_zcr'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

print('Start computing feats and writing to csv!!!')
file = open('data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in tqdm(genres):
    for filename in os.listdir(f'./MIR/genres/{g}'):
        songname = f'./MIR/genres/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        #bug
        rmse = librosa.feature.rms(y)

        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} {np.std(chroma_stft)} {np.std(rmse)} {np.std(spec_cent)} {np.std(spec_bw)} {np.std(rolloff)} {np.std(zcr)} {np.max(rmse)} {np.min(rmse)} {np.max(spec_cent)} {np.min(spec_cent)} {np.max(spec_bw)} {np.min(spec_bw)} {np.max(rolloff)} {np.min(rolloff)} {np.max(zcr)} {np.min(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())



print('FINISH WRITING TO CSV!!!')

############################

from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras import models
from keras import layers


df = pd.read_csv('data.csv')
data = df.copy()
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, 1:-1], dtype = float))

index = []
print('use 990+100 to apply last class')
for i in range(90,990,100):
    j = i + 10
    temp = list(range(i,j))
    index.extend(temp)

X_test, y_test = X[index], y[index]

t_i = train_index = []
print('use 900+100 to apply to last class')
for i in range(0,900,100):
    j = i + 90
    temp = list(range(i,j))
    train_index.extend(temp)
X_train, y_train = X[t_i], y[t_i]

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train,
                    y_train,
                    epochs=30,
                    batch_size=128,
                    validation_split=0.15)

if True:


    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()





preds = model.predict(X_test)
pred = np.argmax(preds,axis=1)
j = 0
print('change 90 to 100 for last class')
for i in range(0,100-10,10):
    print('expected: ', j)
    for k in range(i,i+10):
        print(pred[k])
    print('---')
    j+=1
