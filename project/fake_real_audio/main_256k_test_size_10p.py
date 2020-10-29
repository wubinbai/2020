import json
from keras.models import Model
import librosa
import librosa.display
import pandas as pd
import numpy as np
from glob import glob
import os
from tqdm import tqdm
import traceback
import cv2
import sklearn
from PIL import Image
import math
from imgaug import augmenters as iaa
from keras.layers import *
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras import optimizers as opts
from keras import regularizers
from keras.utils import np_utils, Sequence

num_classes = 25

augment_img = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.CoarseDropout(0.12,size_percent=0.05)], random_order=True)

#shape1 = 128
#shape2 = 47
#shape2=76
#shape2 = 69
tmp_f = '../mel_pictures/train/256k_real/FF47DBDB_9.jpg'
w,h = Image.open(tmp_f).size
shape1 = h
shape2 = w

def conv_block(inputs,
        neuron_num,
        kernel_size,
        use_bias,
        padding= 'same',
        strides= (1, 1),
        with_conv_short_cut = False):
    conv1 = Conv2D(
        neuron_num,
        kernel_size = kernel_size,
        activation= 'relu',
        strides= strides,
        use_bias= use_bias,
        padding= padding
    )(inputs)
    conv1 = BatchNormalization(axis = 1)(conv1)

    conv2 = Conv2D(
        neuron_num,
        kernel_size= kernel_size,
        activation= 'relu',
        use_bias= use_bias,
        padding= padding)(conv1)
    conv2 = BatchNormalization(axis = 1)(conv2)

    if with_conv_short_cut:
        inputs = Conv2D(
            neuron_num,
            kernel_size= kernel_size,
            strides= strides,
            use_bias= use_bias,
            padding= padding
            )(inputs)
        return add([inputs, conv2])

    else:
        return add([inputs, conv2])





def get_model():
    inputs = Input(shape= [shape1, shape2, 3])
    x = ZeroPadding2D((3, 3))(inputs)


    # Define the converlutional block 1

    x = Conv2D(64, kernel_size= (7, 7), strides= (2, 2), padding= 'valid')(x)
    x = BatchNormalization(axis= 1)(x)
    x = MaxPooling2D(pool_size= (3, 3), strides= (2, 2), padding= 'same')(x)

    # Define the converlutional block 2

    x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)

    # Define the converlutional block 3
    x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
    x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True)

    # Define the converlutional block 4
    x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
    x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)

    # Define the converltional block 5
    x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
    x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    # x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs= inputs, outputs= x)
    return model


#shape1 = 128
#shape2 = 47
#shape2 = 69
#shape2=76
tmp_f = '../mel_pictures/train/256k_real/FF47DBDB_9.jpg'
w,h = Image.open(tmp_f).size
shape1 = h
shape2 = w


class DataGenerator(Sequence):
    
    def __init__(self, data_paths, data_labels, batch_size, augument=False, shuffling=False, test_data=False, mixup=False, mixup_prob=0.3):
        self.batch_size = batch_size
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.augument = augument
        self.shuffling = shuffling
        self.test_data = test_data
        self.indexes = np.arange(len(self.data_paths))
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        
    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.ceil(len(self.data_paths) / float(self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffling == True:
            print("shuffle the train data...")
            np.random.shuffle(self.indexes)
            
    def __getitem__(self, index):

        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        batch_paths = [self.data_paths[k] for k in batch_indexs]
        if not self.test_data:
            batch_labels = [self.data_labels[k] for k in batch_indexs]
            X, y = self.data_generation(batch_paths, batch_labels)
            return X, y
        
        else:
            X = self.data_generation(batch_paths, None)
            return X
    
    def mix_up(self, x, y):
        x = np.array(x, np.float32)
        lam = np.random.beta(1.0, 1.0)
        ori_index = np.arange(int(len(x)))
        index_array = np.arange(int(len(x)))
        np.random.shuffle(index_array)        
        
        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]
        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]
        
        return mixed_x, mixed_y
    
    def get_img(self, batch_paths, batch_labels):
        X_batch = []
        y_batch = []
        for i in range(len(batch_paths)):
            
            image = Image.open(batch_paths[i])
            arr = np.asarray(image)
            X_batch.append(arr)
            if not self.test_data:
                y = np_utils.to_categorical(batch_labels[i], num_classes)
                y_batch.append(y)
        if not self.test_data:
            return np.array(X_batch), np.array(y_batch)
        else:
            return np.array(X_batch)

    def data_generation(self, batch_paths, batch_labels):
        if not self.test_data:
            X_batch, y_batch = self.get_img(batch_paths, batch_labels)
        else:
            X_batch = self.get_img(batch_paths, batch_labels)
        
        if self.augument:
            X_batch = self.augment(X_batch)

        if (self.mixup and self.test_data == False):
            dice = np.random.rand(1)
            if dice > self.mixup_prob:
                X_batch, y_batch =  self.mix_up(X_batch, y_batch)    
        X_batch = X_batch / 255.
        if not self.test_data:
            return X_batch, y_batch
        else:
            return X_batch
                
    
    def augment(self, image):
        image_aug = augment_img.augment_images(np.array(image))
        return image_aug





#label_to_dict = {"awake": 0, "diaper": 1, "hug": 2, "hungry": 3, "sleepy": 4, "uncomfortable": 5}


### prepare label_to_dict
label_to_dict = dict()
#### load som utilities to get label
from speaker_distribution import *
assert speaker_256k[3] == '0'
speaker_256k = np.delete(speaker_256k,3)
#### finish loading
for i in range(len(speaker_256k)):
    label_to_dict[speaker_256k[i]] = i


#train_256k_real_wavs = librosa.util.find_files('../eda/train/256k/',recurse=False)


data_paths = glob('../mel_pictures/train/256k_real/*')
data_wavs_name = [data_path.split('/')[-1].split('_')[0] + '.wav' for data_path in data_paths]
data_labels0 = [df.loc[df.loc[df['Audio_Name'] == wav_name].index[0],'Speaker_ID'] for wav_name in tqdm(data_wavs_name)]
data_labels = [label_to_dict[k] for k in data_labels0]
#data_labels = [label_to_dict[data_path.split('/')[-1].split('_')[0]] for data_path in data_paths]

test_paths = glob('../mel_pictures/test/256k/*')
test_files = [test_path.split('/')[-1].split('_')[0] + '.wav' for test_path in test_paths]

from sklearn.model_selection import train_test_split

train_paths, val_paths, train_labels, val_labels =  train_test_split(data_paths, data_labels, test_size=0.1, stratify=data_labels)

###
#val_paths = []
#val_labels = []
#train_paths = []
#train_labels = []
#for i in range(len(data_paths)):
#    temp_path = data_paths[i]
#    temp_label = data_labels[i]
#    if '65' in temp_path or '75' in temp_path or '85' in temp_path or '95' in temp_path or '55' in temp_path or '45' in temp_path:
#        val_paths.append(temp_path)
#        val_labels.append(temp_label)
#    else:
#        train_paths.append(temp_path)
#        train_labels.append(temp_label)




num_classes = 25 
#batch_size = 32*4
#batch_size = 32
batch_size = 8

train_generator = DataGenerator(train_paths,train_labels, batch_size = batch_size, augument=True, shuffling=True, test_data=False, mixup=False, mixup_prob=0.5)
val_generator = DataGenerator(val_paths,val_labels, batch_size = batch_size, augument=False, shuffling=False, test_data=False, mixup=False, mixup_prob=0.3)
test_generator = DataGenerator(test_paths, None, batch_size, augument=False, shuffling=False, test_data=True, mixup=False, mixup_prob=0.3)


epochs = 45

reduce = ReduceLROnPlateau(monitor='val_acc',factor=0.5, patience=2,verbose=1,mode='max')

checkpointer = ModelCheckpoint(filepath='../h5s/best_resnet34_iaa_255_shuffle.h5', monitor='val_acc',mode='max',verbose=1,save_best_only=True)
stop = EarlyStopping(monitor='val_acc', patience=6, verbose=1, mode='max')

adam = opts.Adam(1e-4)
model = get_model()
model.compile(optimizer = adam, loss='categorical_crossentropy',metrics=['acc'])

history = model.fit_generator(train_generator,steps_per_epoch=math.ceil(len(train_paths)/batch_size), epochs=epochs,validation_data=val_generator, validation_steps = math.ceil(len(val_paths) / batch_size), callbacks=[checkpointer,reduce, stop], verbose=1)



model.load_weights("../h5s/best_resnet34_iaa_255_shuffle.h5")

resnet_result = model.predict_generator(test_generator, steps=math.ceil(len(test_paths)/batch_size), verbose=True)
# 生成提交
result_df = pd.DataFrame(resnet_result, columns=[f'index_{i}_prob' for i in range(num_classes)])
result_df['paths'] = test_files

id_ = []
label = []
for path in result_df['paths'].unique():
    id_.append(path)
    label.append(np.argmax(result_df[[f'index_{i}_prob' for i in range(num_classes)]][result_df['paths'] == path].mean().values))

#label_dict = {"awake":0, "diaper":1, "hug":2, "hungry":3, "sleepy":4, "uncomfortable":5}
label_dict = label_to_dict
pred2label = {v:k for k,v in label_dict.items()}

label_string = [pred2label[k] for k in label]
pred_256k_dict = dict()
for i in range(len(id_)):
    pred_256k_dict[id_[i]] = label_string[i]

with open('../jsons/pred_256k_dict.json','w') as f:
    json.dump(pred_256k_dict,f)



#with open('./data/same_file.json','r')as fp:
#    same_file = json.load(fp)
#ids_ = []
#labels = []
#for i in range(len(id_)):
#    ids_.extend(same_file[id_[i].split('.')[0]])
#    labels.extend([pred2label[label[i]]] * 33)
#
#df = pd.DataFrame({'id': ids_, 'label':labels})
#
#df.to_csv("submit_resnet34_double.csv",index=None)
#
#
plt.figure()
plt.plot(history.history['val_acc'])
plt.plot(history.history['acc'])
win = 2.2
plt.savefig('../history_plots/history{}.png'.format(win))


