from tqdm import tqdm

from glob import glob as gl
import numpy as np
import pandas as pd
import glob

def get_test_data():
    test_csv = glob.glob('../input/test/*.csv')
    l = []
    clips_list = []
    for f in tqdm(test_csv):
        df = pd.read_csv(f)
        #selected = df[df.confidence>df.confidence.mean()].frequency
        #selected = np.pad(selected,(5000-len(selected),0))
        selected = select_harm(df) 
        #print(df.shape)
        #l.append((df,f.split('/')[-1].replace('f0.','')))
        #l.append(selected)
        num_clips = selected.shape[0]
        if num_clips > 0:
            l.append(selected)
            clips_list.append(num_clips)
        else:
            print(f)
    return l,clips_list#test_csv

def get_train_data():
    train_csv = glob.glob('../input/train/*/*.csv',recursive=True)
    l = []
    for f in tqdm(train_csv[:]):
        df = pd.read_csv(f)
        #selected = df[df.confidence>df.confidence.mean()].frequency
        #selected = np.pad(selected,(5000-len(selected),0))
        selected = select_harm(df) 
        #print(df.shape)
        #l.append((df,f.split('/')[-1].replace('f0.','')))
        label_f = f.split('/')[-1].replace('f0.','')
        label = label_f.split('_')[0]
        labels = [label] * selected.shape[0]
        #l.append((selected,f.split('/')[-1].replace('f0.','')))
        if len(selected) > 0: 
            l.append((selected,labels))

    return l#train_csv


limit = 15
duration = 30
def select_harm(df):
    freq = df.frequency
    idxs= []
    #for i in range(freq.shape[0]-duration):
    wants = []
    i = 0
    while i < freq.shape[0] - duration:
        select = freq[i:i+duration]
        diff = np.diff(select)
        ans = abs(diff) < limit
        if ans.all():
            go_more = True
            j = i
            while go_more:
                j += 1
                if j == freq.shape[0]-1:
                    go_more = False
                #print('j:',j)
                select2 = freq[j:j+duration]
                diff2 = np.diff(select2)
                ans2 = abs(diff2) < limit
                if not ans2.all():
                    go_more = False
            start = i
            end = j - 1 + duration
            want = freq[start:end]
            wants.append(want)
            idxs.append(i)
            i = j - 1
            #print(i)
        i += 1
    #plt.figure()
    #my_plotas(freq[idxs])
    #for w in wants:
    #    my_plotas(w)
    selected = []
    for w in wants:
        w = np.pad(w,(0,700-len(w)))
        selected.append(w)
    selected = np.array(selected)
    return selected



