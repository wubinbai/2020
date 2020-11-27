import librosa
import os
from tqdm import tqdm
#from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from extract_feats import *



def get_range_feat(arr):
    max_feat = np.max(arr)
    min_feat = np.min(arr)
    range_feat = max_feat - min_feat
    rn = 3
    max_feat = round(max_feat,rn)
    min_feat = round(min_feat,rn)
    range_feat = round(range_feat,rn)
    return (range_feat, min_feat, max_feat)


def my_get_rough_range_feat(my_y):

    # sort and "cut" n pieces, get rid of first and the last piece
    y_sort = np.sort(my_y)
    #my_y.sort() # sort inplace
    length = len(y_sort)
    n = 5
    l = length//n
    r = (n-1) * length//n
    arr = y_sort[l:r]
    max_feat = np.max(arr)
    min_feat = np.min(arr)
    range_feat = max_feat - min_feat
    rn = 3
    max_feat = round(max_feat,rn)
    min_feat = round(min_feat,rn)
    range_feat = round(range_feat,rn)
    return (range_feat, min_feat, max_feat)

def util_print_range_feats(feat_data, name = 'enter name'):
    a,b,c = get_range_feat(feat_data)
    disp_temp = name
    #print('{:<10}'.format(disp_temp),'[',b,',',c,']  = ', a)
    d,e,f = my_get_rough_range_feat(feat_data)
    #print('{:<10}'.format(disp_temp),'[',e,',',f,']  = ', d)
    mean_val = np.mean(feat_data)
    std_val = np.std(feat_data)
    return (a,b,c,d,e,f,mean_val,std_val)

def split_std(data,n_chunk=6):
    chunk_length = len(data)//n_chunk
    #chunks = librosa.util.frame(data,frame_length=chunk_length,hop_length=chunk_length).T
    #all_std = []
    #for chunk in chunks:
    #    all_std.append(np.std(chunk))
    #return np.mean(all_std)
    all_std = []
    for i in range(n_chunk):
        chunk_i = data[i*chunk_length:(i+1)*chunk_length]
        std_i = np.std(chunk_i)
        all_std.append(std_i)
    res = np.mean(all_std)
    return res




def train_and_eval(n_feats_now, to_save='n'):
    # save all feature data
    if to_save == 'y':
        save_data(n_feats_now)
    else:
        pass
    data = get_data()
    fname = data.iloc[:,0]
    # load only data of feature chosen
    Xy = data.iloc[:,1:] # drop name column
    # get X( only :n_feats_now) and y
    X = Xy.iloc[:,:n_feats_now]
    y = Xy.iloc[:,-1]
    # label encode y
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    y = pd.DataFrame(y)
    # fit data
    # stratified kfold
    sfolder = StratifiedKFold(n_splits=10,random_state=1,shuffle=False)
    ### build a classifier
    #rf = RandomForestClassifier(n_estimators=800)
    model = XGBClassifier()

    print('For debug: initiating model')
    # using stratifiedkfold 
    scores = []
    for train_index, test_index in sfolder.split(y,y):
        X_train = X.loc[train_index]
        X_test = X.loc[test_index]
        y_train = y.loc[train_index]
        y_test = y.loc[test_index]
        #print('shape of X_train: ',X_train.shape)
        #print('shape of y_train: ',y_train.shape)
        #print(y_train.head())
        #rf.fit(X_train,y_train)
        model.fit(X_train,y_train)

        #preds = rf.predict(X_test)
        preds = model.predict(X_test)

        for i in range(0,250,5):
            j = i+5
            print(preds[i:j],fname.loc[i:j])

        all_correct = 0
        for i in range(int(preds.shape[0]/5)):
            ans = i
            temp = preds[i*5:i*5+5]
            correct = (temp == ans)
            correct = correct.sum()
            all_correct += correct

        score = all_correct / preds.shape[0]
        scores.append(score)

    #plt.figure()
    #plt.plot(rf.feature_importances_)
    #plt.plot(rf.feature_importances_,'r*')
    #plt.savefig('feat_imp.png')
    # get score
    score = round(np.mean(scores),4)
    print('cross_val score: ', score)
    #### finish stratified kfold
    return score

def save_data(n_feats_now):
    #use ls dir_name > ../files.txt to generate txt, manually add first row as Acoustic....xx
    genres_df = pd.read_csv('files.txt')
    #genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

    #tup = tuple('feat'+str(x) for x in range(n_feats_now))
    #df_all_feats = pd.DataFrame(columns=tup)
    # initialize array of dim 1000 by n_feats_now, then to df
    print('edit NUM_FUNCTIONS = every time you increase a function')
    NUM_FUNCTIONS = 1
    array_all_feats = np.zeros((1000,NUM_FUNCTIONS*10+2))
    df = pd.DataFrame(array_all_feats)
    # use count for indexing df
    count = 0
    for g in tqdm(genres_df.values):
        g = g[0]
        for filename in os.listdir(f'./processed_50_50/{g}'):
            songname = f'./processed_50_50/{g}/{filename}'
            print('processing songname: ', songname)
            temp_feats = get_feats(songname)
            # duration = 30 s
            #y, sr = librosa.load(songname, mono=True, duration=30, res_type='kaiser_fast')
            feats = []
            feats.append(songname)
            #for i in range(n_feats_now):
            #for fun in range(1,NUM_FUNCTIONS+1): 
            #    # for debugging which func
            #    print('fun = ', fun)
            #    temp_feat = eval('f'+str(fun))(y)
            #    for temp in temp_feat:
            #        feats.append(temp)
            for k in temp_feats:
                feats.append(k)
            feats.append(g)
            print(feats)
            df.loc[count,:] = feats
            count+=1
    # dump all feature and label data to a csv file for use
    df.to_csv('saved_data_harm_ratio.csv',index=False)

def get_data():
    df = pd.read_csv('saved_data_harm_ratio.csv')
    print(df.shape)
    return df

def f1(y):
    '''
    hpss
    n_feats = 6
    '''
    y_h, y_p = librosa.effects.hpss(y)
    yhrf = get_range_feat(y_h)
    yprf = get_range_feat(y_p)
    return_feats = (yhrf[0],yhrf[1],yhrf[2],yprf[0],yprf[1],yprf[2])
    return return_feats
    #return util_print_range_feats(y_,'contrast')


def f2(y):
    '''
    rms
    n_feats = 8
    '''
    ### see if hpss work ###
    y_h, y_p = librosa.effects.hpss(y)
    y = y_p
    ### end of hpss ###


    S, phase = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rms(S=S)
    #rmsrf = get_range_feat(rms[0])
    #rmsrrf = my_get_rough_range_feat(rms[0])
    #bundle = (rmsrf, rmsrrf)
    #return (bundle[0][0],bundle[0][1],bundle[0][2],bundle[1][0],bundle[1][1],bundle[1][2])
    return util_print_range_feats(rms[0],'rms')



def f3(y):
    '''
    zcr
    n_feats = 8
    '''
    ### see if hpss work ###
    y_h, y_p = librosa.effects.hpss(y)
    y = y_p
    ### end of hpss ###

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    return util_print_range_feats(zcr,name='zcr')
def f4(y):
    '''
    cent
    n_feats = 8
    '''
    ### see if hpss work ###
    y_h, y_p = librosa.effects.hpss(y)
    y = y_p
    ### end of hpss ###

    S, phase = librosa.magphase(librosa.stft(y=y))
    cent = librosa.feature.spectral_centroid(S=S)[0]
    return util_print_range_feats(cent,name='cent')

 
def f5(y):
    '''
    spec_bw
    n_feats = 8
    '''
    ### see if hpss work ###
    y_h, y_p = librosa.effects.hpss(y)
    y = y_p
    ### end of hpss ###

    S, phase = librosa.magphase(librosa.stft(y=y))
    spec_bw = librosa.feature.spectral_bandwidth(S=S)
    return util_print_range_feats(spec_bw[0],'spec_bw')


def f6(y,sr=22050):
    '''
    contrast
    n_feats = 8
    '''
    ### see if hpss work ###
    y_h, y_p = librosa.effects.hpss(y)
    y = y_p
    ### end of hpss ###

    S = np.abs(librosa.stft(y))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    return util_print_range_feats(contrast,'contrast')


 
def f7(y,sr=22050):
    '''
    rolloff
    n_feats = 8
    '''
    ### see if hpss work ###
    y_h, y_p = librosa.effects.hpss(y)
    y = y_p
    ### end of hpss ###

    # note: 'Approximate maximum frequencies with roll_percent=0.85 (default) rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)'
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.1)
    S, phase = librosa.magphase(librosa.stft(y))
    return util_print_range_feats(rolloff[0],'rolloff')


 
def f8(y, sr=22050):
    '''
    melspectrogram
    n_feats = 8
    '''
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return util_print_range_feats(S_dB,'melspec')
    #return (S_dB.std(),S_dB.max(),S_dB.min(),S_dB.mean(),0,0,0,0)
 
def f9(y,sr=22050):
    '''
    mfcc
    n_feats = 8
    '''
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S)) 
    # toy feat.
    #feats = (mfccs.min(),mfccs.max(),mfccs.mean(),mfccs.std(),split_std(mfccs),split_std(mfccs,n_chunk=12),split_std(mfccs,n_chunk=18),1)
    #return feats

    return util_print_range_feats(mfccs,'mfccs')
 
def f10(y,sr=22050):
    '''
    chroma_stft
    n_feats = 8
    '''
    S = np.abs(librosa.stft(y, n_fft=4096))**2
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    feats = (chroma.min(),chroma.max(),chroma.mean(),chroma.std(),split_std(chroma),split_std(chroma),split_std(chroma),split_std(chroma))
    return feats
    
 
def f11(y):
    '''
    tempo
    n_feats = 8
    '''
    sr = 22050
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    # Compute global onset autocorrelation
    ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
    ac_global = librosa.util.normalize(ac_global)

    tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length)[0]
    feats = (tempogram.std(),tempogram.mean(),tempogram.max(),tempogram.min(),tempo,0,0,0)
    return feats
def f12(y):
    '''
    chroma_cqt
    n_feats = 8
    '''
    chroma_cq = librosa.feature.chroma_cqt(y)
    feats = (chroma_cq.min(),chroma_cq.max(),chroma_cq.mean(),chroma_cq.std(),1,1,1,1)
    return feats
    
 
def f13(y):
    '''
    chroma_cens
    n_feats = 8
    '''
    chroma_cens = librosa.feature.chroma_cens(y=y)
    feats = (chroma_cens.min(),chroma_cens.max(),chroma_cens.mean(),chroma_cens.std(),1,1,1,1)
    return feats
def f14(y):
    '''
    tonnetz
    n_feats = 8
    '''
    tonnetz = librosa.feature.tonnetz(y=y)
    feats = (tonnetz.min(),tonnetz.max(),tonnetz.mean(),tonnetz.std(),1,1,1,1)
    return feats
def f15(y):
    '''
    poly_feats_S
    n_feats = 8
    '''
    S = np.abs(librosa.stft(y))
    p0 = librosa.feature.poly_features(S=S,order=0)
    p1 = librosa.feature.poly_features(S=S,order=1)
    p2 = librosa.feature.poly_features(S=S,order=2)
    # these feats are just toy feats
    feats = (p0.mean(),p1.mean(),p2.mean(),1,1,1,1,1)
    return feats
def f16(y,sr=22050):
    """
    delta_mfcc
    n_feats = 8
    """
    mfcc = librosa.feature.mfcc(y=y,sr=sr)
    mfcc_delta = librosa.feature.delta(mfcc)
    return util_print_range_feats(mfcc_delta,'mfcc_delta')


def f17(y,sr=22050):
    '''
    delta_melspec
    n_feats = 8
    '''
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB_delta = librosa.feature.delta(S_dB)
    return util_print_range_feats(S_dB_delta,'S_dB_delta')



def f18(y,sr=22050):
    '''
    stack_memory
    n_feats = 8
    '''

    return (1,1,1,1,1,1,1,1)

def f19(y,sr=22050):
    ### see if hpss work ###
    y_h, y_p = librosa.effects.hpss(y)
    y = y_p
    ### end of hpss ###


    oenv = librosa.onset.onset_detect(y,sr)
    #print('y: ',y)
    #print('oenv: ', oenv)
    n_oenv = len(oenv)

    #initiate
    class_oenv = 3
    dframe = 0
    dframe_std = 0
    dframe_mean = 0
    dframe_min = 0
    dframe_max = 0
    if n_oenv >= 2:
        if n_oenv <=5:
            class_oenv = 0
        elif 5 < n_oenv <= 10:
            class_oenv = 1
        elif 10 < n_oenv:
            class_oenv = 2
        else:
            raise RuntimeError
        dframe = np.diff(oenv)
        print('dframe',dframe)
        dframe_std = np.std(dframe)
        dframe_mean = np.mean(dframe)
        dframe_min = np.min(dframe)
        dframe_max = np.max(dframe)
    res = (n_oenv, class_oenv, 1, dframe_std, dframe_mean, dframe_min, dframe_max, 1)
    assert len(res) == 8
    return res

def f20(y,sr=22050):
    '''
    range oenv
    n_feats = 8
    '''
    ### see if hpss work ###
    y_h, y_p = librosa.effects.hpss(y)
    y = y_p
    ### end of hpss ###


    oenv = librosa.onset.onset_detect(y,sr)
    n_oenv = len(oenv)

    if n_oenv >= 2:
        dframe = np.diff(oenv)
        try:
            res = util_print_range_feats(dframe)
        except:
            res = (1,1,1,1,1,1,1,1)

        #print(len(res))
        return res
    else:
        return (1,1,1,1,1,1,1,1)


def run_num_feats(n_feats,tosave):
    #for i in range(1,n_feats+1):
        n_feats_now = n_feats
        print('running model with ', n_feats_now, 'feats')
        score = train_and_eval(n_feats_now,to_save=tosave)
        print('the score for ', n_feats_now, 'is', score)
        return score




#####

#####
if __name__ == '__main__':
    #n_feats = 2
    #run_num_feats(n_feats)
    #save_data(2)
    #get_data()
    all_scores = []
    start_feat = int(input('Enter starting num feats to run: '))
    num = int(input('Enter num feats to run: '))
    first_time = 0
    for i in range(start_feat,num+1):
        if first_time !=0:
            my_save = 'n'
        if first_time ==0:
            my_save = input('Enter y or n for save: ')
        first_time+=1
        score = run_num_feats(i,tosave=my_save)
        all_scores.append(score)
    #pass
