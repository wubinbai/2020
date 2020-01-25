import librosa

def extract_feats(f):
    y, sr = librosa.load(f, res_type='kaiser_fast')
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr,n_mfcc=40).T,axis=0)

    feat = mfcc
    label = LABEL
    return feat, label

