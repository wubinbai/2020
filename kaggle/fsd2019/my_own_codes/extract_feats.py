import librosa
from scipy.fft import fft



def pick_pitch(pitches, magnitudes):

    return picked_pitch
def extract_feats(y):
    rms = librosa.feature.rms(y)[0]
    frame_n = np.argmax(rms)
    # CAUTION: BE CAREFUL OF CHOOSING FRAME LOCATION TO ANALYSE: THIS MAY AFFECT RESULTS SIGNIFICANTLY!
    frame_length = 2048*2
    if frame_n <=4:
        y = y[:frame_length]
    else:
        y = y[frame_n*512-frame_length//2:frame_n*512+frame_length//2]
    pitches, magnitudes = librosa.pitch.piptrack(y)
    picked_pitch = pick_pitch(pitches, magnitudes)
    # fft using scipy
    ft = fft(y)
    mag_ft = np.absolute(ft)
    # play round fft of scipy EFFICIENTLY!

    return feats

def get_feats(fname):
    # load
    y, sr = librosa.load(fname)
    # extract feats
    feats = extract_feats(y)
    # return feats
    return feats
