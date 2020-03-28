import librosa
from scipy.fft import fft
import numpy as np
import peakutils
from sklearn.preprocessing import MinMaxScaler

###
def harmonic_detection(fundamental_frequency,peak_frequency,peak_magnitude,num_harmonic=10,sr=22050,threshold_ratio=0.09):
    print('fundamental_frequency is: ', fundamental_frequency)
    #assert fundamental_frequency > 0
    if fundamental_frequency <=0:
        fundamental_frequency = 1.0
    harmonic_frequency = np.zeros(num_harmonic)
    harmonic_magnitude = np.zeros(num_harmonic)
    harmonic_frequency_true = fundamental_frequency * np.arange(1,num_harmonic+1)
    harmonic_index = 0
    while harmonic_index < num_harmonic and harmonic_frequency_true[harmonic_index] < sr/2:
        peak_index = np.argmin(abs(peak_frequency - harmonic_frequency_true[harmonic_index]))
        deviation = abs(peak_frequency[peak_index] - harmonic_frequency_true[harmonic_index])
        threshold = threshold_ratio * fundamental_frequency
        if deviation < threshold:
            harmonic_frequency[harmonic_index] = peak_frequency[peak_index]
            harmonic_magnitude[harmonic_index] = peak_magnitude[peak_index]
        harmonic_index += 1

    return harmonic_frequency, harmonic_magnitude
###



def pick_pitch(pitches, magnitudes):
    pitches = pitches.T
    magnitudes = magnitudes.T

    # maximum magnitude as pitch # ANOTHER  WAY : MAY BE MORE ROBUST: WEIGHTED AVERAGE OF TOP MAGNITUDES FOR EACH FRAME!
    indr = np.argwhere(magnitudes==np.max(magnitudes))[0][0]
    indc = np.argwhere(magnitudes==np.max(magnitudes))[0][1]
    estimated_pitch = pitches[indr,indc]
    picked_pitch = estimated_pitch
    return picked_pitch

def extract_feats(y,sr=22020):
    y_h, y_p = librosa.effects.hpss(y)
    y = y_h
    
    rms = librosa.feature.rms(y)[0]
    frame_n = np.argmax(rms)
    # CAUTION: BE CAREFUL OF CHOOSING FRAME LOCATION TO ANALYSE: THIS MAY AFFECT RESULTS SIGNIFICANTLY!
    frame_length = 2048*2
    if frame_n <=4:
        y = y[:frame_length]
    else:
        y = y[frame_n*512-frame_length//2:frame_n*512+frame_length//2]
    pitches, magnitudes = librosa.pitch.piptrack(y)
    estimated_pitch = pick_pitch(pitches, magnitudes)

    

    # fft using scipy
    ft = fft(y)
    mag_ft_b = np.absolute(ft)
    xs_b = np.linspace(0,sr,len(mag_ft_b))
    choose = 200 #5000
    xs = xs_b[:choose*8]
    mag_ft = mag_ft_b[:choose*8]
    # play round fft of scipy EFFICIENTLY!
    # get estimate pitch and set min_dist
    interval = 1/(xs[1]-xs[0])
    MIN_DIST = interval * estimated_pitch #10.37 is intervals of frequencies b/t xs
    MIN_DIST = int(MIN_DIST) - 1  # lowest integer
    MIN_DIST2 = MIN_DIST//2 - 1 #-1 # CONSIDERING LOWER 1ST FUNDAMENTAL FREQ
    # ACTUALLY, MIN_DIST shouldn't be a fixed value, since as frequency increases, the distance between each consecutive harmonic frequency is becoming larger and larger delta f = fx * 1.0594

    ind = peakutils.indexes(mag_ft, thres=0.01, min_dist=MIN_DIST)
    ind2 = peakutils.indexes(mag_ft, thres=0.01, min_dist=MIN_DIST2)

    if len(ind2) > len(ind):
        ind = ind2

    fundamental_frequency = estimated_pitch
    peak_frequency = xs[ind]
    peak_magnitude = mag_ft[ind]



    ### largest_divide = 1
    fundamental_frequencies = [fundamental_frequency/k for k in range(1,5)] # / 1,2,3,4
    largest_divide = 1
    h = 1
    harmonic_frequency, harmonic_magnitude = harmonic_detection(fundamental_frequency=fundamental_frequencies[h-1], peak_frequency=peak_frequency,peak_magnitude=peak_magnitude)
    non_zero_num = len(np.nonzero(harmonic_frequency)[0])
    reference = non_zero_num
    print('reference: ', reference)
    print('estimated pitch', estimated_pitch)
    for h in range(2,5):
        harmonic_frequency, harmonic_magnitude = harmonic_detection(fundamental_frequency=fundamental_frequencies[h-1], peak_frequency=peak_frequency,peak_magnitude=peak_magnitude)
        non_zero_num = len(np.nonzero(harmonic_frequency)[0])
        print('lower freq scores ', non_zero_num)
        if non_zero_num > reference:
            largest_divide = h
            reference = non_zero_num
        #print(non_zero_num)
    #print('largest_divide = ', largest_divide)
    fundamental_frequency = fundamental_frequency / largest_divide
    #print(fundamental_frequency)

    
    ###
    harmonic_frequency, harmonic_magnitude = harmonic_detection(fundamental_frequency=fundamental_frequency, peak_frequency=peak_frequency,peak_magnitude=peak_magnitude)
    ###
    print('fundamental_frequency: ',fundamental_frequency)
    print('peak_frequency: ', peak_frequency)
    print('peak_magnitude: ', peak_magnitude)
    #print('debugging index: ', ind)
    #print('xs[ind',xs[ind])
    #print('estimated pi',estimated_pitch)
    
    ###
    print('harm freq', harmonic_frequency)
    print('harm mag',harmonic_magnitude)
    feats = harmonic_frequency, harmonic_magnitude


    return feats

def feats_to_ratio(feats):
    try:
        ratio = feats[1]/feats[1][0]
    except:
        mms = MinMaxScaler
        ratio = mms.fit_transform(feats[1])
    return ratio

def get_feats(fname):
    # load
    y, sr = librosa.load(fname)
    # extract feats
    feats = extract_feats(y)
    ratio = feats_to_ratio(feats)
    feats = ratio
    # return feats
    return feats
