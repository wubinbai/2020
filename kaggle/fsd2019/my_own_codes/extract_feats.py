import librosa
from scipy.fft import fft
import numpy as np
import peakutils
from sklearn.preprocessing import MinMaxScaler
temp_path = '/home/wb/2020/Music/sms/workspace'
import sys
sys.path.append(temp_path)
from f0_util import *


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
    # disable h_h
    y = y_h
    
    #rms = librosa.feature.rms(y)[0]
    #frame_n = np.argmax(rms)
    # CAUTION: BE CAREFUL OF CHOOSING FRAME LOCATION TO ANALYSE: THIS MAY AFFECT RESULTS SIGNIFICANTLY!
    #frame_length = 2048*2
    #if frame_n <=4:
    #    y = y[:frame_length]
    #else:
    #    y = y[frame_n*512-frame_length//2:frame_n*512+frame_length//2]
    pitches, magnitudes = librosa.pitch.piptrack(y)
    estimated_pitch = pick_pitch(pitches, magnitudes)

    

    # fft using scipy
    ft = fft(y)
    mag_ft_b = np.absolute(ft)
    xs_b = np.linspace(0,sr,len(mag_ft_b))
    choose = xs_b.shape[0]//2#200 #5000
    xs = xs_b[:choose]
    mag_ft = mag_ft_b[:choose]
    # play round fft of scipy EFFICIENTLY!
    # get estimate pitch and set min_dist

    def my_get_f0(xs,mag_ft,estimated_pitch):
        interval = 1/(xs[1]-xs[0])
        MIN_DIST = interval * estimated_pitch #10.37 is intervals of frequencies b/t xs
        MIN_DIST = int(MIN_DIST) - 1  # lowest integer
        MIN_DIST2 = MIN_DIST//2 - 1 #-1 # CONSIDERING LOWER 1ST FUNDAMENTAL FREQ
        # ACTUALLY, MIN_DIST shouldn't be a fixed value, since as frequency increases, the distance between each consecutive harmonic frequency is becoming larger and larger delta f = fx * 1.0594

        ind = peakutils.indexes(mag_ft, thres=0.01, min_dist=MIN_DIST)
        ind2 = peakutils.indexes(mag_ft, thres=0.01, min_dist=MIN_DIST2)

        if len(ind2) > len(ind):
            ind = ind2
        if ind.shape[0] == 0:
            ind = np.array([0])
        fundamental_frequency = estimated_pitch
        peak_frequency = xs[ind]
        peak_magnitude = mag_ft[ind]
        return fundamental_frequency, peak_frequency, peak_magnitude
  
    fundamental_frequency, peak_frequency, peak_magnitude = my_get_f0(xs,mag_ft,estimated_pitch)

    def my_modify_f0(fundamental_frequency,peak_frequency,peak_magnitude):
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
        return fundamental_frequency
    ###
    #fundamental_frequency = np.mean(util_get_f0_from_audio(y)) 
    #fundamental_frequency = my_modify_f0(fundamental_frequency,peak_frequency,peak_magnitude)

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
    if feats[1][1] != 0:
        ratio = feats[1]/feats[1][0]
    else:
        denominator = feats[1][(feats[1]!=0).argmax()]
        ratio = feats[1]/denominator
    
    # nan to 0 inf to 0
    ratio[np.isnan(ratio)] = 0
    ratio[np.isinf(ratio)] = 0

    return ratio

def select_best_ans(anss):
    '''
    return best ans
    '''
    orig_shape=0
    index = 0
    for i in range(len(anss)):
        temp_shape=np.nonzero(anss[i])[0].shape[0]
        if temp_shape > orig_shape:
            orig_shape = temp_shape
            index = i
    return anss[index]


def get_feats(fname):
    # load
    print('!!!! filename',fname)
    y, sr = librosa.load(fname)
    # but first ... make sure the audio_length is larger than, say 2048
    audio_length = y.shape[0]
    
    # constants
    num_pins = 10
    left = 1024
    right = 1024
    left2 = left*2
    segments = []
    
    if audio_length >= (left)*(num_pins + 1):
        pin_distance = int(audio_length/(num_pins+1))
        pins = [pin_distance * (m+1) for m in list(range(10))]
        for pin in pins:
            segment = y[pin-left:pin+right]
            segments.append(segment)
    elif (left*2) <= audio_length < (left)*(num_pins + 1):
        num_pins = int(np.floor(audio_length/(left*2)))
        for i in range(num_pins):
            segment = y[left2*i:left2*(i+1)]
            segments.append(segment)
    else:
        num_pins = 1 
        assert 0 < audio_length < left*2
        segments.append(y[:])
    
    feats_list = []
    for i in range(num_pins):
        two_feats = extract_feats(segments[i])
        ratio = feats_to_ratio(two_feats)
        feats_list.append(ratio)
    # extract feats
    feats = extract_feats(y)
    ratio = feats_to_ratio(feats)
    feats = ratio
    # return feats
    return feats
