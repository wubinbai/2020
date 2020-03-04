import librosa

def get_zero_crossings():
    f = input('Enter file abs path: ')
    y,sr = librosa.load(f)
    zc_list = librosa.zero_crossings(y)
    zc = zc_list.sum()
    zcr_arr = librosa.feature.zero_crossing_rate(y,sr)[0]
    zcr = zcr_arr.mean()
    print('zero crossing is: ', zc)
    print('zero crossing rate is: ', zcr)
    return zc,zc_list, zcr, zcr_arr
zc, zc_list, zcr, zcr_arr = get_zero_crossings()
