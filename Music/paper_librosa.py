
import librosa
import librosa.display as disp

#filename = 'shbk.flac'
#heal = 'heal.flac'
#dx = 'dx.flac'
#dbz = 'dbz_et4.flac'

#y, sr= librosa.load(filename)
#y_heal, sr_heal = librosa.load('heal.flac')
#spectrogram = np.abs(librosa.stft(y))

def func_melspec(y,sr):
    res = librosa.feature.melspectrogram(y=y,sr=sr)
    return res

#melspec_shbk = melspec(y,sr)
#melspec_heal = melspec(y_heal,sr_heal)
#chroma = librosa.feature.chroma_cqt(y=y,sr=sr)
#tonnets = librosa.feature.tonnetz(y=y,sr=sr)



def display_melspectrogram(melspec,sr):
    plt.figure(figsize=(10, 4))
    S = melspec
    S_dB = librosa.power_to_db(S, ref=np.max)
    disp.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()

#to_display = input('song abbrev. you want to display melspectrogram: ')

#melspec_file = eval('melspec_' + to_display)
#display_melspectrogram(melspec_file)
#if to_display == 'shbk':
#    display_melspectrogram(melspec_shbk)
#if to_display == 'heal':
#    display_melspectrogram(melspec_heal)

def display_melspec_all_in_one():
    fname = input('song abbrev. you want to display melspectrogram: ')
    y, sr = librosa.load(fname)
    melspec = func_melspec(y,sr)
    display_melspectrogram(melspec,sr)
    
display_melspec_all_in_one()
