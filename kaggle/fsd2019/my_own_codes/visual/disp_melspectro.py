import librosa
import librosa.display as disp

def func_melspec(y,sr):
    res = librosa.feature.melspectrogram(y=y,sr=sr)
    return res

def display_melspectrogram(melspec,sr):
    plt.figure(figsize=(10, 4))
    S = melspec
    S_dB = librosa.power_to_db(S, ref=np.max)
    disp.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()

# to be edited
def save_melspectrogram(melspec,sr,destination):
    plt.figure(figsize=(10, 4))
    S = melspec
    S_dB = librosa.power_to_db(S, ref=np.max)
    disp.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()


def display_melspec_all_in_one(f):
    fname = f #= input('song abbrev. you want to display melspectrogram: ')
    y, sr = librosa.load(fname)
    melspec = func_melspec(y,sr)
    display_melspectrogram(melspec,sr)
    
#display_melspec_all_in_one()
