import librosa
import librosa.display as disp

# take care of hop_length
def plot_stft(f,hop_length=512):
    y,sr = librosa.load(f)
    stft = librosa.stft(y,hop_length=hop_length)
    D = np.abs(stft)
    db = librosa.amplitude_to_db(D,ref=np.max)
    plt.figure()
    disp.specshow(db,x_axis='time',y_axis='log')

f = 'A4.wav'
plot_stft(f,512*8)

