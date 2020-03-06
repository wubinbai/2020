import librosa
import librosa.display as disp
import sklearn

y,sr = librosa.load('hx.mp3')
y = y[:len(y)]
x = y

sc = spectral_centroids = librosa.feature.spectral_centroid(y+0.01, sr=sr)[0]
spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr,roll_percent=0.85)[0]

frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)

def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

plt.figure()
disp.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r') # normalize for visualization purposes
plt.plot(t, normalize(spectral_rolloff), color='g')

