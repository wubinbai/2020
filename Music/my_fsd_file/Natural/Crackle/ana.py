import librosa
import librosa.display as disp

#f = '95f92a43.wav'
f = librosa.util.find_files('.')[0]
y,sr = librosa.load(f)
plt.figure()
disp.waveplot(y)
plt.savefig('wave.png')
rms = librosa.feature.rms(y)[0]
plt.figure()
plt.plot(rms)
plt.savefig('rms.png')
zcr = librosa.feature.zero_crossing_rate(y)[0]
plt.figure()
plt.plot(zcr)
plt.savefig('zcr.png')
