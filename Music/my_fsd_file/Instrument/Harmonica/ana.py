import librosa
import librosa.display as disp

#f = '95f92a43.wav'
f = librosa.util.find_files('.')[0]
y,sr = librosa.load(f)
y_h, y_p = librosa.effects.hpss(y)
plt.figure()
plt.subplot(2,1,1)
disp.waveplot(y_h)
plt.subplot(2,1,2)
disp.waveplot(y_p)
plt.tight_layout()
plt.savefig('hpss.png')
rms = librosa.feature.rms(y)[0]
plt.figure()
plt.plot(rms)
plt.savefig('rms.png')
plt.figure()
disp.waveplot(y)
max_rms = np.argmax(rms)
locate = 512 * max_rms
plt.vlines(locate,0,max(y))
plt.savefig('wave.png')
zcr = librosa.feature.zero_crossing_rate(y)[0]
plt.figure()
plt.plot(zcr)
plt.savefig('zcr.png')
