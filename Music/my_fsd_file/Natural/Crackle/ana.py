import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display as disp

#f = '95f92a43.wav'
f = librosa.util.find_files('.')[0]
dir_name = f[:-12] + '/'
y,sr = librosa.load(f)
y_h, y_p = librosa.effects.hpss(y)
plt.figure()
plt.subplot(2,1,1)
disp.waveplot(y_h)
plt.subplot(2,1,2)
disp.waveplot(y_p)
plt.tight_layout()
plt.savefig(dir_name + 'hpss.png')
rms = librosa.feature.rms(y)[0]
rms_h = librosa.feature.rms(y_h)[0]
rms_p = librosa.feature.rms(y_p)[0]

plt.figure()
plt.subplot(2,1,1)
plt.plot(rms_h)
plt.subplot(2,1,2)
plt.plot(rms_p)
plt.savefig(dir_name + 'hpss_rms.png')

plt.figure()
plt.plot(rms)
plt.savefig(dir_name + 'rms.png')
plt.figure()
disp.waveplot(y)
max_rms = np.argmax(rms)
locate = 512 * max_rms
plt.vlines(locate,0,max(y))
plt.savefig(dir_name + 'wave.png')
zcr = librosa.feature.zero_crossing_rate(y)[0]
plt.figure()
plt.plot(zcr)
plt.savefig(dir_name + 'zcr.png')
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)
plt.figure()
disp.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()
plt.savefig(dir_name + 'mel.png')

