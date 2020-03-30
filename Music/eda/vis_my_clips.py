import librosa
import os
import librosa.display as disp

fs = os.listdir('./my_clips')
print(fs)

for f in fs:
    print(f)
    y,sr = librosa.load('./my_clips/' + f)
    rms = librosa.feature.rms(y)[0]
    locate = np.argmax(rms)
    frame = locate * 512
    t = frame / sr

    plt.figure()
    plt.subplot(2,1,1)
    plt.vlines(t,0,1)
    disp.waveplot(y,x_axis='time')
    
    plt.subplot(2,1,2)
    if frame != 0:
        disp.waveplot(y[frame-512:frame+512])
    else:
        disp.waveplot(y[frame:frame+1024])
    plt.title(f)
    plt.tight_layout()
    plt.savefig('./output/'+ f+'.png')
    plt.figure()
    if frame <= 2:
        y = y[:4096]
    else:
        y = y[frame-2048:frame+2048]
    S = librosa.feature.melspectrogram(y)
    S_dB = librosa.power_to_db(S,ref=np.max)
    print(S_dB.shape)
    disp.specshow(S_dB, x_axis='time',y_axis='mel')
    plt.title(f)
    plt.tight_layout()
    plt.savefig('./output/'+ f+'_mel.png')
