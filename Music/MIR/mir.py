import librosa
hop_length = 256
frame_length = 512
import librosa.display as disp

x, sr = librosa.load('audio/simple_loop.wav')
y = x
energy = np.array([sum((x[i:i+frame_length]**2)) for i in range(0,len(y),hop_length)])
energy_n = energy/energy.max()
frames = range(len(energy))
t = librosa.frames_to_time(frames,sr,hop_length=hop_length)

def strip(x, frame_length, hop_length):

    # Compute RMSE.
    rms = librosa.feature.rms(x, frame_length=frame_length, hop_length=hop_length, center=True)
    
    # Identify the first frame index where RMSE exceeds a threshold.
    thresh = 0.01
    frame_index = 0
    while rms[0][frame_index] < thresh:
        frame_index += 1
        
    # Convert units of frames to samples.
    start_sample_index = librosa.frames_to_samples(frame_index, hop_length=hop_length)
    
    # Return the trimmed signal.
    return x[start_sample_index:]

#y = strip(x, frame_length, hop_length)
X = librosa.stft(y)
Xdb = librosa.amplitude_to_db(abs(X))
#plt.ion()
#disp.specshow(Xdb,sr=sr,x_axis='time',y_axis='hz')
#plt.figure()
#plt.show()

rms = librosa.feature.rms(y,hop_length=hop_length)
rms = rms[0]
rms_n = rms/rms.max()

plt.figure()
disp.waveplot(y,sr)
plt.plot(t,energy_n,c='g')
plt.plot(t,rms_n,c='b')
plt.savefig('wave_E_RMS.png')
