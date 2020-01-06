
import librosa
y, sr = librosa.load('../shbk.flac',sr=44100)
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
log_S = librosa.power_to_db(S, ref=np.max)





y_harmonic, y_percussive = librosa.effects.hpss(y)

# Now, let's run the beat tracker.
# We'll use the percussive component for this part
plt.figure(figsize=(12, 6))
tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)
# Let's re-draw the spectrogram, but this time, overlay the detected beats
plt.figure(figsize=(12,4))
import librosa.display
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
# Let's draw transparent lines over the beat frames
plt.vlines(librosa.frames_to_time(beats),
1, 0.5 * sr,
colors='w', linestyles='-', linewidth=2, alpha=0.5)
plt.axis('tight')
plt.colorbar(format='%+02.0f dB')
plt.tight_layout()


print('Estimated tempo:%.2f BPM' % tempo)
print('First 5 beat frames:', beats[:5])
#Frame numbers are great and all, but when do those beats occur?
print('First 5 beat times:', librosa.frames_to_time(beats[:5], sr=sr))

