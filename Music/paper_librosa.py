import librosa
filename = 'shbk.flac'

y, sr= librosa.load(filename, duration=20.0)

spectrogram = np.abs(librosa.stft(y))
melspec = librosa.feature.melspectrogram(y=y, sr=sr)

chroma = librosa.feature.chroma_cqt(y=y,sr=sr)
tonnets = librosa.feature.tonnetz(y=y,sr=sr)

