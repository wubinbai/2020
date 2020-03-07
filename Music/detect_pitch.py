import librosa
f = input('fname is: ')
y,sr = librosa.load(f)
pitches, magnitudes = librosa.piptrack(y,sr)
def detect_pitch(y, sr, t):
  index = magnitudes[:, t].argmax()
  pitch = pitches[index, t]

  return pitch


def get_pitches():
    res = []
    tl = pitches.shape[1]
    for t in range(tl):
        pitch = detect_pitch(y,sr,t)
        res.append(pitch)
    return res


