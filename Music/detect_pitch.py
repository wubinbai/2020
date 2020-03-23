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
    plt.figure()
    plt.plot(res)
    return res

p = get_pitches()
#plt.plot(p,'r*')

#To see what "pitches" is really returning, you want to do something more like:

#  plt.imshow(pitches[:100, :], aspect="auto", interpolation="nearest", origin="bottom")

#or perhaps:

#  plt.plot(np.tile(np.arange(pitches.shape[1]), [100, 1]).T, pitches[:100, :].T, '.')

#  plt.imshow(pitches[:100, :], aspect="auto", interpolation="nearest", origin="bottom")
