import glob
import librosa



import warnings
     
warnings.filterwarnings('ignore')



for name in glob.glob('../audio_input/*'):
    #print(name)
    y, sr = librosa.load(name)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    print('The tempo of ', name, ' is: ', tempo)

