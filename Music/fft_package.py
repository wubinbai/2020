from scipy.fft import fft
import librosa

def audio_to_fft(audio,sr):
    '''
    inputs: audio, sr
    audio: audio seq
    sr: sampling rate
    returns: freqs, mag_freq
    '''
    ft = fft(audio)
    mag_ft_b = np.absolute(ft)
    xs_b = np.linspace(0,sr,len(mag_ft_b))
    half = len(xs_b)//2
    xs = xs_b[:half]
    mag_ft = mag_ft_b[:half]

    freqs = xs
    mag_freq = mag_ft
    return freqs, mag_freq
def file_to_fft(filename):
    audio, sr = librosa.load(filename)
    freqs, mag_freq = audio_to_fft(audio,sr)
    return freqs, mag_freq

