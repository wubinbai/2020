import struct
from scipy.io import wavfile as wav



def get_bit_depth(path):
    """
    Parameters:
    path: absolute path of audio file
    """
    wave_file = open(path,"rb")
    riff_fmt = wave_file.read(36)
    bit_depth_string = riff_fmt[-2:]
    bit_depth = struct.unpack("H",bit_depth_string)[0]
    return bit_depth


