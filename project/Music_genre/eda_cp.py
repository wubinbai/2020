import os
import shutil

os.mkdir('eda')
os.chdir('./eda')
os.mkdir('blues')
os.mkdir('classical')
os.mkdir('country')
os.mkdir('disco')
os.mkdir('hiphop')
os.mkdir('jazz')
os.mkdir('metal')
os.mkdir('pop')
os.mkdir('reggae')
os.mkdir('rock')


shutil.copy('../MIR/genres/blues/blues.00000.wav','blues')
shutil.copy('../MIR/genres/classical/classical.00000.wav','classical')
shutil.copy('../MIR/genres/country/country.00000.wav','country')
shutil.copy('../MIR/genres/disco/disco.00000.wav','disco')
shutil.copy('../MIR/genres/hiphop/hiphop.00000.wav','hiphop')
shutil.copy('../MIR/genres/jazz/jazz.00000.wav','jzzz')
shutil.copy('../MIR/genres/metal/metal.00000.wav','metal')
shutil.copy('../MIR/genres/pop/pop.00000.wav','pop')
shutil.copy('../MIR/genres/reggae/reggae.00000.wav','reggae')
shutil.copy('../MIR/genres/rock/rock.00000.wav','rock')
