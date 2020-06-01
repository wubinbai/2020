from util_packet import get_packet
import librosa
from librosa import display as disp
f = '/home/wb/Downloads/A4h.wav'
y,sr = librosa.load(f)
l=12352
#y = y[4*22050-200:4*22050+200]
plt.figure()
#disp.waveplot(y,x_axis='time')
plt.plot(y,'ok')
upper = np.zeros_like(y)
upper[:100] = np.max(y[:100])
upper[100:200] = np.max(y[100:200])
upper[200:300] = np.max(y[200:300])
upper[300:400] = np.max(y[300:400])
import peakutils
ind = peakutils.indexes(y,min_dist=90)
x = list(range(len(y)))
x = np.array(x)
plt.plot(x[ind],y[ind])
#plt.vlines(ind,0,max(y))
#lower = np.zeros_like(y)


#a,b = get_packet(y) 
