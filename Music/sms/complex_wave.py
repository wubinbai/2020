import numpy as np

N = 500
k = 3
n = np.arange(-N/2,N/2)
wave = np.exp(1j * n * k * 2 * np.pi  / N)
real = np.real(wave)
imag = np.imag(wave)
plt.plot(n,real)
plt.figure()
plt.plot(n,imag)

