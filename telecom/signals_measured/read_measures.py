import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('received_signal5.txt', dtype='complex')

f_s = 50e3
osr = 8

fft = np.fft.fft(data)
fft_freq = np.fft.fftfreq(len(data), 1 / (osr * f_s))

plt.plot(fft_freq, np.abs(fft))
plt.show()