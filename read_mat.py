import numpy as np
import matplotlib.pyplot as plt

fsk_cap = np.fromfile(open("fsk_capture.mat"), dtype=np.complex64)
fsamp = 400e3
Tsamp = 1 / fsamp
datarate = 50e3

time_fsk = np.linspace(
    start=0, stop=len(fsk_cap) * Tsamp, num=len(fsk_cap), endpoint=True
)

plt.plot(time_fsk, np.abs(fsk_cap))
plt.show()