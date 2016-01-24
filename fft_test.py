import numpy as np
import matplotlib.pyplot as plt

Fs = 2000            # Sampling frequency
T = 1.0/Fs             # Sampling period
t = np.arange(1000)*T        # Time vector

S = 0.7*np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t);

sp = np.fft.fft(S)
freqs = np.fft.fftfreq(S.size)
print(freqs.min(), freqs.max())

idx = np.argmax(np.abs(sp))
print idx
freq = freqs[idx]
freq_in_hertz = abs(freq * 1000)

# freq_in_hertz = abs(freqs[12] * Fs)
print freq_in_hertz

# plt.plot(t, S, color='r')
plt.plot(t, sp, color='r')
# plt.plot(t, freq_in_hertz, color='r')
# plt.plot(freq_in_hertz, sp, color='b')
plt.show()

# t = np.arange(1260)
# sig = np.sin(t)

# sp = np.fft.fft(sig)

# print sig.shape
# print sp.shape

# print t.shape
# freq = np.fft.fftfreq(1260)
# print freq.shape

# plt.plot(t, sig, color='b')
# plt.plot(t, sp, color='r')
# plt.plot(t, sp, color='r')
# plt.plot(freq, sp.real, color='b')
# plt.show()

# sp = np.fft.fft(np.sin(t))
# freq = np.fft.fftfreq(t.shape[-1])
# plt.plot(freq, sp.real, freq, sp.imag)
# plt.show()