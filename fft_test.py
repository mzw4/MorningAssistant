import numpy as np
import matplotlib.pyplot as plt

Fs = 1000            # Sampling frequency
T = 1.0/Fs             # Sampling period
L = 1000
t = np.arange(L)*T        # Time vector

S = 0.7 * np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t)

sp = np.fft.fft(S)

# matplot lib example
P2 = np.abs(sp/L);
P1 = P2[:L/2];
P1[2:-1] = 2*P1[2:-1];
f = Fs*np.arange(L/2)/L;
plt.plot(f,P1)
plt.show()



freqs = np.fft.fftfreq(S.size, d=1.0/Fs)
print(freqs.min(), freqs.max())

idx = np.argmax(np.abs(sp))
print idx
freq = freqs[idx]
print freq

# plt.plot(t, S, color='r')
plt.plot(freqs, np.abs(sp.real), color='r')
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