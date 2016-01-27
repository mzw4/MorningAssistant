import array
import pyaudio 
import matplotlib.pyplot as plt
import numpy as np

FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_CHANNEL=2
RATE = 44100
CHUNK = 512
RECORD_SECONDS = 5

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=INPUT_CHANNEL,
                frames_per_buffer =CHUNK)

print("* recording")


for i in range(10):
    try:
        data = array.array('h')
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data.fromstring(stream.read(CHUNK))
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    # frame_data = np.hstack(frames)
    frame_data = np.array(data)

    Fs = 44100
    L = frame_data.size

    #plot raw data
    plt.plot(frame_data, color='b')
    plt.show()
    
    sp = np.fft.fft(frame_data)
    P2 = np.abs(sp/L);
    P1 = P2[:L/2];
    P1[2:-1] = 2*P1[2:-1];
    spectrum_ss = P1
    freqs_ss = np.fft.fftfreq(L, d=1.0/Fs)[:L/2]
    # freqs = np.fft.fftfreq(frame_data.size, d=1.0/self.RATE)

    # get the single sided spectrums
    # spectrum_ss = np.abs(spectrum.real)[:spectrum.size/2]
    # freqs_ss = freqs[:spectrum.size/2]

    # Save the plots for reference
    plt.plot(freqs_ss, spectrum_ss, 'b')
    plt.show()
print("* done recording")