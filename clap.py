# -*- coding: utf-8 -*-

import pyaudio, sys, thread, time
from array import array
import numpy as np
import matplotlib.pyplot as plt
# np.set_printoptions(threshold=np.nan)

# clap = 0
# wait = 2
# flag = 0
# pin = 24
# exitFlag = False    

# def waitForClaps(threadName):
#     global clap
#     global flag
#     global wait
#     global exitFlag
#     global pin
#     print "Waiting for more claps"
#     time.sleep(wait)
#     if clap == 2:
#         print "Two claps"
#     # elif clap == 3:
#     #   print "Three claps"
#     elif clap == 4:
#         exitFlag = True
#     print "Claping Ended"
#     clap = 0
#     flag = 0

class ClapDetector():
    def __init__(self):
        self.WINDOW_SIZE_SEC = 0.5

        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.threshold = 30000
        self.max_value = 0
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                        channels=self.CHANNELS, 
                        rate=self.RATE, 
                        input=True,
                        output=True,
                        frames_per_buffer=self.CHUNK)

    """
    Compare the audio stream data to a clap waveform
    Could use ML for this
    """
    def check_clap(self, frames):
        # concatenate frames
        data = np.hstack(frames)

        # fast fourier transform to get frequency spectrum
        sp = np.fft.fft(data)
        freqs = np.fft.fftfreq(data.size, d=1.0/self.RATE)

        # plt.plot(np.abs(sp.real)[:sp.size/2], 'b')
        # plt.plot(freqs[:sp.size/2], np.abs(sp.real)[:sp.size/2], 'b')
        # plt.show()
        return False

    """
    Records waveform of a clap for the specified time
    """
    def set_clap(self, seconds=0.5):
        frames = []
        for _ in np.arange(int(self.RATE / self.CHUNK * seconds)):
            data = self.stream.read(self.CHUNK)
            frames.append(np.fromstring(data, dtype=np.int16))

        # concatenate frames
        data = np.hstack(frames)

        # fast fourier transform to get frequency spectrum
        sp = np.fft.fft(data)
        freqs = np.fft.fftfreq(data.size, d=1.0/self.RATE)

        plt.plot(freqs[:sp.size/2], np.abs(sp.real)[:sp.size/2], 'b')
        plt.show()

    def listen(self):
        # global clap
        # global flag
        # global pin

        # maintain a moving window with frames for the window size
        # all_frames = []
        # all_freqs = []
        frames = [np.zeros(self.CHUNK)] * int(self.RATE / self.CHUNK * self.WINDOW_SIZE_SEC)

        try:
            print "Clap detection initialized"
            while True:
                print "read input"
                data = self.stream.read(self.CHUNK)
                data_arr = np.fromstring(data, dtype=np.int16)
                # max_val = data_arr.max()
                # print max_val, max_value

                # add frame and remove oldest
                frames.append(data_arr)
                frames.pop(0)

                clap = self.check_clap(frames)
                if clap:
                    print 'Clapped'

                # fast fourier transform to get frequency values
                # sp = np.fft.fft(data_arr)
                # freqs = np.fft.fftfreq(data_arr.size)
                # all_frames.append(sp)
                # all_freqs.append(np.abs(freqs * self.RATE))

                # if max_val > self.threshold:
                #     clap += 1
                #     print "Clapped"
                # if clap == 1 and flag == 0:
                #     thread.start_new_thread( waitForClaps, ("waitThread",) )
                #     flag = 1
                # if exitFlag:
                #     sys.exit(0)

        except (KeyboardInterrupt, SystemExit):
            print "\rExiting"
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

ClapDetector().set_clap()

