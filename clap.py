# -*- coding: utf-8 -*-

DEBUG = False

import pyaudio, sys, thread, time, json, array, pickle, datetime, os, threading
import traceback
from Queue import Queue
import numpy as np
if DEBUG:
    import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

# used for function approximation
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

plots_dir = 'clap_plots/'
clap_data_dir = 'clap_data/'
clap_model_dir = 'clap_data/'
clap_classifier_fname = 'clap_classifier.p'
clap_decay_model_fname = 'clap_decay_model.npy'

articles = []
with open('news.json', 'r') as datafile:
    for line in datafile:
        articles.append(json.loads(line))

class ClapDetector():
    def __init__(self, queue):
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self.run, args=(self._stop,))
        self.thread_queue = queue

        self.analysis_q = Queue()
        self._stop_analysis = threading.Event()
        self.analysis_thread = None

        # recording parameters
        self.WINDOW_SIZE_SEC = 1
        self.WINDOW_STEP_SEC = 0.1
        self.CHUNK = 8192
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.freq_cutoff = 5000
        self.record_window = 2

        # decay model parameters
        self.skip_samples = 1000        # skip this many samples from peak when creating the model
        self.decay_samples = 13000      # consider this many samples after the detected peak
        self.min_peak_distance = 5000   # min distance between peaks
        self.peak_threshold = 20000     # peak threshold
        # clap detection threshold for chi squared gradient error
        self.clap_decay_model_distance_threshold = 5.0

        # initialize audio stream
        self.p = pyaudio.PyAudio()
        self.stream = None

        # load clap detection model
        try:
            self.clap_clf = pickle.load(open(clap_model_dir + clap_classifier_fname, 'r'))
            self.clap_decay_model = np.load(clap_model_dir + clap_decay_model_fname)
        except IOError:
            self.clap_clf = None
            self.clap_decay_model = None

    def start(self):
        if not self.thread.is_alive():
            self._stop.clear()
            self.thread = threading.Thread(target=self.run)
            self.thread.start()

    def stop(self):
        print 'Stopping clap detector'
        self.terminate_audio_stream()
        self._stop.set()

    def running(self):
        return self.thread.is_alive()

    def run(self):
        self.listen_moving()
        # self.listen()

    def init_audio_steam(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                        channels=self.CHANNELS, 
                        rate=self.RATE, 
                        input=True,
                        output=True,
                        frames_per_buffer=self.CHUNK)

    def terminate_audio_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

    # """
    # Continuously records audio listening for a clap
    # Stops thread on detection
    # """
    # def listen(self):
    #     self.init_audio_steam()
    #     print "Clap detection initialized"
    #     try:
    #         while not self._stop.is_set():
    #             # record claps and get frequencies
    #             frame_data = self.record(self.record_window)

    #             clap = self.check_clap(frame_data)
    #             if clap:
    #                 print 'Clapped'
    #                 self.thread_queue.put('clap')
    #                 self.stop()

    #     except (KeyboardInterrupt, SystemExit):
    #         print "\rExiting clap"
    #         self.terminate_audio_stream()

    """
    Record audio for the specified number of seconds
    """
    def record(self, seconds):
        print '######## Recording...'
        self.init_audio_steam()
        data = array.array('h')
        try:
            for _ in np.arange(int(self.RATE / self.CHUNK * seconds)):
                data.fromstring(self.stream.read(self.CHUNK))
        except Exception:
            traceback.print_exc()

        self.terminate_audio_stream()
        data_arr = np.array(data, dtype=np.int16)
        print 'got %s bytes' % data_arr.size
        return data_arr


    """
    Continuously records audio listening for a clap
    Stops thread on detection
    """
    def listen_moving(self):
        self.init_audio_steam()
        self.thread_queue.queue.clear()
        self.analysis_q.queue.clear()
        print "Clap detection initialized"
        num_frames = int(self.RATE/self.CHUNK * self.WINDOW_SIZE_SEC)
        frames = [np.zeros(self.CHUNK) for _ in range(num_frames)]
        last_checked = None
        try:
            while not self._stop.is_set():
                data = array.array('h')
                try:
                    data.fromstring(self.stream.read(self.CHUNK))
                except IOError:
                    print 'Stream error'
                    sys.exit(0)
                frames.append(np.array(data, dtype=np.int16))
                frames.pop(0)

                now = datetime.datetime.now()
                if (not last_checked or now - last_checked > datetime.timedelta(seconds=self.WINDOW_STEP_SEC))\
                    and (not self.analysis_thread or not self.analysis_thread.is_alive()):
                    self.analysis_thread = threading.Thread(target=self.analyze, args=(frames,))
                    self.analysis_thread.start()
                    last_checked = now

                if not self.analysis_q.empty() and self.analysis_q.get() == 'clap':
                    self.thread_queue.put('clap')
                    self.stop()
        except (KeyboardInterrupt, SystemExit):
            print "\rExiting clap"
            self.terminate_audio_stream()

    def analyze(self, frames):
        # clocked in at < 0.006 sec
        # start = datetime.datetime.now()
        print 'Analyzing...'
        data = np.array(frames).flatten()
        clap = self.check_clap(data)
        if clap:
            print 'Clapped!!!!'
            self.analysis_q.put('clap')
        # print (datetime.datetime.now() - start).total_seconds()

    """
    Compare the audio stream data to a clap waveform
    Could use ML for this
    """
    def check_clap(self, frame_data):
        X = np.array([])
        max_indices = self.get_peaks(5, frame_data)
        # print 'Max indices: ' + str(max_indices)

        # compute decay function
        for i in max_indices:
            points = self.compute_decay_model(i, frame_data)
            if points.size > 0:
                # check if points match
                dist = self.distance(self.clap_decay_model, points)
                print 'distance: %f' % dist
                if dist < self.clap_decay_model_distance_threshold:
                    return True
        return False

        # iterate through frames using the window size that was used to train the model
        # window_size = 0.5*self.RATE
        # for i in range(int(frame_data.size - window_size))[::self.RATE/10]:
        #     # sys.stdout.write('\r %d' % i)

        #     # get frequencies
        #     data = frame_data[i:i+window_size]
        #     # data = frame_data
        #     freqs, spectrum = self.fft(data)

        #     if X.size > 0:
        #         X = np.concatenate((X, [spectrum]))
        #     else:
        #         X = np.array([spectrum])
        #     # X = np.array([spectrum])

        # # classify data using classification model
        # preds = self.clap_clf.predict(X)
        # return preds.any()

    """
    Fast fourier transform to get single sided frequency spectrum
    """
    def fft(self, data):
        # perform fft
        sp = np.fft.fft(data)
        L = data.size

        # get the single sided spectrums
        # http://www.mathworks.com/help/matlab/ref/fft.html
        spectrum_ss = np.abs(sp.real/L)[1:L/2+1]
        spectrum_ss[2:-1] *= 2
        freqs_ss = np.fft.fftfreq(L, d=1.0/self.RATE)[:L/2]
        return (freqs_ss[:self.freq_cutoff], spectrum_ss[:self.freq_cutoff])

    def distance(self, model, data):
        assert model.shape[0] == data.shape[0]

        # chi squared
        # dists = data - model
        # avg_dist = np.mean(dists)
        # data_centered = data - avg_dist
        # dists_centered = dists - np.mean(dists)

        # plt.plot(model, 'b')
        # plt.plot(data, 'r')
        # plt.savefig('%s.png'%datetime.datetime.now())
        # plt.clf()

        # chi squared gradient error
        model_gradient = np.abs(np.gradient(model))
        gradient_dists_chi = np.abs(np.gradient(data)) - model_gradient
        gradient_dists_chi = np.square(gradient_dists_chi) / model_gradient
        # can't normalize because we don't have a global min/max
        error_chi = np.mean(gradient_dists_chi)

        if DEBUG:
            plt.plot(model_gradient, 'b')
            plt.plot(np.gradient(data), 'r')
            plt.savefig('%s.png'%datetime.datetime.now())
            plt.clf()

        # use centered data to calculate chi error
        # dists_chi = np.square(dists_centered) / model
        # dists_chi = (dists_chi * 1/dists_chi.max()) - dists_chi.min()
        # error_chi = np.mean(dists_chi)

        # variance = np.log(np.var(dists))
        # print 'variance: %f' % variance

        # shape comparison
        # plt.plot(dists)
        # plt.show()
        return error_chi

    def get_peaks(self, num_peaks, data):
        # find peak amplitudes at least min_peak_distance samples apart
        max_indices = []
        for i in np.argsort(data)[::-1]:
            if len(max_indices) >= num_peaks or data[i] < self.peak_threshold:
                break
            if not max_indices or np.all(np.abs(max_indices - i) > self.min_peak_distance):
                max_indices += [i]
        return max_indices

    def compute_decay_model(self, index, data):
        i = index + self.skip_samples
        data = np.abs(data[i:min(data.shape[0], i+self.decay_samples)])
        if data.shape[0] < self.decay_samples:
            print 'Sample %d out of range' % index
            return np.array([])

        # take the max of every 100 samples to get outer function
        data_modified = np.array([np.max(data[i:i+100]) for i in range(data.shape[0])[::10]])

        # x = np.array(range(data.size))
        # y = data
        xmod = np.array(np.arange(data_modified.shape[0]))
        ymod = data_modified
        X = xmod[:, np.newaxis]

        # plt.scatter(x, y, label="training points")
        # plt.scatter(xmod, ymod , color='r')

        model = make_pipeline(PolynomialFeatures(4), Ridge())
        model.fit(X, ymod)
        y_out = model.predict(X)
        # plt.scatter(xmod, y_plot, color='b')
        # plt.show()
        return y_out

    def train_claps(self, seconds=None, samples=10):
        self.train(seconds, samples, True)

    def train_noclaps(self, seconds=None, samples=10):
        self.train(seconds, samples, False)

    """
    Records sample claps and non-claps to train detection algorithm
    """
    def train(self, seconds, samples, is_clap):
        for i in range(3)[::-1]:
            print i+1
            time.sleep(1)

        # need to change chunk size here for some reason
        self.CHUNK = 8192
        if not seconds:
            seconds = self.record_window
        all_spectrums = []
        all_decay_models = []

        for i in range(samples):
            phrase = 'Clap now!' if is_clap else 'Do anything but clap now!'
            print '%d %s' % (i, phrase)

            # record claps
            frame_data = self.record(seconds)

            # update frequency data for classification model
            freqs, spectrum = self.fft(frame_data)
            all_spectrums.append(spectrum)

            if DEBUG:
                # Save raw plots
                plt.plot(frame_data, color='b')
                fname = 'claps_raw' if is_clap else 'noclap_raw'
                # plt.show()
                plt.savefig('%s%s%d.png' % (plots_dir, fname, i))
                plt.clf()

                # Save spectrum plots
                plt.plot(freqs, spectrum, 'b')
                fname = 'claps_spectrum' if is_clap else 'noclap_spectrum'
                # plt.show()
                plt.savefig('%s%s%d.png' % (plots_dir, fname, i))
                plt.clf()

            if is_clap:
                # update waveform data for decay model
                max_indices = self.get_peaks(1, frame_data)
                if max_indices:
                    decay_model = self.compute_decay_model(max_indices[0], frame_data)
                    all_decay_models.append(decay_model)

                    if DEBUG:
                        # Save decay plots
                        plt.plot(decay_model, color='b')
                        fname = 'claps_decay'
                        # plt.show()
                        plt.savefig('%s%s%d.png' % (plots_dir, fname, i))
                        plt.clf()

        # Save the training data
        np.save('%sfreqs.npy' % clap_data_dir, freqs)
        fname = 'claps/spectrum/claps_spectrum' if is_clap else 'noclaps/spectrum/noclap_spectrum'
        np.save('%s%s%s' % (clap_data_dir, fname, datetime.datetime.now()), all_spectrums)
        
        if is_clap:
            fname = 'claps/decay/decay'
            np.save('%s%s%s' % (clap_data_dir, fname, datetime.datetime.now()), all_decay_models)

    def train_model(self):
        ### Train spectrum data
        # form training data and labels
        X = np.empty((0, self.freq_cutoff), int)
        y = np.empty((0, 1), int)

        data_dir = 'clap_data/claps/spectrum/'
        for fname in os.listdir(data_dir):
            data = np.load("%s%s"% (data_dir, fname))
            X = np.append(X, data, axis=0)
            y = np.append(y, [1] * data.shape[0])

        data_dir = 'clap_data/noclaps/spectrum/'
        for fname in os.listdir(data_dir):
            data = np.load("%s%s"% (data_dir, fname))
            X = np.append(X, data, axis=0)
            y = np.append(y, [0] * data.shape[0])

        # pca = PCA(n_components=200)
        # X_pca = pca.fit_transform(X)

        # fit the model
        # clf = LogisticRegression(penalty='l1')
        clf = LinearDiscriminantAnalysis()
        clf.fit(X, y)
        preds = clf.predict(X)
        # X_new = clf.transform(X)

        # clf2 = LinearDiscriminantAnalysis()
        # clf2.fit(X_new, y)
        # preds2 = clf2.predict(X_new)

        # print X.shape, X_pca.shape
        print preds
        print np.sum(preds), preds.size
        # print preds2, np.sum(preds2)

        # save model
        pickle.dump(clf, open(clap_model_dir + clap_classifier_fname, 'w'))
        self.clap_clf = clf

        ### Train decay data
        X = np.empty((0, self.decay_samples/10), int)

        data_dir = 'clap_data/claps/decay/'
        for fname in os.listdir(data_dir):
            if fname.endswith('npy'):
                data = np.load("%s%s"% (data_dir, fname))
                print data.shape, X.shape
                X = np.append(X, data, axis=0)

        print X.shape
        X_avg = np.mean(X, axis=0)
        plt.plot(X_avg)
        plt.show()

        # Average decay data
        np.save('%s%s' % (clap_model_dir, clap_decay_model_fname), X_avg)


if __name__ == '__main__':
    # ClapDetector(Queue()).train_claps(samples=20)
    # ClapDetector(Queue()).train_noclaps(samples=20)
    # ClapDetector(Queue()).train_model()
    # ClapDetector(Queue()).listen()
    ClapDetector(Queue()).listen_moving()

