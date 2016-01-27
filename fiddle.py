import pyaudio, sys, thread, time, json, array, subprocess, re
import numpy as np
from Queue import Queue
# import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle
import pyaudio
from clap import ClapDetector

def ml():
    claps = np.load('clap_data/claps_spectrum.npy')
    noclaps = np.load('clap_data/background_spectrum.npy')

    # form training data and labels
    X = np.vstack((claps, noclaps))
    y = np.array([1 for _ in range(claps.shape[0])] + [0 for _ in range(noclaps.shape[0])])

    # perform logistic regression
    clf = LogisticRegression(penalty='l1')
    clf.fit(X, y)
    print clf.predict(X)

    # save model
    pickle.dump(clf, open('clap_classifier.p', 'w'))



SPEAK_COMMAND = 'say'
VOICE = '-v Alex'
VOICE_SPEED = '-r 200'

# SPEAK_COMMAND = 'espeak'
# VOICE = '-ven-uk-rp'
# VOICE_SPEED = ''

def execute(proc, cmd):
    proc = subprocess.Popen(cmd, shell=True)
    proc.communicate()

# Speak the given text
def speak(text):
    # open a speech commanbd subprocess
    cur_speech_process = subprocess.Popen('%s %s "%s" %s' %\
     (SPEAK_COMMAND, VOICE, str(text), VOICE_SPEED), shell=True)
    # wait for the process to finish
    cur_speech_process.communicate()

def record(seconds):
    CHUNK = 512
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                channels=CHANNELS, 
                rate=RATE, 
                input=True,
                output=True,
                frames_per_buffer=CHUNK)

    print 'Recording...'
    data = array.array('h')            
    try:
        for _ in np.arange(int(RATE / CHUNK * seconds)):
            data.fromstring(stream.read(CHUNK))
    finally:
        stream.stop_stream()
        stream.close()
    data_arr = np.array(data, dtype=np.int16)
    print 'got %s bytes' % data_arr.size
    return data_arr

def listen(self):
    p = pyaudio.PyAudio()
    self.stream = p.open(format=self.FORMAT,
                channels=self.CHANNELS, 
                rate=self.RATE, 
                input=True,
                output=True,
                frames_per_buffer=self.CHUNK)
    # self.init_audio_steam()
    # print "Clap detection initialized"
    try:
        while not self._stop.is_set():
            # record claps and get frequencies
            frame_data = self.record(self.record_window)

            clap = self.check_clap(frame_data)
            if clap:
                print 'Clapped'
                self.thread_queue.put('clap')
                self.stop()

    except (KeyboardInterrupt, SystemExit):
        print "\rExiting"
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

speak('hi')
q = Queue()
cd = ClapDetector(q)
cd.start()

while q.empty():
    pass
q.get()
speak('hello')
cd.start()

while q.empty():
    pass

import io, os, subprocess, wave, base64
import math, audioop, collections, threading
import platform, stat
import json

try: # try to use python2 module
    from urllib2 import Request, urlopen, URLError, HTTPError
except ImportError: # otherwise, use python3 module
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError

class AudioData(object):
    def __init__(self, frame_data, sample_rate, sample_width):
        assert sample_rate > 0, "Sample rate must be a positive integer"
        assert sample_width % 1 == 0 and sample_width > 0, "Sample width must be a positive integer"
        self.frame_data = frame_data
        self.sample_rate = sample_rate
        self.sample_width = int(sample_width)

    def get_wav_data(self):
        """
        Returns a byte string representing the contents of a WAV file containing the audio represented by the ``AudioData`` instance.
        Writing these bytes directly to a file results in a valid WAV file.
        """
        with io.BytesIO() as wav_file:
            wav_writer = wave.open(wav_file, "wb")
            try: # note that we can't use context manager due to Python 2 not supporting it
                wav_writer.setframerate(self.sample_rate)
                wav_writer.setsampwidth(self.sample_width)
                wav_writer.setnchannels(1)
                wav_writer.writeframes(self.frame_data)
            finally:  # make sure resources are cleaned up
                wav_writer.close()
            wav_data = wav_file.getvalue()
        return wav_data

    def get_flac_data(self):
        """
        Returns a byte string representing the contents of a FLAC file containing the audio represented by the ``AudioData`` instance.
        Writing these bytes directly to a file results in a valid FLAC file.
        """
        wav_data = self.get_wav_data()

        # determine which converter executable to use
        system = platform.system()
        path = os.path.dirname(os.path.abspath(__file__)) # directory of the current module file, where all the FLAC bundled binaries are stored
        flac_converter = shutil_which("flac") # check for installed version first
        if flac_converter is None: # flac utility is not installed
            if system == "Windows" and platform.machine() in ["i386", "x86", "x86_64", "AMD64"]: # Windows NT, use the bundled FLAC conversion utility
                flac_converter = os.path.join(path, "flac-win32.exe")
            elif system == "Linux" and platform.machine() in ["i386", "x86", "x86_64", "AMD64"]:
                flac_converter = os.path.join(path, "flac-linux-i386")
            elif system == "Darwin" and platform.machine() in ["i386", "x86", "x86_64", "AMD64"]:
                flac_converter = os.path.join(path, "flac-mac")
            else:
                raise OSError("FLAC conversion utility not available - consider installing the FLAC command line application using `brew install flac` or your operating system's equivalent")

        # mark FLAC converter as executable
        try:
            stat_info = os.stat(flac_converter)
            os.chmod(flac_converter, stat_info.st_mode | stat.S_IEXEC)
        except OSError: pass

        # run the FLAC converter with the WAV data to get the FLAC data
        process = subprocess.Popen("\"{0}\" --stdout --totally-silent --best -".format(flac_converter), stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        flac_data, stderr = process.communicate(wav_data)
        return flac_data

class AudioSource(object):
    def __init__(self):
        raise NotImplementedError("this is an abstract class")

    def __enter__(self):
        raise NotImplementedError("this is an abstract class")

    def __exit__(self, exc_type, exc_value, traceback):
        raise NotImplementedError("this is an abstract class")

class Microphone(AudioSource):
        """
        This is available if PyAudio is available, and is undefined otherwise.
        Creates a new ``Microphone`` instance, which represents a physical microphone on the computer. Subclass of ``AudioSource``.
        If ``device_index`` is unspecified or ``None``, the default microphone is used as the audio source. Otherwise, ``device_index`` should be the index of the device to use for audio input.
        A device index is an integer between 0 and ``pyaudio.get_device_count() - 1`` (assume we have used ``import pyaudio`` beforehand) inclusive. It represents an audio device such as a microphone or speaker. See the `PyAudio documentation <http://people.csail.mit.edu/hubert/pyaudio/docs/>`__ for more details.
        The microphone audio is recorded in chunks of ``chunk_size`` samples, at a rate of ``sample_rate`` samples per second (Hertz).
        Higher ``sample_rate`` values result in better audio quality, but also more bandwidth (and therefore, slower recognition). Additionally, some machines, such as some Raspberry Pi models, can't keep up if this value is too high.
        Higher ``chunk_size`` values help avoid triggering on rapidly changing ambient noise, but also makes detection less sensitive. This value, generally, should be left at its default.
        """
        def __init__(self, device_index = None, sample_rate = 16000, chunk_size = 1024):
            assert device_index is None or isinstance(device_index, int), "Device index must be None or an integer"
            if device_index is not None: # ensure device index is in range
                audio = pyaudio.PyAudio(); count = audio.get_device_count(); audio.terminate() # obtain device count
                assert 0 <= device_index < count, "Device index out of range"
            assert isinstance(sample_rate, int) and sample_rate > 0, "Sample rate must be a positive integer"
            assert isinstance(chunk_size, int) and chunk_size > 0, "Chunk size must be a positive integer"
            self.device_index = device_index
            self.format = pyaudio.paInt16 # 16-bit int sampling
            self.SAMPLE_WIDTH = pyaudio.get_sample_size(self.format) # size of each sample
            self.SAMPLE_RATE = sample_rate # sampling rate in Hertz
            self.CHUNK = chunk_size # number of frames stored in each buffer

            self.audio = None
            self.stream = None

        def __enter__(self):
            assert self.stream is None, "This audio source is already inside a context manager"
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                input_device_index = self.device_index, channels = 1,
                format = self.format, rate = self.SAMPLE_RATE, frames_per_buffer = self.CHUNK,
                input = True, # stream is an input stream
            )
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if not self.stream.is_stopped():
                self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            self.audio.terminate()

class Recognizer(AudioSource):
    def __init__(self):
        """
        Creates a new ``Recognizer`` instance, which represents a collection of speech recognition functionality.
        """
        self.energy_threshold = 300 # minimum audio energy to consider for recording
        self.dynamic_energy_threshold = True
        self.dynamic_energy_adjustment_damping = 0.15
        self.dynamic_energy_ratio = 1.5
        self.pause_threshold = 0.8 # seconds of non-speaking audio before a phrase is considered complete
        self.phrase_threshold = 0.3 # minimum seconds of speaking audio before we consider the speaking audio a phrase - values below this are ignored (for filtering out clicks and pops)
        self.non_speaking_duration = 0.5 # seconds of non-speaking audio to keep on both sides of the recording

    def listen(self, source, timeout = None):
        print source 

        """
        Records a single phrase from ``source`` (an ``AudioSource`` instance) into an ``AudioData`` instance, which it returns.
        This is done by waiting until the audio has an energy above ``recognizer_instance.energy_threshold`` (the user has started speaking), and then recording until it encounters ``recognizer_instance.pause_threshold`` seconds of non-speaking or there is no more audio input. The ending silence is not included.
        The ``timeout`` parameter is the maximum number of seconds that it will wait for a phrase to start before giving up and throwing an ``speech_recognition.WaitTimeoutError`` exception. If ``timeout`` is ``None``, it will wait indefinitely.
        """
        assert isinstance(source, AudioSource), "Source must be an audio source"
        assert self.pause_threshold >= self.non_speaking_duration >= 0

        seconds_per_buffer = (source.CHUNK + 0.0) / source.SAMPLE_RATE
        pause_buffer_count = int(math.ceil(self.pause_threshold / seconds_per_buffer)) # number of buffers of non-speaking audio before the phrase is complete
        phrase_buffer_count = int(math.ceil(self.phrase_threshold / seconds_per_buffer)) # minimum number of buffers of speaking audio before we consider the speaking audio a phrase
        non_speaking_buffer_count = int(math.ceil(self.non_speaking_duration / seconds_per_buffer)) # maximum number of buffers of non-speaking audio to retain before and after
        print 1
        # read audio input for phrases until there is a phrase that is long enough
        elapsed_time = 0 # number of seconds of audio read
        while True:
            frames = collections.deque()
            print 2
            # store audio input until the phrase starts
            while True:
                elapsed_time += seconds_per_buffer
                if timeout and elapsed_time > timeout: # handle timeout if specified
                    raise WaitTimeoutError("listening timed out")
                print 3
                buffer = source.stream.read(source.CHUNK)
                print 4
                if len(buffer) == 0: break # reached end of the stream
                frames.append(buffer)
                if len(frames) > non_speaking_buffer_count: # ensure we only keep the needed amount of non-speaking buffers
                    frames.popleft()
                print 4

                # detect whether speaking has started on audio input
                energy = audioop.rms(buffer, source.SAMPLE_WIDTH) # energy of the audio signal
                if energy > self.energy_threshold: break

                # dynamically adjust the energy threshold using assymmetric weighted average
                if self.dynamic_energy_threshold:
                    damping = self.dynamic_energy_adjustment_damping ** seconds_per_buffer # account for different chunk sizes and rates
                    target_energy = energy * self.dynamic_energy_ratio
                    self.energy_threshold = self.energy_threshold * damping + target_energy * (1 - damping)
            print 3
            # read audio input until the phrase ends
            pause_count, phrase_count = 0, 0
            while True:
                elapsed_time += seconds_per_buffer

                buffer = source.stream.read(source.CHUNK)
                if len(buffer) == 0: break # reached end of the stream
                frames.append(buffer)
                phrase_count += 1

                # check if speaking has stopped for longer than the pause threshold on the audio input
                energy = audioop.rms(buffer, source.SAMPLE_WIDTH) # energy of the audio signal
                if energy > self.energy_threshold:
                    pause_count = 0
                else:
                    pause_count += 1
                if pause_count > pause_buffer_count: # end of the phrase
                    break

            # check how long the detected phrase is, and retry listening if the phrase is too short
            phrase_count -= pause_count
            if phrase_count >= phrase_buffer_count: break # phrase is long enough, stop listening

        # obtain frame data
        for i in range(pause_count - non_speaking_buffer_count): frames.pop() # remove extra non-speaking frames at the end
        frame_data = b"".join(list(frames))

        return AudioData(frame_data, source.SAMPLE_RATE, source.SAMPLE_WIDTH)


    def recognize_google(self, audio_data, key = None, language = "en-US", show_all = False):
        """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using the Google Speech Recognition API.
        The Google Speech Recognition API key is specified by ``key``. If not specified, it uses a generic key that works out of the box. This should generally be used for personal or testing purposes only, as it **may be revoked by Google at any time**.
        To obtain your own API key, simply following the steps on the `API Keys <http://www.chromium.org/developers/how-tos/api-keys>`__ page at the Chromium Developers site. In the Google Developers Console, Google Speech Recognition is listed as "Speech API".
        The recognition language is determined by ``language``, an IETF language tag like `"en-US"` or ``"en-GB"``, defaulting to US English. A list of supported language codes can be found `here <http://stackoverflow.com/questions/14257598/>`__. Basically, language codes can be just the language (``en``), or a language with a dialect (``en-US``).
        Returns the most likely transcription if ``show_all`` is false (the default). Otherwise, returns the raw API response as a JSON dictionary.
        Raises a ``speech_recognition.UnknownValueError`` exception if the speech is unintelligible. Raises a ``speech_recognition.RequestError`` exception if the key isn't valid, the quota for the key is maxed out, or there is no internet connection.
        """
        assert isinstance(audio_data, AudioData), "`audio_data` must be audio data"
        assert key is None or isinstance(key, str), "`key` must be `None` or a string"
        assert isinstance(language, str), "`language` must be a string"

        flac_data, sample_rate = audio_data.get_flac_data(), audio_data.sample_rate
        if key is None: key = "AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw"
        url = "http://www.google.com/speech-api/v2/recognize?client=chromium&lang={0}&key={1}".format(language, key)
        request = Request(url, data = flac_data, headers = {"Content-Type": "audio/x-flac; rate={0}".format(sample_rate)})

        # obtain audio transcription results
        try:
            response = urlopen(request)
        except HTTPError as e:
            raise RequestError("recognition request failed: {0}".format(getattr(e, "reason", "status {0}".format(e.code)))) # use getattr to be compatible with Python 2.6
        except URLError as e:
            raise RequestError("recognition connection failed: {0}".format(getattr(e, "reason", "status {0}".format(e.code)))) # use getattr to be compatible with Python 2.6
        response_text = response.read().decode("utf-8")

        # ignore any blank blocks
        actual_result = []
        for line in response_text.split("\n"):
            if not line: continue
            result = json.loads(line)["result"]
            if len(result) != 0:
                actual_result = result[0]
                break

        if show_all: return actual_result

        # return the best guess
        if "alternative" not in actual_result: raise UnknownValueError()
        for entry in actual_result["alternative"]:
            if "transcript" in entry:
                return entry["transcript"]

        # no transcriptions available
        raise UnknownValueError()


def shutil_which(pgm):
    """
    python2 backport of python3's shutil.which()
    """
    path = os.getenv('PATH')
    for p in path.split(os.path.pathsep):
        p = os.path.join(p, pgm)
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p

class WaitTimeoutError(Exception): pass
class RequestError(Exception): pass
class UnknownValueError(Exception): pass

