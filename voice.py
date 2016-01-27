# Preparing pywit
import wit, threading, json
import speech_recognition as sr

# TOKEN = 'UIZ3JN2S2DF4HFI7F6EGVJHAFC7R2NGP'

class VoiceInputParser():#threading.Thread):
    def __init__(self, bot):
        self.bot = bot
        # threading.Thread.__init__(self)
        self._stop = threading.Event()
        # wit.init()

        self.RATE = 44100
        self.CHUNK = 8192

    def run(self):
        pass
        # while(not self._stop.is_set()):
        #     print 'listening'
        # self.listen_async()
    
    def stop(self):
        self._stop.set()

    def listen_async(self):
        wit.voice_query_auto_async(TOKEN, self.handle_response)

    def listen(self):
        # response = wit.voice_query_auto(TOKEN)
        # return json.loads(response)

        # https://github.com/Uberi/speech_recognition
        # IMPORTANT! must specify chunk_size for this to work with multiple streams and clap detection
        r = sr.Recognizer()
        with sr.Microphone(chunk_size=self.CHUNK, sample_rate=self.RATE) as source:
            print "Say something!"
            audio = r.listen(source)

        # recognize speech using Google Speech Recognition
        text = None
        try:
            text = r.recognize_google(audio)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return text

    def handle_response(self, response):
        try:
            response = json.loads(response)
            if response['_text'] == 'stop':
                self.bot.stop_talking()

            # continuously call the listen function
            self.listen()
        except Exception as e:
            print e

if __name__ == '__main__':
    wit.init()

    # Let's hack!
    resp = wit.text_query('Turn on the lights', TOKEN)
    response = wit.voice_query_auto(TOKEN)

    print('Response: {}'.format(resp))
    print('Response: {}'.format(response))

    # Wrapping up
    wit.close()