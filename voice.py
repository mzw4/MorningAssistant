# Preparing pywit
import wit, threading, json

TOKEN = 'UIZ3JN2S2DF4HFI7F6EGVJHAFC7R2NGP'

class VoiceInputParser():#threading.Thread):
    def __init__(self, bot):
        self.bot = bot
        # threading.Thread.__init__(self)
        self._stop = threading.Event()
        wit.init()

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
        response = wit.voice_query_auto(TOKEN)
        return json.loads(response)

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