# -*- coding: utf-8 -*-

import json, time, subprocess, re
from datetime import datetime, timedelta
from scrapy.crawler import CrawlerProcess
from voice import VoiceInputParser

import pyaudio
import wave
import sys

SPEAK_COMMAND = 'say'
VOICE = '-v Alex'
VOICE_SPEED = '-r 200'

# SPEAK_COMMAND = 'espeak'
# VOICE = '-ven-uk-rp'
# VOICE_SPEED = ''

def execute(proc, cmd):
    proc = subprocess.Popen(cmd, shell=True)
    proc.communicate()

class Bot():
    def __init__(self):
        self.running = False
        self.news = NewsReader()
        self.voice_parser = VoiceInputParser(self)
        self.cur_speech_process = None
        self.last_update = datetime.now()
        self.alarms = []
        self.alarm_tones = []
        self.alarm_index = 0
        self.cur_alarm_process = None

    def run(self):
        self.running = True
        
        # create thread to listen for voice input
        # self.voice_parser.start()

        while(self.running):
            # for time in self.alarms:
            #     self.check_alarm(time) 
            self.speak('What can I do for you, Mike?')
            response = self.voice_parser.listen()

            text = response['_text']
            if 'news' in text:
                self.read_news()

            if 'read' in text:
                num_dict = {
                    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
                }
                arr = text.split(' ')

                for a in ['article', 'articles']:
                    if a in arr:
                        num = arr[arr.index(a) + 1]
                        index = num_dict[num]
                        self.read_full_article(int(index)-1)


    def stop(self):
        self.running = False

    # Update all bot data
    def update_data(self):
        # update news
        self.news.update()
        self.last_update = datetime.now()

    # Speak the given text
    def speak(self, text):
        # open a speech commanbd subprocess
        self.cur_speech_process = subprocess.Popen('%s %s "%s" %s' %\
         (SPEAK_COMMAND, VOICE, str(text), VOICE_SPEED), shell=True)
        # wait for the process to finish
        self.cur_speech_process.communicate()

    def stop_talking(self):
        print 'stop talking'
        if self.cur_speech_process:
            self.cur_speech_process.kill()
        if self.cur_alarm_process:
            self.cur_alarm_process.kill()

    def set_alarm(self, date, tone):
        self.alarms += [date]
        self.alarm_tones += [tone]

    def check_alarm(self, time):
        now = datetime.now()
        if now > time and now < time + timedelta(milliseconds=10):
            self.alarm_index = (self.alarm_index + 1) % len(self.alarms)
            self.raise_alarm()

    def raise_alarm(self):
        # execute(self.cur_alarm_process, 'afplay %s' % self.alarm_tones[self.alarm_index])
        self.cur_alarm_process = subprocess.Popen('afplay %s' % self.alarm_tones[self.alarm_index], shell=True)
        self.cur_alarm_process.communicate()

    """
    Read the news headlines one by one.
    User commands:
    'read article <index>': read the full article
    'abstract article <index>': read the abstract
    'repeat current': start over current article
    'repeat all': start over
    """
    def read_news(self):
        self.speak('Today\'s headlines')
        time.sleep(1)
        headlines = [article['title'][0] for article in self.news.articles]

        # read the headlines
        for i, h in enumerate(headlines):
            self.speak(i+1)
            self.speak(h)

    def read_full_article(self, index):
        self.speak('Reading article ' + str(index+1))
        lines = self.news.articles[index]['body']
        for line in lines[5:]:
            text = line.encode('utf-8').replace('\"', '\\"')
            # print text
            self.speak(text)


class NewsReader():
    def __init__(self):
        self.articles = []

    def update(self):
        # crawl the web
        # self.crawl()

        # load the news data file
        self.articles = []
        with open('news.json') as datafile:
            for line in datafile:
                # print line
                self.articles.append(json.loads(line))

    def crawl(self):
        process = CrawlerProcess({
            'BOT_NAME': 'news_crawler',
            'SPIDER_MODULES': ['news_crawler.news_crawler.spiders'],
            # 'DOWNLOAD_DELAY': 2,
            'ITEM_PIPELINES': {
                   'news_crawler.news_crawler.pipelines.NewsJsonOutputPipeline': 300,
                }
            })

        # crawl cnn.com
        process.crawl('cnn_spider')

        # the script will block here until the crawling is finished
        process.start() 
        
if __name__ == '__main__':
    CHUNK = 1024

    if len(sys.argv) < 2:
        print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
        sys.exit(-1)

    wf = wave.open(sys.argv[1], 'rb')

    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(CHUNK)

    while data != '':
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()

    p.terminate()

    b = Bot()
    b.update_data()
    # b.read_news()
    b.set_alarm(datetime.now() + timedelta(seconds=5), 'red.mp3')
    b.run()
    # b.read_full_article(2)


