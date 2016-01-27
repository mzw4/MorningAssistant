# -*- coding: utf-8 -*-

import sys, json, time, subprocess, re, array, traceback
from datetime import datetime, timedelta
from Queue import Queue

import numpy as np
import pyaudio
from scrapy.crawler import CrawlerProcess
import speech_recognition as sr

from voice import VoiceInputParser
from clap import ClapDetector

SPEAK_COMMAND = 'say'
VOICE = '-v Alex'
VOICE_SPEED = '-r 200'

# SPEAK_COMMAND = 'espeak'
# VOICE = '-ven-uk-rp'
# VOICE_SPEED = ''

def execute(proc, cmd):
    proc = subprocess.Popen(cmd, shell=True)
    proc.communicate()

class States:
    Idle, Speaking, Listening = range(3)

class Bot():
    def __init__(self):
        self.running = False
        self.news = NewsReader()
        self.thread_queue = Queue()
        self.voice_parser = VoiceInputParser(self)
        self.clap_detector = ClapDetector(self.thread_queue)
        self.cur_speech_process = None
        self.speech_q = Queue()
        self.last_update = datetime.now()
        self.alarms = []
        self.alarm_tones = []
        self.alarm_index = 0
        self.cur_alarm_process = None
        self.state = States.Idle

    """
    Runs the robot state machine
    """
    def run(self):
        self.running = True
        
        # create thread to listen for voice input
        # self.voice_parser.start()
        try:
            while(self.running):
                if self.state == States.Idle:
                    """
                    Idle state
                    """
                    if not self.clap_detector.running():
                        self.clap_detector.start()
                    self.check_clap()

                elif self.state == States.Speaking:
                    """
                    Speaking state
                    """
                    if not self.clap_detector.running():
                        self.clap_detector.start()
                    self.check_clap()

                    if not self.speech_q.empty():
                        self.speak(self.speech_q.get())
                else:
                    pass

                # check for alarms
                for time in self.alarms:
                    self.check_alarm(time)

        except (Exception, KeyboardInterrupt, SystemExit) as e:
            print "\rExiting main"
            traceback.print_exc()
            self.clap_detector.stop()
            sys.exit(0)

    """
    Check the thread queue for clap detection
    If clapped, receive a new command
    """
    def check_clap(self):
        if not self.thread_queue.empty() and self.thread_queue.get() == 'clap':
            self.stop_talking()
            # self.clap_detector.start()
            self.receive_command()

    def listen(self):
        # stop the clap detector thread when listening for command
        # self.clap_detector.stop()
        self.speak('What can I do for you?')
        # response = 
        # self.clap_detector.start()
        return self.voice_parser.listen()

    def receive_command(self):
        # response = self.listen()
        self.speak('What can I do for you?')
        response = None
        while not response:
            response = self.voice_parser.listen()
        
        print response
        if 'news' in response:
            self.read_news()

        if 'read' in response:
            num_dict = {
                'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
            }
            arr = response.split(' ')

            for a in ['article', 'articles']:
                if a in arr:
                    num = arr[arr.index(a) + 1]
                    try:
                        index = int(num)
                    except:
                        index = num_dict[num]
                    self.read_full_article(int(index)-1)

        if 'train' in response:
            self.clap_detector.stop()
            if 'no claps' in response:
                self.clap_detector.train_noclaps(samples=20)
            elif 'claps' in response:
                self.clap_detector.train_claps(samples=20)

        print 'done'


    def stop(self):
        self.running = False

    # Update all bot data
    def update_data(self):
        # update news
        self.news.update()
        self.last_update = datetime.now()

    # add speech to the queue for consumption
    def queue_speech(self, text):
        self.speech_q.put(text)

    # Speak the given text
    def speak(self, text):
        print text
        # open a speech commanbd subprocess
        self.cur_speech_process = subprocess.Popen('%s %s "%s" %s' %\
         (SPEAK_COMMAND, VOICE, str(text), VOICE_SPEED), shell=True)
        # wait for the process to finish
        self.cur_speech_process.communicate()

    def stop_talking(self):
        print 'stop talking'
        try:
            if self.cur_speech_process:
                self.cur_speech_process.kill()
            if self.cur_alarm_process:
                self.cur_alarm_process.kill()
        except OSError:
            traceback.print_exc()

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
        self.queue_speech('Today\'s headlines')
        # time.sleep(1)
        headlines = [article['title'][0] for article in self.news.articles]

        # read the headlines
        for i, h in enumerate(headlines):
            self.queue_speech(i+1)
            text = h.encode('utf-8').replace('\"', '\\"')
            self.queue_speech(text)
        self.state = States.Speaking

    def read_full_article(self, index):
        self.queue_speech('Reading article ' + str(index+1))
        lines = self.news.articles[index]['snippet']
        for line in lines:
            text = line.encode('utf-8').replace('\"', '\\"')
            self.queue_speech(text)
        self.state = States.Speaking


class NewsReader():
    def __init__(self):
        self.articles = []
        self.load()

    def update(self):
        # crawl the web
        self.crawl()
        self.load()

    def load(self):
        # load the news data file
        self.articles = []
        with open('news.json', 'r') as datafile:
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
        process.crawl('gnews_spider')

        # the script will block here until the crawling is finished
        process.start()

if __name__ == '__main__':
    b = Bot()
    # b.update_data()
    b.run()
    # b.voice_parser.listen()
    # b.read_news()
    # b.set_alarm(datetime.now() + timedelta(seconds=5), 'red.mp3')
    # b.read_full_article(2)


    