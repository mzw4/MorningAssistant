import json, time, subprocess

class NewsReader():
    def __init__(self):
        self.articles = []
        self.cur_speech_process = None

    def load_articles(self):
        self.articles = []
        with open('news_crawler/news.json') as datafile:
            for line in datafile:
                self.articles.append(json.loads(line))
        
    def get_news(self):
        return [article['title'][0] for article in self.articles]

# system('espeak --voices')