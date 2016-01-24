from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

class NewsCrawler():
    def crawl(self):
        # print get_project_settings()['BOT_NAME']
        process = CrawlerProcess({
            'BOT_NAME': 'news_crawler',
            'SPIDER_MODULES': ['news_crawler.news_crawler.spiders'],
            # 'NEWSPIDER_MODULE': 'news_crawler.spiders',
            })

        # crawl cnn.com
        process.crawl('cnn_spider')

        # the script will block here until the crawling is finished
        process.start() 

NewsCrawler().crawl()