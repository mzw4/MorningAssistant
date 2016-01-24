# -*- coding: utf-8 -*-
import scrapy, json
from news_crawler.news_crawler.items import ArticleItem

class CnnSpider(scrapy.Spider):
    name = "cnn_spider"
    allowed_domains = ["cnn.com"]
    start_urls = (
        'http://www.cnn.com/',
    )

    def parse(self, response):
        # top_stories_links = response.xpath('//ul[@data-vr-zone="home-top-col2"]/li//a/@href')
        links = response.css('ul[data-vr-zone="home-top-col2"] a::attr("href")')
        print links
        for link in links:
            url = response.urljoin(link.extract())
            yield scrapy.Request(url, callback=self.parse_article)

    def parse_article(self, response):
        article = ArticleItem()
        print response.url
        if 'money.cnn.com' in response.url:
            # parse money.cnn.com
            article['title'] = response.xpath('//h1/text()').extract()
            article['body'] = response.xpath('//div[@id="storytext"]/p/text()').extract()
            article['date'] = response.xpath('//span[@class="cnnDateStamp"]/text()').extract()
        elif 'cnn.com/videos' in response.url:
            # parse cnn video link
            article['title'] = response.xpath('//h1/text()').extract()
            article['body'] = response.xpath('//div[@class="media__video-description"]/p/text()').extract()
            article['date'] = response.xpath('//p[@class="metadata__data-added"]/text()').extract()
        else:
            # parse default cnn article
            article['title'] = response.xpath('//h1/text()').extract()
            article['body'] = response.xpath('//div[@class="zn-body__read-all"]/p/text()').extract()
            article['developments'] = response.xpath('//div[@itemprop="articleBody"]\
                /section/div[@class="l-container"]/p/text()').extract()
            article['date'] = response.xpath('//p[@class="update-time"]/text()').extract()
        yield article