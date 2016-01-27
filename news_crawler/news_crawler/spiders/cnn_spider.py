# -*- coding: utf-8 -*-
import scrapy, json
from news_crawler.news_crawler.items import *

class GNewsSpider(scrapy.Spider):
    name = "gnews_spider"
    allowed_domains = ["google.com", 'cnn.com', 'usatoday.com', 'foxnews.com',
        'www.todayonline.com', 'latimes.com', 'seattletimes.com', 'rferl.org',
        'mlive.com', 'huffingtonpost.com', 'nola.com', 'freep.com']

    start_urls = (
        'http://www.news.google.com/',
    )

    def parse(self, response):
        # top_stories_links = response.xpath('//ul[@data-vr-zone="home-top-col2"]/li//a/@href')
        # links = response.css('ul[data-vr-zone="home-top-col2"] a::attr("href")')
        # links = response.css('ul.cn--idx-0 a::attr("href")')
        # links = response.xpath('//ul[@class="cn--idx-0"]//a').extract()
        # links = response.xpath('//script/text()').extract()
        # links = 
        articles = response.css('.section-stream-content .story')
        for article in articles:
            title = article.xpath('.//div[@class="esc-lead-article-title-wrapper"]//span[@class="titletext"]/text()').extract()
            snippet = article.xpath('.//div[@class="esc-lead-snippet-wrapper"]/text()').extract()
            source = article.xpath('.//span[@class="al-attribution-source"]/text()').extract()
            date = article.xpath('.//span[@class="al-attribution-timestamp"]/text()').extract()
            yield GoogleArticleItem(title=title, snippet=snippet, date=date, source=source)

            link = article.css('.esc-lead-article-title a::attr(href)').extract()[0]
            url = response.urljoin(link)
            yield scrapy.Request(url, callback=self.parse_article)

        # print response.body
        # print links
        # for link in links:
        #     url = response.urljoin(link.extract())
        #     yield scrapy.Request(url, callback=self.parse_article)

    def parse_article(self, response):
        print response.url
        # article = ArticleItem()
        # if 'money.cnn.com' in response.url:
        #     # parse money.cnn.com
        #     article['title'] = response.xpath('//h1/text()').extract()
        #     article['body'] = response.xpath('//div[@id="storytext"]/p/text()').extract()
        #     article['date'] = response.xpath('//span[@class="cnnDateStamp"]/text()').extract()
        # elif 'cnn.com/videos' in response.url:
        #     # parse cnn video link
        #     article['title'] = response.xpath('//h1/text()').extract()
        #     article['body'] = response.xpath('//div[@class="media__video-description"]/p/text()').extract()
        #     article['date'] = response.xpath('//p[@class="metadata__data-added"]/text()').extract()
        # else:
        #     # parse default cnn article
        #     article['title'] = response.xpath('//h1/text()').extract()
        #     article['body'] = response.xpath('//div[@class="zn-body__read-all"]/p/text()').extract()
        #     article['developments'] = response.xpath('//div[@itemprop="articleBody"]\
        #         /section/div[@class="l-container"]/p/text()').extract()
        #     article['date'] = response.xpath('//p[@class="update-time"]/text()').extract()
        # yield article