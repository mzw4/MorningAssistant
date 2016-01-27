# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy

class CnnArticleItem(scrapy.Item):
    title = scrapy.Field()
    developments = scrapy.Field()
    body = scrapy.Field()
    date = scrapy.Field()

class GoogleArticleItem(scrapy.Item):
    title = scrapy.Field()
    date = scrapy.Field()
    snippet = scrapy.Field()
    source = scrapy.Field()