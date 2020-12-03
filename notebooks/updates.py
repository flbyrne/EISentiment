import pandas as pd
import numpy as np
import json
import urllib.request
import requests
import datetime
from datetime import datetime, timedelta,date
from bs4 import BeautifulSoup
import re
import dill
from ediblepickle import checkpoint
from retrying import retry
import os

tickers = dill.load(open('data/tickers.pkd', 'rb'))
cache_dir = 'data/BenzNewscache'

@retry(wait_fixed=2000,stop_max_attempt_number=10)
def get_response(path):

    response=requests.get(path)
    if response.ok:
        return response
    else:
        raise NameError
        
@checkpoint(key=lambda args,kwargs:args[0], work_dir=cache_dir)        
def update_ticker_news(ticker,date_last_update):
    url='https://www.benzinga.com/stock-articles/'+ticker+'/news'
    all_headlines={}
    next_page=''
    done=False
    while next_page!=None and not done:
        try:
            print(url)
            newspage=BeautifulSoup(get_response(url).text,'lxml')
        except NameError:
            return None
        news_list=newspage.find_all('div', attrs={'class':'item-list'})
        for item in news_list:
            if item.find('h3')!=None:
                date=datetime.strptime(item.find('h3').text,'%A, %B %d, %Y')
                if date>date_last_update:
                    headlines=item.find_all('span', attrs={'class':'field-content'})
                    for index,headline in enumerate(headlines):
                        headlines[index]=headline.find('a').text
                    all_headlines[date]=headlines
                else:
                    return all_headlines
        next_page=newspage.find('a', attrs={'title':"Go to next page"})
        if next_page==None:
            break
        else:
            url='https://www.benzinga.com'+next_page['href']
    return all_headlines

def update_all_news():
    all_ticker_news = dill.load(open('data/all_ticker_news.pkd', 'rb'))
    for ticker in tickers:
        if all_ticker_news[ticker]:
            date_last_update=max(all_ticker_news[ticker].keys())
            latest_news=update_ticker_news(ticker,date_last_update)
            all_ticker_news[ticker]=dict(
                [x for x in latest_news.items()]+[x for x in all_ticker_news[ticker].items()])
    dill.dump(all_ticker_news, open('data/all_ticker_news.pkd', 'wb'))
    

def get_EOD_data(symbol,sdate,edate):
    
    base_url='https://api.tiingo.com/tiingo/daily/'+symbol+'/prices?'
    api_key='9fa4df1e30e2fb224f1a625b7c43867b1880d05f'
    parameters={
        'token':api_key,
        'startDate':sdate,
        'endDate': edate,
    }
    response = requests.get(base_url, params=parameters)
    return response

@checkpoint(key=lambda args,kwargs:args[1], work_dir=cache_dir)
def get_st_messages(path,filename):
    response=get_response(path)
    x=response.json()
    messages=x['messages']
    d_x={}
    d_x['id'],d_x['message'],d_x['time'],d_x['username'],d_x['score'],d_x['sentiment']=[],[],[],[],[],[]
    for message in messages:
        d_x['id'].append(message['id'])
        d_x['message'].append(message['body'])
        d_x['time'].append(message['created_at'])
        d_x['username'].append(message['user']['username'])
        d_x['score'].append(message['user']['like_count'])
        if message['entities']['sentiment']!=None:
            d_x['sentiment'].append(message['entities']['sentiment']['basic'])
        else:
            d_x['sentiment'].append(None)
    return pd.DataFrame.from_dict(d_x, orient='columns')