import dill
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime,timedelta
import pandas as pd
import requests
from bs4 import BeautifulSoup
import requests
from retrying import retry
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn import base
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def get_UR_prediction():
    date=datetime.now()
    table=pd.read_html('https://tradingeconomics.com/united-states/unemployment-rate')[1]
    dates=list(table['Calendar'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d')))
    table.set_index('Calendar', inplace=True)
    for d in dates:
        if d>=date:
            next_release_date=datetime.strftime(d+timedelta(hours=8.5),'%Y-%m-%d')
            try:
                predicted_rate=float(table.loc[datetime.strftime(d,'%Y-%m-%d'),'TEForecast'][:-1])
            except:
                predicted_rate=np.nan
            break
    return next_release_date,predicted_rate

@retry(wait_fixed=2000,stop_max_attempt_number=10)
def get_response(path):

    response=requests.get(path)
    if response.ok:
        return response
    else:
        raise NameError
        
def get_news(ticker):
    url='https://www.benzinga.com/stock-articles/'+ticker+'/news'
    all_headlines=[]
    while True:
        try:
            newspage=BeautifulSoup(get_response(url).text,'lxml')
        except NameError:
            return None
        news_list=newspage.find_all('div', attrs={'class':'item-list'})
        for item in news_list:
            if item.find('h3')!=None:
                date=datetime.strptime(item.find('h3').text,'%A, %B %d, %Y')
                if date>=datetime(datetime.now().year,datetime.now().month,datetime.now().day)-timedelta(days=2):
                    headlines=item.find_all('span', attrs={'class':'field-content'})
                    headlines=[(date,x.find('a').text )for x in headlines]
                    all_headlines=all_headlines+headlines
                else:
                    return all_headlines
    return all_headlines

def read_word_file(word_type):
    l=[]
    filename=word_type+'_words.txt'
    with open(filename,'r') as wf:
        for line in wf:
            if line[0] != ';' and line[0] != '\n':
                l.append(line.strip())
    return l

l_pos_words=read_word_file('positive')
l_neg_words=read_word_file('negative')

def pos_sentiment(txt):
    word_list=[x.lower() for x in txt.split()]
    return len([x for x in word_list if x in l_pos_words])

def neg_sentiment(txt):
    word_list=[x.lower() for x in txt.split()]
    return -len([x for x in word_list if x in l_neg_words])

class AddTfIdfVect(base.BaseEstimator, base.TransformerMixin):
    def __init__(self):
        self.tfidf=TfidfVectorizer(stop_words=STOP_WORDS.union({'ll', 've'}))
    
    def fit(self, X, y=None):
        self.tfidf.fit(X['NewsText'])
        return self
    
    def transform(self, X):
        X_transformed=pd.DataFrame(self.tfidf.transform(X['NewsText']).todense())
        for x in range(X_transformed.shape[1],1666):
            X_transformed[x]=0
        X_transformed['PosSentiment']=X['PosSentiment'].values
        X_transformed['NegSentiment']=X['NegSentiment'].values
        return X_transformed


def predictP(ticker,latest_closes,weeks):
    

    estimator,mape,prediction_list = dill.load(
        open('predict_data/past_predictions/'+ticker+'.pkd', 'rb'))
    prediction_list=[(x[0].to_pydatetime(),x[1]) for x in prediction_list]
    
    today=datetime(datetime.now().year,datetime.now().month,datetime.now().day)
    release_date,predicted_rate=get_UR_prediction()
    news=get_news(ticker)
    
    if not predicted_rate or release_date!=today+timedelta(days=1) or today!=latest_closes.index[-1] or not news:
        return prediction_list
    
    df_UR=pd.read_html('https://data.bls.gov/timeseries/LNU04000000')[1]
    df_UR=df_UR.rename(columns={'Unnamed: 0':'Year'}).set_index('Year')
    df_UR=pd.DataFrame(pd.DataFrame(df_UR.loc[datetime.now().year]).dropna().iloc[-1])

    current_UR=df_UR.iloc[0,0]
    current_UR_date=df_UR.columns[0]+' '+str(df_UR.index[0])

    startDate=today-timedelta(weeks=weeks)
    
    df_EOD_all = dill.load(open('predict_data/eod/'+ticker+'.pkd', 'rb'))
    df=df_EOD_all[:startDate-timedelta(days=1)][['adjClose']].append(latest_closes)
    df['nextClose']=df['adjClose'].shift(-1)
    arima_model=ARIMA([x for x in df['adjClose']],order=(4,1,0))
    
    arima_prediction=arima_model.fit(disp=0).forecast()[0][0]
    close=df['adjClose'].iloc[-1]
    
    df_news=pd.DataFrame(news,columns=['date','NewsText']).set_index('date')
    df_news['PosSentiment']=df_news['NewsText'].apply(pos_sentiment)
    df_news['NegSentiment']=df_news['NewsText'].apply(neg_sentiment)
    vectorizer=AddTfIdfVect()
    vect_news=vectorizer.fit_transform(df_news[['NewsText','PosSentiment','NegSentiment']])
    news_classifier = dill.load(open('predict_data/news_classifier.pkd', 'rb'))
    df_news['SentimentClass']=news_classifier.predict(vect_news)
    df_news['SentimentIndex']=df_news['SentimentClass']\
    .apply(lambda x: 1 if x=='Positive' else -1 if x=='Negative' else 0)
    
    news_sentiment=sum(df_news['SentimentIndex'])
    
    price_prediction=estimator\
    .predict(np.array([predicted_rate,predicted_rate-current_UR,arima_prediction,news_sentiment])\
             .reshape(1,-1))
    
    prediction_list.append((today,price_prediction[0]))
    
    return prediction_list