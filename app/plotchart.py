from datetime import datetime,timedelta
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column
from bokeh.models import Legend, Title
import pandas as pd
import requests
import dill
import numpy as np
import predictP

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

def get_cw_signals(months):
    df_UR=pd.read_html('https://data.bls.gov/timeseries/LNU04000000')[1]
    df_UR=df_UR.rename(columns={'Unnamed: 0':'Year'}).set_index('Year')
    urthis_yr=[x[0] for x in pd.DataFrame(df_UR.loc[datetime.now().year]).dropna().values]
    urprev_yr=list(df_UR.loc[datetime.now().year-1].values)
    df_CW=pd.DataFrame((urprev_yr+urthis_yr)[-months-1:],columns=['rate'])
    df_CW['prev_rate']=df_CW.rate.shift(1)
    df_CW['change']=df_CW.rate-df_CW.prev_rate
    df_CW['cw_signal']=-np.sign(df_CW.change)
    df_CW=df_CW.dropna()

    cw_signals=list(df_CW.cw_signal)
    return cw_signals

def plotChart(ticker):  
    from datetime import datetime,timedelta
    from math import pi
    
    ticker=ticker.upper()
    weeks=26
   
    today=datetime(datetime.now().year,datetime.now().month,datetime.now().day)
    startDate=str(today-timedelta(weeks=weeks))
    endDate=str(today)

    df_EOD=pd.DataFrame(get_EOD_data(ticker,startDate,endDate).json())
    df_EOD['date']=df_EOD['date'].apply(lambda x:x[:10]).apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
 
    date = df_EOD['date']
    close = df_EOD['adjClose']
    open_=df_EOD['adjOpen']
    high=df_EOD['adjHigh']
    low=df_EOD['adjLow']
    df_EOD['prevClose']=df_EOD['adjClose'].shift(1)
    df_EOD['direction']=np.sign(df_EOD['adjClose']-df_EOD['prevClose'])
    df_EOD=df_EOD.set_index('date')
    highest_high=max(high)

    predictions=predictP.predictP(ticker,df_EOD[['adjClose']],26)
    
    colorchoice=["green" if x>=0 else "red" for x in df_EOD['direction']]
    
    openBarStart=[x-timedelta(hours=15) for x in date]
    closeBarEnd=[x+timedelta(hours=15) for x in date]
    
    df_predictions=pd.DataFrame(
        [x[1] for x in predictions]
        ,[x[0]+timedelta(days=1) for x in predictions],columns=['prediction'])
    df_arrows=df_EOD[['adjHigh','adjLow','adjClose','prevClose','direction']].merge(
        df_predictions,left_on='date',right_index=True,how='outer')
    if len(predictions)>6:
        df_arrows=df_arrows.set_index('date')
    df_arrows['pred_dir']=np.sign(df_arrows.prediction-df_arrows.adjClose.shift(1))
    df_arrows=df_arrows[df_arrows['pred_dir'].notna()].fillna(method='ffill')
    up_arrows=df_arrows[df_arrows['pred_dir']>=0]
    up_arrows['low']=up_arrows.adjHigh*1.01
    up_arrows['high']=up_arrows.adjHigh*1.04
    down_arrows=df_arrows[df_arrows['pred_dir']<0]
    down_arrows['high']=down_arrows.adjLow*0.99
    down_arrows['low']=down_arrows.adjLow*0.96

    if len(predictions)>6:
        df_arrows=df_arrows[:-1]
    arrow_values=list(df_arrows['pred_dir'])
    circle_colors=[]
    for i,d in enumerate(df_arrows.index):

        if df_EOD.loc[d]['direction']==arrow_values[i]:
            circle_colors.append('green')
        else:
            circle_colors.append('red')
    accuracy=circle_colors.count('green')/len(circle_colors)*100
    
    #Create plot
    figTitle='{} DAILY OPEN-HIGH-LOW-CLOSE {} WEEKS / PREDICTION ACCURACY: {}%'.format(ticker,weeks,int(accuracy))
    p=figure(title=figTitle, x_axis_label='Date',plot_width=1000,plot_height=435
        , y_axis_label='Closing Price',x_axis_type='datetime')
    p.title.align='center'
    p.quad(top=high,bottom=low,left=date,right=date, color=colorchoice)
    p.quad(top=open_,bottom=open_,left=openBarStart,right=date, color=colorchoice)
    p.quad(top=close,bottom=close,left=date,right=closeBarEnd, color=colorchoice)
    
    p.circle(df_arrows.index,highest_high,size=10,color=circle_colors)

    p.quad(top=up_arrows['high'],bottom=up_arrows['low']
           ,left=[x-timedelta(hours=15) for x in up_arrows.index]
           ,right=[x+timedelta(hours=15) for x in up_arrows.index],color='green')
    p.quad(top=down_arrows['high'],bottom=down_arrows['low']
           ,left=[x-timedelta(hours=15) for x in down_arrows.index]
           ,right=[x+timedelta(hours=15) for x in down_arrows.index],color='red')

    p.rect(x=up_arrows.index,y=up_arrows['high']
           ,width=10,height=10,color='green',angle=pi/4
           ,height_units='screen',width_units='screen')
    p.rect(x=down_arrows.index,y=down_arrows['low']
           ,width=10,height=10,color='red',angle=pi/4
           ,height_units='screen',width_units='screen')
    p.add_layout(Title(text="The arrows indicate buy and sell signals and the green circles indicate if past signals were correct", align="right",text_color="grey"), "below")


    df_profit=df_arrows.dropna()
    df_profit['nshares']=100000/df_profit.prevClose
    df_profit['EISprofit']=(df_profit.adjClose-df_profit.prevClose)*df_profit.pred_dir*df_profit['nshares']
    df_profit['CWprofit']=(df_profit.adjClose-df_profit.prevClose)*get_cw_signals(int(weeks/4))*df_profit['nshares']

    p1=figure(title='EISentiment v. CONVENTIONAL WISDOM PROFITS FOR {} ON A $100,000 INVESTMENT PER TRADE'.format(ticker)
          , x_axis_label='Date',plot_width=1000,plot_height=435
    , y_axis_label='Closing Price',x_axis_type='datetime')
    p1.title.align='center'

    EISlabel='EISentiment Profits, Total Profits=${}'.format(int(df_profit.EISprofit.sum()))
    CWlabel='Conventional Wisdom Profits=${}'.format(int(df_profit.CWprofit.sum()))

    ei=p1.vbar(x=df_profit.index,top=df_profit['EISprofit']
            ,width=timedelta(days=10),color='green')
    cw=p1.vbar(x=df_profit.index+timedelta(days=10),top=df_profit['CWprofit']
            ,width=timedelta(days=10),color='grey')
    p1.line(x=date,y=0,color='black')
    legend=Legend(items=[(EISlabel,[ei]),(CWlabel,[cw])],location="top_right")
    p1.add_layout(legend,'above')
        
    return column(children=[p,p1],sizing_mode= "scale_width")

    
