# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 01:11:16 2023

@author: UOBASUB
"""
import csv
import io
from datetime import datetime, timedelta, date
import pandas as pd 
import numpy as np
import math
from google.cloud.storage import Blob
from google.cloud import storage
import plotly.graph_objects as go
from plotly.subplots import make_subplots
np.seterr(divide='ignore', invalid='ignore')
pd.options.mode.chained_assignment = None
from scipy.signal import argrelextrema
from scipy import signal
from scipy.misc import derivative
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import plotly.io as pio
pio.renderers.default='browser'
import bisect
import pandas_ta as ta

def ema(df):
    df['30ema'] = df['close'].ewm(span=30, adjust=False).mean()
    df['40ema'] = df['close'].ewm(span=40, adjust=False).mean()
    df['28ema'] = df['close'].ewm(span=28, adjust=False).mean()
    df['50ema'] = df['close'].ewm(span=50, adjust=False).mean()
    df['100ema'] = df['close'].ewm(span=50, adjust=False).mean()
    df['1ema'] = df['close'].ewm(span=1, adjust=False).mean()

def vwap(df):
    v = df['volume'].values
    h = df['high'].values
    l = df['low'].values
    # print(v)
    df['vwap'] = np.cumsum(v*(h+l)/2) / np.cumsum(v)
    #df['disVWAP'] = (abs(df['close'] - df['vwap']) / ((df['close'] + df['vwap']) / 2)) * 100
    #df['disVWAPOpen'] = (abs(df['open'] - df['vwap']) / ((df['open'] + df['vwap']) / 2)) * 100
    #df['disEMAtoVWAP'] = ((df['close'].ewm(span=12, adjust=False).mean() - df['vwap'])/df['vwap']) * 100

    df['volumeSum'] = df['volume'].cumsum()
    df['volume2Sum'] = (v*((h+l)/2)*((h+l)/2)).cumsum()
    #df['myvwap'] = df['volume2Sum'] / df['volumeSum'] - df['vwap'].values * df['vwap']
    #tp = (df['low'] + df['close'] + df['high']).div(3).values
    # return df.assign(vwap=(tp * v).cumsum() / v.cumsum())


def sigma(df):
    try:
        val = df.volume2Sum / df.volumeSum - df.vwap * df.vwap
    except(ZeroDivisionError):
        val = df.volume2Sum / (df.volumeSum+0.000000000001) - df.vwap * df.vwap
    return math.sqrt(val) if val >= 0 else val


def PPP(df):

    df['STDEV_TV'] = df.apply(sigma, axis=1)
    stdev_multiple_0 = 0.50
    stdev_multiple_1 = 1
    stdev_multiple_1_5 = 1.5
    stdev_multiple_2 = 2.00
    stdev_multiple_25 = 2.50

    df['STDEV_0'] = df.vwap + stdev_multiple_0 * df['STDEV_TV']
    df['STDEV_N0'] = df.vwap - stdev_multiple_0 * df['STDEV_TV']

    df['STDEV_1'] = df.vwap + stdev_multiple_1 * df['STDEV_TV']
    df['STDEV_N1'] = df.vwap - stdev_multiple_1 * df['STDEV_TV']
    
    df['STDEV_15'] = df.vwap + stdev_multiple_1_5 * df['STDEV_TV']
    df['STDEV_N15'] = df.vwap - stdev_multiple_1_5 * df['STDEV_TV']

    df['STDEV_2'] = df.vwap + stdev_multiple_2 * df['STDEV_TV']
    df['STDEV_N2'] = df.vwap - stdev_multiple_2 * df['STDEV_TV']
    
    df['STDEV_25'] = df.vwap + stdev_multiple_25 * df['STDEV_TV']
    df['STDEV_N25'] = df.vwap - stdev_multiple_25 * df['STDEV_TV']


def VMA(df):
    df['vma'] = df['volume'].rolling(4).mean()
      

def historV1(df, num, quodict, trad:list=[], quot:list=[]):
    #trad = AllTrades
    pzie = [(i[0],i[1]) for i in trad]
    dct ={}
    for i in pzie:
        if i[0] not in dct:
            dct[i[0]] =  i[1]
        else:
            dct[i[0]] +=  i[1]
            
    
    pzie = [i for i in dct if dct[i]]#  > 500 list(set(pzie))
    
    hist, bin_edges = np.histogram(pzie, bins=num)
    
    cptemp = []
    zipList = []
    cntt = 0
    for i in range(len(hist)):
        pziCount = 0
        acount = 0
        bcount = 0
        ncount = 0
        for x in trad:
            if bin_edges[i] <= x[0] < bin_edges[i+1]:
                pziCount += (x[1]*x[0])
                if x[4] == 'A':
                    acount += (x[1]*x[0])
                elif x[4] == 'B':
                    bcount += (x[1]*x[0])
                elif x[4] == 'N':
                    ncount += (x[1]*x[0])
                
        #if pziCount > 100:
        cptemp.append([bin_edges[i],pziCount,cntt,bin_edges[i+1]])
        zipList.append([acount,bcount,ncount])
        cntt+=1
        
    for i in cptemp:
        i+=countCandle(trad,quot,i[0],i[3],df['name'][0],quodict)

    for i in range(len(cptemp)):
        cptemp[i] += zipList[i]
        
    
    sortadlist = sorted(cptemp, key=lambda stock: float(stock[1]), reverse=True)
    
    return [cptemp,sortadlist]  
#hs[1]

def countCandle(trad,quot,num1,num2, stkName, quodict):
    enum = ['Bid(SELL)','BelowBid(SELL)','Ask(BUY)','AboveAsk(BUY)','Between']
    color = ['red','darkRed','green','darkGreen','black']

   
    lsr = splitHun(stkName,trad, quot, num1, num2, quodict)
    ind = lsr.index(max(lsr))   #lsr[:4]
    return [enum[ind],color[ind],lsr]


def splitHun(stkName, trad, quot, num1, num2, quodict):
    Bidd = 0
    belowBid = 0
    Askk = 0
    aboveAsk = 0
    Between = 1
    
    return [Bidd,belowBid,Askk,aboveAsk,Between]
 

def valueAreaV1(lst):
    #lst = [i for i in lst if i[1] > 0]
    pocIndex = sorted(lst, key=lambda stock: float(stock[1]), reverse=True)[0][2]
    sPercent = sum([i[1] for i in lst]) * .70
    pocVolume = lst[lst[pocIndex][2]][1]
    #topIndex = pocIndex - 2
    #dwnIndex = pocIndex + 2
    topVol = 0
    dwnVol = 0
    total = pocVolume
    #topBool1 = topBool2 = dwnBool1 = dwnBool2 =True

    if 0 <= pocIndex - 1 and 0 <= pocIndex - 2:
        topVol = lst[lst[pocIndex - 1][2]][1] + lst[lst[pocIndex - 2][2]][1]
        topIndex = pocIndex - 2
        #topBool2 = True
    elif 0 <= pocIndex - 1 and 0 > pocIndex - 2:
        topVol = lst[lst[pocIndex - 1][2]][1]
        topIndex = pocIndex - 1
        #topBool1 = True
    else:
        topVol = 0
        topIndex = pocIndex

    if pocIndex + 1 < len(lst) and pocIndex + 2 < len(lst):
        dwnVol = lst[lst[pocIndex + 1][2]][1] + lst[lst[pocIndex + 2][2]][1]
        dwnIndex = pocIndex + 2
        #dwnBool2 = True
    elif pocIndex + 1 < len(lst) and pocIndex + 2 >= len(lst):
        dwnVol = lst[lst[pocIndex + 1][2]][1]
        dwnIndex = pocIndex + 1
        #dwnBool1 = True
    else:
        dwnVol = 0
        dwnIndex = pocIndex

    # print(pocIndex,topVol,dwnVol,topIndex,dwnIndex)
    while sPercent > total:
        if topVol >= dwnVol:
            total += topVol
            if total > sPercent:
                break

            if 0 <= topIndex - 1 and 0 <= topIndex - 2:
                topVol = lst[lst[topIndex - 1][2]][1] + \
                    lst[lst[topIndex - 2][2]][1]
                topIndex = topIndex - 2

            elif 0 <= topIndex - 1 and 0 > topIndex - 2:
                topVol = lst[lst[topIndex - 1][2]][1]
                topIndex = topIndex - 1

            if topIndex == 0:
                topVol = 0

        else:
            total += dwnVol

            if total > sPercent:
                break

            if dwnIndex + 1 < len(lst) and dwnIndex + 2 < len(lst):
                dwnVol = lst[lst[dwnIndex + 1][2]][1] + \
                    lst[lst[dwnIndex + 2][2]][1]
                dwnIndex = dwnIndex + 2

            elif dwnIndex + 1 < len(lst) and dwnIndex + 2 >= len(lst):
                dwnVol = lst[lst[dwnIndex + 1][2]][1]
                dwnIndex = dwnIndex + 1

            if dwnIndex == len(lst)-1:
                dwnVol = 0

        if dwnIndex == len(lst)-1 and topIndex == 0:
            break
        elif topIndex == 0:
            topVol = 0
        elif dwnIndex == len(lst)-1:
            dwnVol = 0

        # print(total,sPercent,topIndex,dwnIndex,topVol,dwnVol)
        # time.sleep(3)

    return [lst[topIndex][0], lst[dwnIndex][0], lst[pocIndex][0]]

def plotChart(df, lst2, num1, num2, x_fake, df_dx, optionOrderList, stockName='', prevdtstr:str='', sord:list=[], trends:list=[], lstVwap:list=[], bigOrders:list=[], pea:bool=False, timeStamp:int=None, previousDay:bool=False, OptionTimeFrame:list=[], overall:list=[]):
  
    '''
    average = round(np.average(df_dx), 3)
    now = round(df_dx[len(df_dx)-1], 3)
    if average > 0:
        strTrend = "Uptrend"
    elif average < 0:
        strTrend = "Downtrend"
    else:
        strTrend = "No trend!"
    '''
    strTrend = ''
    sortadlist = lst2[1]
    sortadlist2 = lst2[0]
    
    #Tbid = sum([i[6][0] + i[6][1] for i in sortadlist2])
    #Task = sum([i[6][2] + i[6][3] for i in sortadlist2])
    #Tmid = sum([i[6][4] for i in sortadlist2])
    #TBet = sum([i[6][4] for i in sortadlist2])
    
    #strTbid = str(round(Tbid/(Tbid+Task+Tmid),2))
    #strTask = str(round(Task/(Tbid+Task+Tmid),2))
    #strMid = str(round(Tmid/(Tbid+Task+Tmid),2))
    #strTBet = str(round(TBet/(Tbid+Task+TBet),2))
    
    
    putDec = 0
    CallDec = 0
    NumPut = sum([float(i[3]) for i in OptionTimeFrame ])
    NumCall = sum([float(i[2]) for i in OptionTimeFrame])
    if len(OptionTimeFrame) > 0:
        putDec = round(NumPut / sum([float(i[3])+float(i[2]) for i in OptionTimeFrame]),2)
        CallDec = round(NumCall / sum([float(i[3])+float(i[2]) for i in OptionTimeFrame]),2)
        

    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
                        specs=[[{}, {}, ],
                               [{}, {}, ]], #'+ '<br>' +' ( Put:'+str(putDecHalf)+'('+str(NumPutHalf)+') | '+'Call:'+str(CallDecHalf)+'('+str(NumCallHalf)+') '
                        horizontal_spacing=0.02, vertical_spacing=0.03, subplot_titles=(stockName + ' '+strTrend+'('+str('')+')' +' (Sell:'+str(putDec)+' ('+str(round(NumPut,2))+') | '+'Buy:'+str(CallDec)+' ('+str(round(NumCall,2))+') ', 'Volume Profile ' + str(datetime.now()), ),
                         column_widths=[0.7,0.3], row_width=[0.30, 0.70,]) #row_width=[0.15, 0.85,],

    
            
    for pott in OptionTimeFrame:
        pott.insert(4,df['timestamp'].searchsorted(pott[8]))
        #pott.insert(4,df['time'].searchsorted(pott[0]))
        #print(pott)
        
    optColor = [     'green' if float(i[2]) > float(i[3])
                else 'red' if float(i[3]) > float(i[2])
                else 'gray' if float(i[3]) == float(i[2])
                else i for i in OptionTimeFrame]

    fig.add_trace(
        go.Bar(
            x=pd.Series([i[0] for i in OptionTimeFrame]),
            y=pd.Series([float(i[2]) if float(i[2]) > float(i[3]) else float(i[3]) if float(i[3]) > float(i[2]) else 0 for i in OptionTimeFrame]),
            #textposition='auto',
            #orientation='h',
            #width=0.2,
            marker_color=optColor,
            hovertext=pd.Series([i[0]+' '+i[1] for i in OptionTimeFrame]),
            
        ),
        row=2, col=1
    )
        
    fig.add_trace(
        go.Bar(
            x=pd.Series([i[0] for i in OptionTimeFrame]),
            y=pd.Series([float(i[3]) if float(i[2]) > float(i[3]) else float(i[2]) if float(i[3]) > float(i[2]) else 0 for i in OptionTimeFrame]),
            #textposition='auto',
            #orientation='h',
            #width=0.2,
            marker_color= [     'red' if float(i[2]) > float(i[3])
                        else 'green' if float(i[3]) > float(i[2])
                        else 'gray' if float(i[3]) == float(i[2])
                        else i for i in OptionTimeFrame],
            hovertext=pd.Series([i[0]+' '+i[1] for i in OptionTimeFrame]),
            
        ),
        row=2, col=1
    )

    pms = pd.Series([i[2] for i in OptionTimeFrame]).rolling(4).mean()
    cms = pd.Series([i[3] for i in OptionTimeFrame]).rolling(4).mean()
    fig.add_trace(go.Scatter(x=pd.Series([i[0] for i in OptionTimeFrame]), y=pms, line=dict(color='green'), mode='lines', name='Put VMA'), row=2, col=1)
    fig.add_trace(go.Scatter(x=pd.Series([i[0] for i in OptionTimeFrame]), y=cms, line=dict(color='red'), mode='lines', name='Call VMA'), row=2, col=1)
    
    #hovertext = []
    # for i in range(len(df)):
    #strr = df['time'][0]+'\n' +'Open: '+ str(df['open'[0]])+'\n'
    #hovertext.append(str(df.bidAskString[i])+' '+str(df.bidAsk[i]))

    fig.add_trace(go.Candlestick(x=df['time'],
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 # hoverinfo='text',
                                 name="OHLC"),
                  row=1, col=1)
    
    
    
    #fig.add_trace(go.Bar(x=df['time'], y=df['volume'],showlegend=False), row=2, col=1)
    
    #fig.add_trace(go.Scatter(x=df['time'], y=df['vma'], mode='lines', name='VMA'), row=2, col=1)
    
    '''
    for i in range(df.first_valid_index()+1,len(df.index)):
        prev = i - 1
        try:
            if df['superTrend'][i] != df['superTrend'][prev] and not np.isnan(df['superTrend'][i]) :
                #print(i,df['inUptrend'][i])
                fig.add_annotation(x=df['time'][i], y=df['open'][i],
                                   text = 'BUY'  if df['superTrend'][i] else 'SELL',
                                   showarrow=True,
                                   arrowhead=6,
                                   font=dict(
                    #family="Courier New, monospace",
                    size=6,
                    #color="#ffffff"
                ),)  
        except(KeyError):
            pass
    '''
    
    localMin = argrelextrema(df.close.values, np.less_equal, order=100)[0] 
    localMax = argrelextrema(df.close.values, np.greater_equal, order=100)[0]
     
    if len(localMin) > 0:
        mcount = 0 
        for p in localMin:
            fig.add_annotation(x=df['time'][p], y=df['close'][p],
                               text= str(mcount) +'lMin' ,
                               showarrow=True,
                               arrowhead=4,
                               font=dict(
                #family="Courier New, monospace",
                size=10,
                # color="#ffffff"
            ),)
            mcount+=1
    if len(localMax) > 0:
        mcount = 0 
        for b in localMax:
            fig.add_annotation(x=df['time'][b], y=df['close'][b],
                               text=str(mcount) + 'lMax',
                               showarrow=True,
                               arrowhead=4,
                               font=dict(
                #family="Courier New, monospace",
                size=10,
                # color="#ffffff"
            ),)
            mcount+=1
    
    '''
    fig.add_trace(go.Scatter(x=df['time'].iloc[df['time'].searchsorted('09:30:00'):] , y=pd.Series([round(i[2] / (i[2]+i[3]),2) for i in overall]), mode='lines',name='Put Volume'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['time'].iloc[df['time'].searchsorted('09:30:00'):] , y=pd.Series([round(i[3] / (i[2]+i[3]),2) for i in overall]), mode='lines',name='Call Volume'), row=2, col=1)

    
    
    
    fig.add_trace(go.Scatter(x=df['time'], y=df_dx, mode='lines',name='Derivative'), row=2, col=2)
    fig.add_hline(y=0, row=2, col=2, line_color='black')
        
    localDevMin = argrelextrema(df_dx, np.less_equal, order=60)[0] 
    localDevMax = argrelextrema(df_dx, np.greater_equal, order=60)[0]
    
    if len(localDevMin) > 0:
        for p in localDevMin:
            fig.add_annotation(x=x_fake[p], y=df_dx[p],
                               text='<b>' + 'lMin' + '</b>',
                               showarrow=True,
                               arrowhead=4,
                               font=dict(
                #family="Courier New, monospace",
                size=10,
                # color="#ffffff"
            ),row=2, col=2)
            
    if len(localDevMax) > 0:
        for b in localDevMax:
            fig.add_annotation(x=x_fake[b], y=df_dx[b],
                               text='<b>' + 'lMax' + '</b>',
                               showarrow=True,
                               arrowhead=4,
                               font=dict(
                #family="Courier New, monospace",
                size=10,
                # color="#ffffff"
            ),row=2, col=2)
    
    for ps in mlst:
        ps.append(df['time'].searchsorted(ps[6])-1)
    fig.add_trace(go.Scatter(x=pd.Series([i[6] for i in mlst if i[7] < len(df['close'])]), y=pd.Series([i[0] for i in mlst if i[7] < len(df['close'])]), mode='markers',name='TradedPrice',  hovertext=pd.Series([str(i[1]) + ' ' + str(i[5])  for i in mlst if i[7] < len(df['close'])]),), row=2, col=2)
    
    fig.add_trace(go.Scatter(x=pd.Series([i[6] for i in mlst if i[7] < len(df['close'])]), y=pd.Series([df['close'][i[7]] for i in mlst if i[7] < len(df['close'])]), mode='lines',name='ClosingPrice'), row=2, col=2)
    
    #fig.add_trace(go.Scatter(x=pd.Series([df['time'][i[7]] for i in mlst]), y=pd.Series([i[0] for i in mlst]), mode='lines',name='TradedPrice'), row=1, col=1)
    '''
    
    if pea:
        peak, _ = signal.find_peaks(df['100ema'])
        bottom, _ = signal.find_peaks(-df['100ema'])
    
        if len(peak) > 0:
            for p in peak:
                fig.add_annotation(x=df['time'][p], y=df['open'][p],
                                   text='<b>' + 'P' + '</b>',
                                   showarrow=True,
                                   arrowhead=4,
                                   font=dict(
                    #family="Courier New, monospace",
                    size=10,
                    # color="#ffffff"
                ),)
        if len(bottom) > 0:
            for b in bottom:
                fig.add_annotation(x=df['time'][b], y=df['open'][b],
                                   text='<b>' + 'T' + '</b>',
                                   showarrow=True,
                                   arrowhead=4,
                                   font=dict(
                    #family="Courier New, monospace",
                    size=10,
                    # color="#ffffff"
                ),)
    #fig.update_layout(title=df['name'][0])
    fig.update(layout_xaxis_rangeslider_visible=False)
    #lst2 = histor(df)

    

    #sPercent = sum([i[1] for i in adlist]) * .70
    #tp = valueAreaV1(lst2[0])
    

    fig.add_shape(type="rect",
                  y0=num1, y1=num2, x0=-1, x1=len(df),
                  fillcolor="crimson",
                  opacity=0.09,
                  )

    
    '''
    try:
        colors = ['rgba(255,0,0,'+str(round([i[7], i[8], i[9]][[i[7], i[8], i[9]].index(max([i[7], i[8], i[9]]))]/sum([i[7], i[8], i[9]]),2))+')' if [i[7], i[8], i[9]].index(max([i[7], i[8], i[9]])) == 0  #'rgba(255,0,0,'+str(round(i[6][:4][i[6][:4].index(max(i[6][:4]))]/sum(i[6]),2))+')' if i[5] == 'red' 
                  else 'rgba(0,139,139,'+str(round([i[7], i[8], i[9]][[i[7], i[8], i[9]].index(max([i[7], i[8], i[9]]))]/sum([i[7], i[8], i[9]]),2))+')' if [i[7], i[8], i[9]].index(max([i[7], i[8], i[9]])) == 1
                  else '#778899' if [i[7], i[8], i[9]].index(max([i[7], i[8], i[9]])) == 2
                  else 'gray' for i in sortadlist2]#['darkcyan', ] * len(sortadlist2)#
    except(ZeroDivisionError):
        colors = [     'rgba(255,0,0)' if [i[7], i[8], i[9]].index(max([i[7], i[8], i[9]])) == 0
                  else 'rgba(0,139,139)'if [i[7], i[8], i[9]].index(max([i[7], i[8], i[9]])) == 1
                  else '#778899' if [i[7], i[8], i[9]].index(max([i[7], i[8], i[9]])) == 2
                  else 'gray' for i in sortadlist2]
    '''
    colors = []
    for i in sortadlist2:
        thList = [i[7], i[8], i[9]]
        if sum(thList) > 0:
            if thList.index(max(thList)) == 0: 
                colors.append('rgba(255,0,0,'+str(round(thList[0]/sum(thList),2))+')')
            elif thList.index(max(thList)) == 1: 
                colors.append('rgba(0,139,139,'+str(round(thList[1]/sum(thList),2))+')')  
            elif thList.index(max(thList)) == 2: 
                colors.append('#778899')
        elif sum(thList) == 0:
            colors.append('gray')


    
    #print(colors)
    #colors[0] = 'crimson'
    fig.add_trace(
        go.Bar(
            x=pd.Series([i[1] for i in sortadlist2]),
            y=pd.Series([i[0] for i in sortadlist2]),
            text=np.around(pd.Series([i[0] for i in sortadlist2]), 2),
            textposition='auto',
            orientation='h',
            #width=0.2,
            marker_color=colors,
            hovertext=pd.Series([str(round(i[7],3)) + ' ' + str(round(i[8],3))  + ' ' + str(round(i[9],3)) +' ' + str(round([i[7], i[8], i[9]][[i[7], i[8], i[9]].index(max([i[7], i[8], i[9]]))]/sum([i[7], i[8], i[9]]),2)) if sum([i[7], i[8], i[9]]) > 0 
                                 else '' for i in sortadlist2]),
        ),
        row=1, col=2
    )
    #hs[1]  [hs[1][20][7], hs[1][20][8], hs[1][20][9]] [[hs[1][20][7], hs[1][20][8], hs[1][20][9]].index(max([hs[1][20][7], hs[1][20][8], hs[1][20][9]]))]
    '''
    fig.add_trace(
        go.Bar(
            x=pd.Series([i[1] for i in newOpp]),
            y=pd.Series([float(i[0][1:]) for i in newOpp]),
            text=pd.Series([i[0] for i in newOpp]),
            textposition='auto',
            orientation='h',
            #width=0.2,
            marker_color=[     'red' if 'P' in i[0] 
                        else 'green' if 'C' in i[0]
                        else i for i in newOpp],
            hovertext=pd.Series([i[0]  + ' ' + str(i[2]) for i in newOpp]),
        ),
        row=1, col=3
    )
    '''
    
    


    fig.add_trace(go.Scatter(x=[sortadlist2[0][1], sortadlist2[0][1]], y=[
                  num1, num2],  opacity=0.5), row=1, col=2)
    
    
    #fig.add_trace(go.Scatter(x=x_fake, y=df_dx, mode='lines',name='Derivative'), row=2, col=2)
    
    fig.add_trace(go.Scatter(x=df['time'], y=df['vwap'], mode='lines', name='VWAP'))
    
    #if 2 in lstVwap:
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_2'], mode='lines', opacity=0.2, name='UPPERVWAP2', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N2'], mode='lines', opacity=0.2, name='LOWERVWAP2', line=dict(color='black')))
    #if 0 in lstVwap:
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_25'], mode='lines', opacity=0.2, name='UPPERVWAP2.5', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N25'], mode='lines', opacity=0.2, name='LOWERVWAP2.5', line=dict(color='black')))
    #if 1 in lstVwap:    
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_1'], mode='lines', opacity=0.2, name='UPPERVWAP1', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N1'], mode='lines', opacity=0.2, name='LOWERVWAP1', line=dict(color='black')))
        
    #if 1.5 in lstVwap:     
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_15'], mode='lines', opacity=0.2, name='UPPERVWAP1.5', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N15'], mode='lines', opacity=0.2, name='LOWERVWAP1.5', line=dict(color='black')))

    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_0'], mode='lines', opacity=0.2, name='UPPERVWAP0.5', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N0'], mode='lines', opacity=0.2, name='LOWERVWAP0.5', line=dict(color='black')))

    #fig.add_trace(go.Scatter(x=df['time'], y=df['1ema'], mode='lines', opacity=0.19, name='1ema',marker_color='rgba(0,0,0)'))
    
    #fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_2'], mode='lines', name='UPPERVWAP'))
    
    '''
    fig.add_trace(go.Scatter(x=df['time'], y=df['close'], mode='lines',name='Close',marker_color='rgba(0,0,0)'))
    
    
    
    
    
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_15'], mode='lines', name='UPPERVWAP15'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N15'], mode='lines', name='LOWERVWAP15'))
    '''
    '''
    fig.add_trace(
    go.Scatter(
        x=pd.Series([i[1] for i in linePlot]),
        y=pd.Series([i[0] for i in linePlot]),
    ),
    row=1, col=2)
    '''
    #fig.add_trace(go.Scatter(x=df['time'], y=df['1ema'], mode='lines', name='1ema') , row=1, col=1)
    #fig.add_trace(go.Scatter(x=df['time'], y=df['1ema'], mode='lines', name='1ema',line=dict(color="#000000")))
    fig.add_hline(y=df['close'][len(df)-1], row=1, col=2)
    
    
    #fig.add_hline(y=0, row=1, col=4)
    
 
    trcount = 0
    #indsBuy = [(len(df)-1,i[1]) if i[0] >= len(df) else i for i in [(i[10],i[1]) for i in sord if i[11] == stockName and i[1] == 'Ask(BUY)']]
    #indsSell = [(len(df)-1,i[1]) if i[0] >= len(df) else i for i in [(i[10],i[1]) for i in sord if i[11] == stockName and i[1] == 'Bid(SELL)']]
    #indsBetw = [(len(df)-1,i[1]) if i[0] >= len(df) else i for i in [(i[10],i[1]) for i in sord if i[11] == stockName and i[1] == 'Between']]
    putCand = [i for i in OptionTimeFrame if int(i[2]) > int(i[3]) if int(i[4]) < len(df)] # if int(i[4]) < len(df)
    callCand = [i for i in OptionTimeFrame if int(i[3]) > int(i[2]) if int(i[4]) < len(df)] # if int(i[4]) < len(df) +i[3]+i[5] +i[2]+i[5]
    MidCand = [i for i in OptionTimeFrame if int(i[3]) == int(i[2]) if int(i[4]) < len(df)]
    indsAbove = [i for i in OptionTimeFrame if round(i[6],2) > 0.61 and int(i[4]) < len(df) and float(i[2]) >= (sum([i[2]+i[3] for i in OptionTimeFrame]) / len(OptionTimeFrame))] # and int(i[4]) < len(df) [(len(df)-1,i[1]) if i[0] >= len(df) else i for i in [(int(i[10]),i[1]) for i in sord if i[11] == stockName and i[1] == 'AboveAsk(BUY)']]
    
    indsBelow = [i for i in OptionTimeFrame if round(i[7],2) > 0.61 and int(i[4]) < len(df) and float(i[3]) >= (sum([i[3]+i[2] for i in OptionTimeFrame]) / len(OptionTimeFrame))] # and int(i[4]) < len(df) imbalance = [(len(df)-1,i[1]) if i[0] >= len(df) else i for i in [(i[10],i[1]) for i in sord if i[11] == stockName and i[13] == 'Imbalance' and i[1] != 'BelowBid(SELL)' and i[1] != 'AboveAsk(BUY)']]
    #indsHAbove = [(len(df)-1,i[1]) if i[0] >= len(df) else i for i in [(i[10],i[1]) for i in sord if i[11] == stockName and i[1] == 'Ask(BUY)' and float(i[0]) >= 0.40 and int(i[2]) > 160000]]
    #indsHBelow  = [(len(df)-1,i[1]) if i[0] >= len(df) else i for i in [(i[10],i[1]) for i in sord if i[11] == stockName and i[1] == 'Bid(SELL)' and float(i[0]) >= 0.40 and int(i[2]) > 160000]]
    
    if len(MidCand) > 0:
       fig.add_trace(go.Candlestick(
           x=[df['time'][i[4]] for i in MidCand],
           open=[df['open'][i[4]] for i in MidCand],
           high=[df['high'][i[4]] for i in MidCand],
           low=[df['low'][i[4]] for i in MidCand],
           close=[df['close'][i[4]] for i in MidCand],
           increasing={'line': {'color': 'gray'}},
           decreasing={'line': {'color': 'gray'}},
           hovertext=[i[1] for i in MidCand ],
           name='highlight' ),
       row=1, col=1)
       trcount+=1

    if len(putCand) > 0:
        fig.add_trace(go.Candlestick(
            x=[df['time'][i[4]] for i in putCand],
            open=[df['open'][i[4]] for i in putCand],
            high=[df['high'][i[4]] for i in putCand],
            low=[df['low'][i[4]] for i in putCand],
            close=[df['close'][i[4]] for i in putCand],
            increasing={'line': {'color': 'teal'}},
            decreasing={'line': {'color': 'teal'}},
            hovertext=[i[1] for i in putCand ],
            name='highlight' ),
        row=1, col=1)
        trcount+=1
        
    if len(callCand) > 0:
        fig.add_trace(go.Candlestick(
            x=[df['time'][i[4]] for i in callCand],
            open=[df['open'][i[4]] for i in callCand],
            high=[df['high'][i[4]] for i in callCand],
            low=[df['low'][i[4]] for i in callCand],
            close=[df['close'][i[4]] for i in callCand],
            increasing={'line': {'color': 'pink'}},
            decreasing={'line': {'color': 'pink'}},
            hovertext=[i[1] for i in callCand ],
            name='highlight' ),
        row=1, col=1)
        trcount+=1
  
    if len(indsAbove) > 0:
        fig.add_trace(go.Candlestick(
            x=[df['time'][i[4]] for i in indsAbove],
            open=[df['open'][i[4]] for i in indsAbove],
            high=[df['high'][i[4]] for i in indsAbove],
            low=[df['low'][i[4]] for i in indsAbove],
            close=[df['close'][i[4]] for i in indsAbove],
            increasing={'line': {'color': '#00FFFF'}},
            decreasing={'line': {'color': '#00FFFF'}},
            hovertext=['('+str(i[2])+')'+str(round(i[6],2))+' '+str('Bid')+' '+'('+str(i[3])+')'+str(round(i[7],2))+' Ask'  for i in indsAbove], #+i[12].replace('], ', '],<br>')+'<br>'
            name='Bid' ),
        row=1, col=1)
        trcount+=1
    
    if len(indsBelow) > 0:
        fig.add_trace(go.Candlestick(
            x=[df['time'][i[4]] for i in indsBelow],
            open=[df['open'][i[4]] for i in indsBelow],
            high=[df['high'][i[4]] for i in indsBelow],
            low=[df['low'][i[4]] for i in indsBelow],
            close=[df['close'][i[4]] for i in indsBelow],
            increasing={'line': {'color': '#FF1493'}},
            decreasing={'line': {'color': '#FF1493'}},
            hovertext=['('+str(i[2])+')'+str(round(i[6],2))+' '+str('Bid')+' '+'('+str(i[3])+')'+str(round(i[7],2))+' Ask'  for i in indsBelow], #+i[12].replace('], ', '],<br>')+'<br>'
            name='Ask' ),
        row=1, col=1)
        trcount+=1
    
    #for ttt in trends[0]:
        #fig.add_shape(ttt, row=1, col=1)
    

    #fig.add_trace(go.Scatter(x=df['time'], y=df['2ema'], mode='lines', name='2ema'))
    

    fig.add_trace(go.Scatter(x=df['time'], y= [sortadlist2[0][0]]*len(df['time']) ,
                             line_color='orange',
                             text = 'Current Day POC',
                             textposition="bottom left",
                             showlegend=False,
                             visible=False,
                             mode= 'lines',
                            
                            ),
                  row=1, col=1
                 )

    fig.add_trace(go.Scatter(x=df['time'],
                             y= [(num1 + num2)/2]*len(df['time']) ,
                             line_color= '#FF99FF',
                             text = 'VA MidPoint',
                             textposition="bottom left",
                             name='VA MidPoint',
                             showlegend=False,
                             visible=False,
                             mode= 'lines',
                            
                            ),
                  row=1, col=1
                 )

    for v in range(len(sortadlist)):
        res = [0,0,0]
        fig.add_trace(go.Scatter(x=df['time'],
                                 y= [sortadlist[v][0]]*len(df['time']) ,
                                 line_color= 'rgb(0,104,139)' if (str(sortadlist[v][3]) == 'B(SELL)' or str(sortadlist[v][3]) == 'BB(SELL)' or str(sortadlist[v][3]) == 'B') else 'brown' if (str(sortadlist[v][3]) == 'A(BUY)' or str(sortadlist[v][3]) == 'AA(BUY)' or str(sortadlist[v][3]) == 'A') else 'rgb(0,0,0)',
                                 text = str(sortadlist[v][4]) + ' ' + str(sortadlist[v][1]) + ' ' + str(sortadlist[v][3])  + ' ' + str(sortadlist[v][6]),
                                 #text='('+str(priceDict[sortadlist[v][0]]['ASKAVG'])+'/'+str(priceDict[sortadlist[v][0]]['BIDAVG']) +')'+ '('+str(priceDict[sortadlist[v][0]]['ASK'])+'/'+str(priceDict[sortadlist[v][0]]['BID']) +')'+  '('+ sortadlist[v][3] +') '+str(sortadlist[v][4]),
                                 textposition="bottom left",
                                 name=str(sortadlist[v][0]),
                                 showlegend=False,
                                 visible=False,
                                 mode= 'lines',
                                
                                ),
                      row=1, col=1
                     )
        
        
    


    for tmr in range(0,len(fig.data)): 
        fig.data[tmr].visible = True
        
    steps = []
    for i in np.arange(0,len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],
            #label=str(pricelist[i-1])
        )
        for u in range(0,i):
            step["args"][0]["visible"][u] = True
            
        
        step["args"][0]["visible"][i] = True
        steps.append(step)
    
    #print(steps)
    #if previousDay:
        #nummber = 6
    #else:
        #nummber = 0
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Price: "},
        pad={"t": 10},
        steps=steps[17+trcount:]#[8::3]
    )]

    fig.update_layout(
        sliders=sliders
    )
    
    
    
    fig.update_layout(height=800, xaxis_rangeslider_visible=False, showlegend=False)
    #fig.add_trace(go.Scatter(x=df['time'], y=df['BbandsMid'], mode='lines', name='BbandsMid'))
    #fig.add_trace(go.Scatter(x=df['time'], y=df['BbandsUpp'], mode='lines', name='BbandsUpp'))
    #fig.add_trace(go.Scatter(x=df['time'], y=df['BbandsLow'], mode='lines', name='BbandsLow'))


    
    
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=2)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    #fig.update_xaxes(showticklabels=False, row=1, col=2)
    #fig.show(config={'modeBarButtonsToAdd': ['drawline']})
    return fig



'''
31863 == COOPER == HGH4
41512 == GOLD == GCG4
56044 == GAS ==  NGF4
78460 == OIL == CLF4 
260937 == NQ == NQZ3
314863 == ES == ESZ3
'''
symbolNumList = ['17077', '750', '686071', '41512', '56065', '31863', '204839', '75685', '7062']
symbolNameList = ['ESH4','NQH4','CLG4', 'GCG4', 'NGG4', 'HGH4', 'YMH4', 'BTCZ3', 'RTYH4']
#stkName = 'GCG4'

#symbolNum = symbolNumList[symbolNameList.index(stkName)]


from dash import Dash, dcc, html, Input, Output, callback, State
inter = 60000
app = Dash()
app.layout = html.Div([
    
    dcc.Graph(id='graph'),
    dcc.Interval(
        id='interval',
        interval=inter,
        n_intervals=0,
      ),

    html.Div(dcc.Input(id='input-on-submit', type='text')),
    html.Button('Submit', id='submit-val', n_clicks=0),
    html.Div(id='container-button-basic',children="Enter a symbol from |'ESH4' 'NQH4' 'CLG4' 'GCG4' 'NGG4' 'HGH4' 'YMH4' 'BTCZ3' 'RTYH4'| and submit"),
    dcc.Store(id='stkName-value')
])

@callback(
    Output('stkName-value', 'data'),
    Output('container-button-basic', 'children'),
    Input('submit-val', 'n_clicks'),
    State('input-on-submit', 'value'),
    prevent_initial_call=True
)

def update_output(n_clicks, value):
    value = str(value).upper().strip()
    if value in symbolNameList:
        print('The input symbol was "{}" '.format(value))
        return str(value).upper(), str(value).upper()
    else:
        return 'The input symbol was '+str(value)+" is not accepted please try different symbol from  |'ESH4' 'NQH4' 'CLG4' 'GCG4' 'NGG4' 'HGH4' 'YMH4' 'BTCZ3' 'RTYH4'|  ", 'The input symbol was '+str(value)+" is not accepted please try different symbol  |'ESH4' 'NQH4' 'CLG4' 'GCG4' 'NGG4' 'HGH4' 'YMH4' 'BTCZ3' 'RTYH4'|  "

@callback(Output('graph', 'figure'),
          Input('interval', 'n_intervals'),
          State('stkName-value', 'data'))

    
def update_graph_live(n_intervals, data):
    print('inFunction')	

    if data in symbolNameList:
        stkName = data
        symbolNum = symbolNumList[symbolNameList.index(stkName)]
    else:
        stkName = 'ESH4'  
        symbolNum = symbolNumList[symbolNameList.index(stkName)]

    gclient = storage.Client(project="stockapp-401615")
    bucket = gclient.get_bucket("stockapp-storage")
    blob = Blob('FuturesOHLC', bucket) 
    FuturesOHLC = blob.download_as_text()
    
    
    
    csv_reader  = csv.reader(io.StringIO(FuturesOHLC))
    
    csv_rows = []
    for row in csv_reader:
        csv_rows.append(row)
        
    
    
    aggs = [ ]  
    newOHLC = [i for i in csv_rows if i[1] == symbolNum]
    '''
    btemp = [i for i in csv_rows if i[1] == symbolNum]
    if len(btemp) > 2:
        newOHLC = []
        for i in range(0, len(btemp)-1, 2):
            newOHLC.append([btemp[i][0], btemp[i][1], btemp[i][2], str(max(int(btemp[i][3]),int(btemp[i+1][3]))), str(min(int(btemp[i][4]),int(btemp[i+1][4]))), btemp[i+1][5], int(btemp[i][6])+int(btemp[i+1][6])])
    '''
    for i in newOHLC:
        #if int(i[0]) >= 1702508400000000000: 
            #if i[1] == symbolNum:
            hourss = datetime.fromtimestamp(int(int(i[0])// 1000000000)).hour
            if hourss < 10:
                hourss = '0'+str(hourss)
            minss = datetime.fromtimestamp(int(int(i[0])// 1000000000)).minute
            if minss < 10:
                minss = '0'+str(minss)
            opttimeStamp = str(hourss) + ':' + str(minss) + ':00'
            aggs.append([int(i[2])/1e9, int(i[3])/1e9, int(i[4])/1e9, int(i[5])/1e9, int(i[6]), opttimeStamp, int(i[0]), int(i[1])])
            
            
    newAggs = []
    for i in aggs:
        if i not in newAggs:
            newAggs.append(i)
            
          
    df = pd.DataFrame(newAggs, columns = ['open', 'high', 'low', 'close', 'volume', 'time', 'timestamp', 'name',])
    
    
    vwap(df)
    ema(df)
    PPP(df)
    
    
    gclient = storage.Client(project="stockapp-401615")
    bucket = gclient.get_bucket("stockapp-storage")
    blob = Blob('FuturesTrades', bucket) 
    FuturesTrades = blob.download_as_text()
    
    
    
    csv_reader  = csv.reader(io.StringIO(FuturesTrades))
    
    csv_rows = []
    for row in csv_reader:
        csv_rows.append(row)
        
    
    STrades = [i for i in csv_rows if i[4] == symbolNum]
    AllTrades = []
    for i in STrades:
        #if int(i[0]) >= 1702508400000000000: 
            #if i[4] == symbolNum:
            hourss = datetime.fromtimestamp(int(int(i[0])// 1000000000)).hour
            if hourss < 10:
                hourss = '0'+str(hourss)
            minss = datetime.fromtimestamp(int(int(i[0])// 1000000000)).minute
            if minss < 10:
                minss = '0'+str(minss)
            opttimeStamp = str(hourss) + ':' + str(minss) + ':00'
            AllTrades.append([int(i[1])/1e9, int(i[2]), int(i[0]), 0, i[3], opttimeStamp])
            
            
    
    
    hs = historV1(df,50,{},AllTrades,[])
    
    va = valueAreaV1(hs[0])
    
    '''
    x = np.array([i for i in range(len(df))])
    y = np.array([i for i in df['50ema']])
    
    
    
    # Simple interpolation of x and y
    f = interp1d(x, y)
    x_fake = np.arange(0.1, len(df)-1, 1)  #0.10
    
    # derivative of y with respect to x
    df_dx = derivative(f, x_fake, dx=1e-6)
    '''
    
    mTrade = [i for i in AllTrades ]#if i[1] >= 50
    
     
    mTrade = sorted(mTrade, key=lambda d: d[1], reverse=True)
    
    [mTrade[i].insert(4,i) for i in range(len(mTrade))] 
    
    newwT = []
    for i in mTrade:
        newwT.append([i[0],i[1],i[2],i[5], i[4],i[3],i[6]])
    
    #mlst = sorted(mTrade, key=lambda d: d[6], reverse=False) 
    #mlst = [i for i in mlst if i[1] >= 100]
    
    ntList = []
    checkDup = []
    for i in newwT:
        if i[0] not in checkDup:
            ntList.append(i)
    
    dtime = df['time'].values.tolist()
    dtimeEpoch = df['timestamp'].values.tolist()
    
    
    tempTrades = [i for i in AllTrades]
    tempTrades = sorted(tempTrades, key=lambda d: d[6], reverse=False) 
    tradeTimes = [i[6] for i in tempTrades]
    
    timeDict = {}
    for ttm in dtime:
        for tradMade in tempTrades[bisect.bisect_left(tradeTimes, ttm):]:
            if datetime.strptime(tradMade[6], "%H:%M:%S") > datetime.strptime(ttm, "%H:%M:%S") + timedelta(minutes=1):
                try:
                    timeDict[ttm] += [timeDict[ttm][0]/sum(timeDict[ttm]), timeDict[ttm][1]/sum(timeDict[ttm]), timeDict[ttm][2]/sum(timeDict[ttm])]
                except(KeyError,ZeroDivisionError):
                    timeDict[ttm] = [0,0,0]
                break
            
            if ttm not in timeDict:
                timeDict[ttm] = [0,0,0]
            if ttm in timeDict:
                if tradMade[5] == 'B':
                    timeDict[ttm][0] += tradMade[0] * tradMade[1]
                elif tradMade[5] == 'A':
                    timeDict[ttm][1] += tradMade[0] * tradMade[1] 
                elif tradMade[5] == 'N':
                    timeDict[ttm][2] += tradMade[0] * tradMade[1] 
                
    '''
    try:
        timeDict[ttm] += [timeDict[ttm][0]/sum(timeDict[ttm]), timeDict[ttm][1]/sum(timeDict[ttm]), timeDict[ttm][2]/sum(timeDict[ttm])]
    except(ZeroDivisionError):
        timeDict[ttm] += [0, 0,0]
    '''

    for i in timeDict:
        if len(timeDict[i]) == 3:
            try:
                timeDict[i] += [timeDict[i][0]/sum(timeDict[i]), timeDict[i][1]/sum(timeDict[i]), timeDict[ttm][2]/sum(timeDict[ttm])]
            except(ZeroDivisionError):
                timeDict[i] += [0, 0,0]
    
    
    timeFrame = [[i,'']+timeDict[i] for i in timeDict]

    for i in range(len(timeFrame)):
        timeFrame[i].append(dtimeEpoch[i])
    #df['superTrend'] = ta.supertrend(df['high'], df['low'], df['close'], 5, 3.8)['SUPERTd_5_3.8'].replace(-1,0)
    
    fg = plotChart(df, [hs[1],ntList[:2]], va[0], va[1], [], [], bigOrders=[], optionOrderList=[], stockName=symbolNameList[symbolNumList.index(symbolNum)], previousDay=False, prevdtstr='', pea=False, sord = [], OptionTimeFrame = timeFrame, overall=[]) #trends=FindTrends(df,n=10)
        
    return fg


if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8080)
    #app.run_server(debug=False, use_reloader=False)