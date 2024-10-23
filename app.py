# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 01:11:16 2023

@author: UOBASUB
"""
import csv
import io
from datetime import datetime, timedelta, date, time
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
from collections import defaultdict
from scipy.signal import savgol_filter
#import yfinance as yf
#import dateutil.parser

def ema(df):
    #df['30ema'] = df['close'].ewm(span=30, adjust=False).mean()
    #df['35ema'] = df['close'].ewm(span=35, adjust=False).mean()
    #df['38ema'] = df['close'].ewm(span=38, adjust=False).mean()
    df['40ema'] = df['close'].ewm(span=40, adjust=False).mean()
    #df['28ema'] = df['close'].ewm(span=28, adjust=False).mean()
    #df['25ema'] = df['close'].ewm(span=25, adjust=False).mean()
    #df['23ema'] = df['close'].ewm(span=23, adjust=False).mean()
    #df['50ema'] = df['close'].ewm(span=50, adjust=False).mean()
    #df['15ema'] = df['close'].ewm(span=15, adjust=False).mean()
    #df['20ema'] = df['close'].ewm(span=20, adjust=False).mean()
    #df['10ema'] = df['close'].ewm(span=10, adjust=False).mean()
    #df['100ema'] = df['close'].ewm(span=100, adjust=False).mean()
    #df['150ema'] = df['close'].ewm(span=150, adjust=False).mean()
    #df['200ema'] = df['close'].ewm(span=200, adjust=False).mean()
    #df['2ema'] = df['close'].ewm(span=2, adjust=False).mean()
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
      
'''
def historV1(df, num, quodict, trad:list=[], quot:list=[], rangt:int=1):
    #trad = AllTrades
    pzie = [(i[0],i[1]) for i in trad if i[1] >= rangt]
    dct ={}
    for i in pzie:
        if i[0] not in dct:
            dct[i[0]] =  i[1]
        else:
            dct[i[0]] +=  i[1]
            
    
    pzie = [i for i in dct ]#  > 500 list(set(pzie))
    
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
                pziCount += (x[1])
                if x[4] == 'A':
                    acount += (x[1])
                elif x[4] == 'B':
                    bcount += (x[1])
                elif x[4] == 'N':
                    ncount += (x[1])
                
        #if pziCount > 100:
        cptemp.append([bin_edges[i],pziCount,cntt,bin_edges[i+1]])
        zipList.append([acount,bcount,ncount])
        cntt+=1
        
    for i in cptemp:
        i+=countCandle(trad,[],i[0],i[3],df['name'][0],{})

    for i in range(len(cptemp)):
        cptemp[i] += zipList[i]
        
    
    sortadlist = sorted(cptemp, key=lambda stock: float(stock[1]), reverse=True)
    
    return [cptemp,sortadlist] 
'''
def historV1(df, num, quodict, trad:list=[], quot:list=[]): #rangt:int=1
    #trad = AllTrades
    
    pzie = [(i[0], i[1]) for i in trad]
    dct = defaultdict(int)
    
    for key, value in pzie:
        dct[key] += value
    
    '''
    pzie = [(i[0],i[1]) for i in trad] #rangt:int=1
    dct ={}
    for i in pzie:
        if i[0] not in dct:
            dct[i[0]] =  i[1]
        else:
            dct[i[0]] +=  i[1]
    '''
    
    pzie = [i for i in dct ]#  > 500 list(set(pzie))
    mTradek = sorted(trad, key=lambda d: d[0], reverse=False)
    
    
    hist, bin_edges = np.histogram(pzie, bins=num)
    
    priceList = [i[0] for i in mTradek]

    cptemp = []
    zipList = []
    cntt = 0
    for i in range(len(hist)):
        pziCount = 0
        acount = 0
        bcount = 0
        ncount = 0
        for x in mTradek[bisect.bisect_left(priceList, bin_edges[i]) :  bisect.bisect_left(priceList, bin_edges[i+1])]:
            pziCount += (x[1])
            if x[4] == 'A':
                acount += (x[1])
            elif x[4] == 'B':
                bcount += (x[1])
            elif x[4] == 'N':
                ncount += (x[1])
                
        #if pziCount > 100:
        cptemp.append([bin_edges[i],pziCount,cntt,bin_edges[i+1]])
        zipList.append([acount,bcount,ncount])
        cntt+=1
        
    for i in cptemp:
        i+=countCandle(trad,[],i[0],i[3],df['name'][0],{})

    for i in range(len(cptemp)):
        cptemp[i] += zipList[i]
        
    
    sortadlist = sorted(cptemp, key=lambda stock: float(stock[1]), reverse=True)
    
    return [cptemp,sortadlist]  


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
 
'''
def valueAreaV1(lst):
    lst = [i for i in lst if i[1] > 0]
    for xm in range(len(lst)):
        lst[xm][2] = xm
        
        
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
        if topVol > dwnVol:
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
'''
def valueAreaV1(lst):
    mkk = [i for i in lst if i[1] > 0]
    if len(mkk) == 0:
        mkk = hs[0]
    for xm in range(len(mkk)):
        mkk[xm][2] = xm
        
    pocIndex = sorted(mkk, key=lambda stock: float(stock[1]), reverse=True)[0][2]
    sPercent = sum([i[1] for i in mkk]) * .70
    pocVolume = mkk[mkk[pocIndex][2]][1]
    #topIndex = pocIndex - 2
    #dwnIndex = pocIndex + 2
    topVol = 0
    dwnVol = 0
    total = pocVolume
    #topBool1 = topBool2 = dwnBool1 = dwnBool2 =True

    if 0 <= pocIndex - 1 and 0 <= pocIndex - 2:
        topVol = mkk[mkk[pocIndex - 1][2]][1] + mkk[mkk[pocIndex - 2][2]][1]
        topIndex = pocIndex - 2
        #topBool2 = True
    elif 0 <= pocIndex - 1 and 0 > pocIndex - 2:
        topVol = mkk[mkk[pocIndex - 1][2]][1]
        topIndex = pocIndex - 1
        #topBool1 = True
    else:
        topVol = 0
        topIndex = pocIndex

    if pocIndex + 1 < len(mkk) and pocIndex + 2 < len(mkk):
        dwnVol = mkk[mkk[pocIndex + 1][2]][1] + mkk[mkk[pocIndex + 2][2]][1]
        dwnIndex = pocIndex + 2
        #dwnBool2 = True
    elif pocIndex + 1 < len(mkk) and pocIndex + 2 >= len(mkk):
        dwnVol = mkk[mkk[pocIndex + 1][2]][1]
        dwnIndex = pocIndex + 1
        #dwnBool1 = True
    else:
        dwnVol = 0
        dwnIndex = pocIndex

    # print(pocIndex,topVol,dwnVol,topIndex,dwnIndex)
    while sPercent > total:
        if topVol > dwnVol:
            total += topVol
            if total > sPercent:
                break

            if 0 <= topIndex - 1 and 0 <= topIndex - 2:
                topVol = mkk[mkk[topIndex - 1][2]][1] + \
                    mkk[mkk[topIndex - 2][2]][1]
                topIndex = topIndex - 2

            elif 0 <= topIndex - 1 and 0 > topIndex - 2:
                topVol = mkk[mkk[topIndex - 1][2]][1]
                topIndex = topIndex - 1

            if topIndex == 0:
                topVol = 0

        else:
            total += dwnVol

            if total > sPercent:
                break

            if dwnIndex + 1 < len(mkk) and dwnIndex + 2 < len(mkk):
                dwnVol = mkk[mkk[dwnIndex + 1][2]][1] + \
                    mkk[mkk[dwnIndex + 2][2]][1]
                dwnIndex = dwnIndex + 2

            elif dwnIndex + 1 < len(mkk) and dwnIndex + 2 >= len(mkk):
                dwnVol = mkk[mkk[dwnIndex + 1][2]][1]
                dwnIndex = dwnIndex + 1

            if dwnIndex == len(mkk)-1:
                dwnVol = 0

        if dwnIndex == len(mkk)-1 and topIndex == 0:
            break
        elif topIndex == 0:
            topVol = 0
        elif dwnIndex == len(mkk)-1:
            dwnVol = 0

        # print(total,sPercent,topIndex,dwnIndex,topVol,dwnVol)
        # time.sleep(3)

    return [mkk[topIndex][0], mkk[dwnIndex][0], mkk[pocIndex][0]]


def find_clusters(numbers, threshold):
    clusters = []
    current_cluster = [numbers[0]]

    # Iterate through the numbers
    for i in range(1, len(numbers)):
        # Check if the current number is within the threshold distance from the last number in the cluster
        if abs(numbers[i] - current_cluster[-1]) <= threshold:
            current_cluster.append(numbers[i])
        else:
            # If the current number is outside the threshold, store the current cluster and start a new one
            clusters.append(current_cluster)
            current_cluster = [numbers[i]]

    # Append the last cluster
    clusters.append(current_cluster)
    
    return clusters


def plotChart(df, lst2, num1, num2, x_fake, df_dx,  stockName='', mboString = '',   trends:list=[], pea:bool=False,  previousDay:list=[], OptionTimeFrame:list=[], clusterNum:int=5, troInterval:list=[]):
  
    
    average = round(np.average(df_dx), 3)
    now = round(df_dx[len(df_dx)-1], 3)
    if average > 0:
        strTrend = "Uptrend"
    elif average < 0:
        strTrend = "Downtrend"
    else:
        strTrend = "No trend!"
    
    #strTrend = ''
    sortadlist = lst2[1]
    sortadlist2 = lst2[0]
    

    #buys = [abs(i[2]-i[3]) for i in OptionTimeFrame if i[2]-i[3] > 0 ]
    #sells = [abs(i[2]-i[3]) for i in OptionTimeFrame if i[2]-i[3] < 0 ]

    
    tobuys =  sum([x[1] for x in [i for i in sortadlist if i[3] == 'B']])
    tosells = sum([x[1] for x in [i for i in sortadlist if i[3] == 'A']])
    
    ratio = str(round(max(tosells,tobuys)/min(tosells,tobuys),3))
    
    tpString = ' (Buy:' + str(tobuys) + '('+str(round(tobuys/(tobuys+tosells),2))+') | '+ '(Sell:' + str(tosells) + '('+str(round(tosells/(tobuys+tosells),2))+'))  Ratio : '+str(ratio)+' ' + mboString
    
    '''
    putDec = 0
    CallDec = 0
    NumPut = sum([float(i[3]) for i in OptionTimeFrame[len(OptionTimeFrame)-11:]])
    NumCall = sum([float(i[2]) for i in OptionTimeFrame[len(OptionTimeFrame)-11:]])
    
    thputDec = 0
    thCallDec = 0
    thNumPut = sum([float(i[3]) for i in OptionTimeFrame[len(OptionTimeFrame)-5:]])
    thNumCall = sum([float(i[2]) for i in OptionTimeFrame[len(OptionTimeFrame)-5:]])
    
    if len(OptionTimeFrame) > 0:
        try:
            putDec = round(NumPut / sum([float(i[3])+float(i[2]) for i in OptionTimeFrame[len(OptionTimeFrame)-11:]]),2)
            CallDec = round(NumCall / sum([float(i[3])+float(i[2]) for i in OptionTimeFrame[len(OptionTimeFrame)-11:]]),2)
            
            thputDec = round(thNumPut / sum([float(i[3])+float(i[2]) for i in OptionTimeFrame[len(OptionTimeFrame)-5:]]),2)
            thCallDec = round(thNumCall / sum([float(i[3])+float(i[2]) for i in OptionTimeFrame[len(OptionTimeFrame)-5:]]),2)
        except(ZeroDivisionError):
            putDec = 0
            CallDec = 0
            thputDec = 0
            thCallDec = 0
        
    '''
    fig = make_subplots(rows=3, cols=2, shared_xaxes=True, shared_yaxes=True,
                        specs=[[{}, {},],
                               [{"colspan": 1},{"type": "table", "rowspan": 2},],
                               [{"colspan": 1},{},],], #[{"colspan": 1},{},][{}, {}, ]'+ '<br>' +' ( Put:'+str(putDecHalf)+'('+str(NumPutHalf)+') | '+'Call:'+str(CallDecHalf)+'('+str(NumCallHalf)+') ' (Sell:'+str(sum(sells))+') (Buy:'+str(sum(buys))+') 
                         horizontal_spacing=0.01, vertical_spacing=0.00, subplot_titles=(stockName + ' '+strTrend + '('+str(average)+') '+ str(now)+ ' '+ tpString, 'VP ' + str(datetime.now().time()) ), #' (Sell:'+str(putDec)+' ('+str(round(NumPut,2))+') | '+'Buy:'+str(CallDec)+' ('+str(round(NumCall,2))+') \n '+' (Sell:'+str(thputDec)+' ('+str(round(thNumPut,2))+') | '+'Buy:'+str(thCallDec)+' ('+str(round(thNumCall,2))+') \n '
                         column_widths=[0.80,0.20], row_width=[0.12, 0.12, 0.76,] ) #,row_width=[0.30, 0.70,]

    
            
    
    '''   
    optColor = [     'teal' if float(i[2]) > float(i[3]) #rgba(0,128,0,1.0)
                else 'crimson' if float(i[3]) > float(i[2])#rgba(255,0,0,1.0)
                else 'rgba(128,128,128,1.0)' if float(i[3]) == float(i[2])
                else i for i in OptionTimeFrame]

    fig.add_trace(
        go.Bar(
            x=pd.Series([i[0] for i in OptionTimeFrame]),
            y=pd.Series([float(i[2]) if float(i[2]) > float(i[3]) else float(i[3]) if float(i[3]) > float(i[2]) else float(i[2]) for i in OptionTimeFrame]),
            #textposition='auto',
            #orientation='h',
            #width=0.2,
            marker_color=optColor,
            hovertext=pd.Series([i[0]+' '+i[1] for i in OptionTimeFrame]),
            
        ),
         row=4, col=1
    )
        
    fig.add_trace(
        go.Bar(
            x=pd.Series([i[0] for i in OptionTimeFrame]),
            y=pd.Series([float(i[3]) if float(i[2]) > float(i[3]) else float(i[2]) if float(i[3]) > float(i[2]) else float(i[3]) for i in OptionTimeFrame]),
            #textposition='auto',
            #orientation='h',
            #width=0.2,
            marker_color= [  'crimson' if float(i[2]) > float(i[3]) #rgba(255,0,0,1.0)
                        else 'teal' if float(i[3]) > float(i[2]) #rgba(0,128,0,1.0)
                        else 'rgba(128,128,128,1.0)' if float(i[3]) == float(i[2])
                        else i for i in OptionTimeFrame],
            hovertext=pd.Series([i[0]+' '+i[1] for i in OptionTimeFrame]),
            
        ),
        row=4, col=1
    )

    
    bms = pd.Series([i[2] for i in OptionTimeFrame]).rolling(6).mean()
    sms = pd.Series([i[3] for i in OptionTimeFrame]).rolling(6).mean()
    #xms = pd.Series([i[3]+i[2] for i in OptionTimeFrame]).rolling(4).mean()
    fig.add_trace(go.Scatter(x=pd.Series([i[0] for i in OptionTimeFrame]), y=bms, line=dict(color='teal'), mode='lines', name='Buy VMA'), row=4, col=1)
    fig.add_trace(go.Scatter(x=pd.Series([i[0] for i in OptionTimeFrame]), y=sms, line=dict(color='crimson'), mode='lines', name='Sell VMA'), row=4, col=1)
    '''
    fig.add_trace(go.Candlestick(x=df['time'],
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 # hoverinfo='text',
                                 name="OHLC"),
                  row=1, col=1)
    
    
    
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


    colo = []
    for fk in sortadlist2:
        colo.append([str(round(fk[0],7))+'A',fk[7],fk[8], fk[7]/(fk[7]+fk[8]+fk[9]+1), fk[9]])
        colo.append([str(round(fk[0],7))+'B',fk[8],fk[7], fk[8]/(fk[7]+fk[8]+fk[9]+1), fk[9]])
        colo.append([str(round(fk[0],7))+'N',fk[9],fk[7], fk[9]/(fk[7]+fk[8]+fk[9]+1), fk[8]])
        
    fig.add_trace(
        go.Bar(
            x=pd.Series([i[1] for i in colo]),
            y=pd.Series([float(i[0][:len(i[0])-1]) for i in colo]),
            #text=np.around(pd.Series([float(i[0][:len(i[0])-1]) for i in colo]), 6),
            textposition='auto',
            orientation='h',
            #width=0.2,
            marker_color=[     'teal' if 'B' in i[0] and i[3] < 0.65
                        else '#00FFFF' if 'B' in i[0] and i[3] >= 0.65
                        else 'crimson' if 'A' in i[0] and i[3] < 0.65
                        else 'pink' if 'A' in i[0] and i[3] >= 0.65
                        else 'gray' if 'N' in i[0]
                        else i for i in colo],
            hovertext=pd.Series([i[0][:len(i[0])-1] + ' '+ str(round(i[1] / (i[1]+i[2]+i[4]+1),2)) for i in colo])#  + ' '+ str(round(i[2]/ (i[1]+i[2]+i[4]+1),2))     #pd.Series([str(round(i[7],3)) + ' ' + str(round(i[8],3))  + ' ' + str(round(i[9],3)) +' ' + str(round([i[7], i[8], i[9]][[i[7], i[8], i[9]].index(max([i[7], i[8], i[9]]))]/sum([i[7], i[8], i[9]]),2)) if sum([i[7], i[8], i[9]]) > 0 else '' for i in sortadlist2]),
        ),
        row=1, col=2
    )



    fig.add_trace(go.Scatter(x=[sortadlist2[0][1], sortadlist2[0][1]], y=[
                  num1, num2],  opacity=0.5), row=1, col=2)
    
    
    fig.add_trace(go.Scatter(x=df['time'], y=df['derivative'], mode='lines',name='Derivative'), row=3, col=1)
    fig.add_hline(y=0, row=3, col=1)

    fig.add_trace(go.Scatter(x=df['time'], y=df['vwap'], mode='lines', name='VWAP', line=dict(color='crimson')))
    
    
    if 'POC' in df.columns:
        fig.add_trace(go.Scatter(x=df['time'], y=df['POC'], mode='lines',name='POC',opacity=0.80,marker_color='#0000FF'))
        fig.add_trace(go.Scatter(x=df['time'], y=df['POC2'], mode='lines',name='POC2',opacity=0.80,marker_color='black'))
        #fig.add_trace(go.Scatter(x=df['time'], y=df['POC'].cumsum() / (df.index + 1), mode='lines', opacity=0.50, name='CUMPOC',marker_color='#0000FF'))
        #fig.add_trace(go.Scatter(x=df['time'], y=df['HighVA'], mode='lines', opacity=0.30, name='HighVA',marker_color='rgba(0,0,0)'))
        #fig.add_trace(go.Scatter(x=df['time'], y=df['LowVA'], mode='lines', opacity=0.30,name='LowVA',marker_color='rgba(0,0,0)'))
      
    #fig.add_trace(go.Scatter(x=df['time'], y=df['100ema'], mode='lines', opacity=0.3, name='100ema', line=dict(color='black')))
    #fig.add_trace(go.Scatter(x=df['time'], y=df['150ema'], mode='lines', opacity=0.3, name='150ema', line=dict(color='black')))
    #fig.add_trace(go.Scatter(x=df['time'], y=df['200ema'], mode='lines', opacity=0.3, name='200emaa', line=dict(color='black')))
    
    fig.add_trace(go.Scatter(x=df['time'], y=df['uppervwapAvg'], mode='lines', opacity=0.50,name='uppervwapAvg', ))
    fig.add_trace(go.Scatter(x=df['time'], y=df['lowervwapAvg'], mode='lines',opacity=0.50,name='lowervwapAvg', ))
    fig.add_trace(go.Scatter(x=df['time'], y=df['vwapAvg'], mode='lines', opacity=0.50,name='vwapAvg', ))
    
    
    #fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_2'], mode='lines', opacity=0.1, name='UPPERVWAP2', line=dict(color='black')))
    #fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N2'], mode='lines', opacity=0.1, name='LOWERVWAP2', line=dict(color='black')))

    #fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_25'], mode='lines', opacity=0.15, name='UPPERVWAP2.5', line=dict(color='black')))
    #fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N25'], mode='lines', opacity=0.15, name='LOWERVWAP2.5', line=dict(color='black')))
   
    #fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_1'], mode='lines', opacity=0.1, name='UPPERVWAP1', line=dict(color='black')))
    #fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N1'], mode='lines', opacity=0.1, name='LOWERVWAP1', line=dict(color='black')))
            
    #fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_15'], mode='lines', opacity=0.1, name='UPPERVWAP1.5', line=dict(color='black')))
    #fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N15'], mode='lines', opacity=0.1, name='LOWERVWAP1.5', line=dict(color='black')))

    #fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_0'], mode='lines', opacity=0.1, name='UPPERVWAP0.5', line=dict(color='black')))
    #fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N0'], mode='lines', opacity=0.1, name='LOWERVWAP0.5', line=dict(color='black')))
    
    #fig.add_trace(go.Scatter(x=df['time'], y=df['1ema'], mode='lines', opacity=0.19, name='1ema',marker_color='rgba(0,0,0)'))
    #fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_2'], mode='lines', name='UPPERVWAP'))
    '''
    localMin = argrelextrema(df.low.values, np.less_equal, order=18)[0] 
    localMax = argrelextrema(df.high.values, np.greater_equal, order=18)[0]
     
    if len(localMin) > 0:
        for p in localMin:
            fig.add_annotation(x=df['time'][p], y=df['low'][p],
                               text='<b>' + 'lMin ' + str(df['low'][p]) + '</b>',
                               showarrow=True,
                               arrowhead=4,
                               font=dict(
                #family="Courier New, monospace",
                size=10,
                # color="#ffffff"
            ),)
    if len(localMax) > 0:
        for b in localMax:
            fig.add_annotation(x=df['time'][b], y=df['high'][b],
                               text='<b>' + 'lMax '+ str(df['high'][b]) +  '</b>',
                               showarrow=True,
                               arrowhead=4,
                               font=dict(
                #family="Courier New, monospace",
                size=10,
                # color="#ffffff"
            ),)
            
    
    '''
    fig.add_hline(y=df['close'][len(df)-1], row=1, col=2)
    
    
    #fig.add_hline(y=0, row=1, col=4)
    
 
    trcount = 0
    
    for trd in sortadlist:
        trd.append(df['timestamp'].searchsorted(trd[2])-1)  
        
    for i in OptionTimeFrame:
        try:
            i[10] = []
        except(IndexError):
            i.append([])
            
        
        
    
    for i in sortadlist:
        OptionTimeFrame[i[7]][10].append(i)
        

    putCand = [i for i in OptionTimeFrame if int(i[2]) > int(i[3]) if int(i[4]) < len(df)] # if int(i[4]) < len(df)
    callCand = [i for i in OptionTimeFrame if int(i[3]) > int(i[2]) if int(i[4]) < len(df)] # if int(i[4]) < len(df) +i[3]+i[5] +i[2]+i[5]
    MidCand = [i for i in OptionTimeFrame if int(i[3]) == int(i[2]) if int(i[4]) < len(df)]
    
    tpCandle =  sorted([i for i in OptionTimeFrame if len(i[10]) > 0 if int(i[4]) < len(df)], key=lambda x: sum([trt[1] for trt in x[10]]),reverse=True)[:8] 

    
    '''
    est_now = datetime.utcnow() + timedelta(hours=-4)
    start_time = est_now.replace(hour=8, minute=00, second=0, microsecond=0)
    end_time = est_now.replace(hour=17, minute=30, second=0, microsecond=0)
    
    # Check if the current time is between start and end times
    if start_time <= est_now <= end_time:
        ccheck = 0.64
    else:
    '''
    ccheck = 0.64
    indsAbove = [i for i in OptionTimeFrame if round(i[6],2) >= ccheck and int(i[4]) < len(df) and float(i[2]) >= (sum([i[2]+i[3] for i in OptionTimeFrame]) / len(OptionTimeFrame))]#  float(bms[i[4]])  # and int(i[4]) < len(df) [(len(df)-1,i[1]) if i[0] >= len(df) else i for i in [(int(i[10]),i[1]) for i in sord if i[11] == stockName and i[1] == 'AboveAsk(BUY)']]
    
    indsBelow = [i for i in OptionTimeFrame if round(i[7],2) >= ccheck and int(i[4]) < len(df) and float(i[3]) >= (sum([i[3]+i[2] for i in OptionTimeFrame]) / len(OptionTimeFrame))]#  float(sms[i[4]]) # and int(i[4]) < len(df) imbalance = [(len(df)-1,i[1]) if i[0] >= len(df) else i for i in [(i[10],i[1]) for i in sord if i[11] == stockName and i[13] == 'Imbalance' and i[1] != 'BelowBid(SELL)' and i[1] != 'AboveAsk(BUY)']]

    
        

    '''
    for i in OptionTimeFrame:
        tvy = ''
        for xp in i[10]:
            mks = ''
            for vb in i[10][xp]:
                mks+= str(vb)+' '
            
            tvy += str(xp) +' | ' + mks + '<br>' 
        i.append(tvy)
        #i.append('\n'.join([f'{key}: {value}\n' for key, value in i[10].items()]))
    '''
    #print(OptionTimeFrame[0])
    for i in OptionTimeFrame:
        mks = ''
        tobuyss =  sum([x[1] for x in [t for t in i[10] if t[3] == 'B']])
        tosellss = sum([x[1] for x in [t for t in i[10] if t[3] == 'A']])
        lenbuys = len([t for t in i[10] if t[3] == 'B'])
        lensells = len([t for t in i[10] if t[3] == 'A'])
        
        try:
            tpStrings = '(Sell:' + str(tosellss) + '('+str(round(tosellss/(tobuyss+tosellss),2))+') | '+ '(Buy:' + str(tobuyss) + '('+str(round(tobuyss/(tobuyss+tosellss),2))+')) ' + str(lenbuys+lensells) +' '+  str(tobuyss+tosellss)+'<br>' 
        except(ZeroDivisionError):
            tpStrings =' '
        
        for xp in i[10]:#sorted(i[10], key=lambda x: x[0], reverse=True):
            try:
                taag = 'Buy' if xp[3] == 'B' else 'Sell' if xp[3] == 'A' else 'Mid'
                mks += str(xp[0]) + ' | ' + str(xp[1]) + ' ' + taag + ' ' + str(xp[4]) + ' '+ xp[6] + '<br>' 
            except(IndexError):
                pass
        try:
            i[11] = mks + tpStrings 
        except(IndexError):
            i.append(mks + tpStrings)
    
    
    troAbove = []
    troBelow = []
    
    for tro in tpCandle:
        troBuys = sum([i[1] for i in tro[10] if i[3] == 'B'])
        troSells = sum([i[1] for i in tro[10] if i[3] == 'A'])
        
        try:
            if round(troBuys/(troBuys+troSells),2) >= 0.61:
                troAbove.append(tro+[troBuys, troSells, troBuys/(troBuys+troSells)])
        except(ZeroDivisionError):
            troAbove.append(tro+[troBuys, troSells, 0])
            
        try:
            if round(troSells/(troBuys+troSells),2) >= 0.61:
                troBelow.append(tro+[troSells, troBuys, troSells/(troBuys+troSells)])
        except(ZeroDivisionError):
            troBelow.append(tro+[troSells, troBuys, 0])
    
         

    
    if len(MidCand) > 0:
       fig.add_trace(go.Candlestick(
           x=[df['time'][i[4]] for i in MidCand],
           open=[df['open'][i[4]] for i in MidCand],
           high=[df['high'][i[4]] for i in MidCand],
           low=[df['low'][i[4]] for i in MidCand],
           close=[df['close'][i[4]] for i in MidCand],
           increasing={'line': {'color': 'gray'}},
           decreasing={'line': {'color': 'gray'}},
           hovertext=['('+str(i[2])+')'+str(round(i[6],2))+' '+str('Bid')+' '+'('+str(i[3])+')'+str(round(i[7],2))+' Ask' +  '<br>' +i[11]+ str(i[2]-i[3])   for i in MidCand], #+ i[11] + str(sum([i[10][x][2] for x in i[10]]))
           hoverlabel=dict(
                bgcolor="gray",
                font=dict(color="black", size=10),
                ),
           name='' ),
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
            hovertext=['('+str(i[2])+')'+str(round(i[6],2))+' '+str('Bid')+' '+'('+str(i[3])+')'+str(round(i[7],2))+' Ask' + '<br>' +i[11]+ str(i[2]-i[3]) for i in putCand], #i[11] + str(sum([i[10][x][2] for x in i[10]]))
            hoverlabel=dict(
                 bgcolor="teal",
                 font=dict(color="white", size=10),
                 ),
            name='' ),
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
            hovertext=['('+str(i[2])+')'+str(round(i[6],2))+' '+str('Bid')+' '+'('+str(i[3])+')'+str(round(i[7],2))+' Ask' + '<br>' +i[11]+ str(i[2]-i[3]) for i in callCand], #i[11] + str(sum([i[10][x][2] for x in i[10]]))
            hoverlabel=dict(
                 bgcolor="pink",
                 font=dict(color="black", size=10),
                 ),
            name='' ),
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
            hovertext=['('+str(i[2])+')'+str(round(i[6],2))+' '+str('Bid')+' '+'('+str(i[3])+')'+str(round(i[7],2))+' Ask' + '<br>' +i[11]+ str(i[2]-i[3]) for i in indsAbove], #i[11] + str(sum([i[10][x][2] for x in i[10]]))
            hoverlabel=dict(
                 bgcolor="#00FFFF",
                 font=dict(color="black", size=10),
                 ),
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
            hovertext=['('+str(i[2])+')'+str(round(i[6],2))+' '+str('Bid')+' '+'('+str(i[3])+')'+str(round(i[7],2))+' Ask' + '<br>' +i[11]+ str(i[2]-i[3]) for i in indsBelow], #i[11] + str(sum([i[10][x][2] for x in i[10]]))
            hoverlabel=dict(
                 bgcolor="#FF1493",
                 font=dict(color="white", size=10),
                 ),
            name='Ask' ),
        row=1, col=1)
        trcount+=1
     
    
    if len(tpCandle) > 0:
        fig.add_trace(go.Candlestick(
            x=[df['time'][i[4]] for i in tpCandle],
            open=[df['open'][i[4]] for i in tpCandle],
            high=[df['high'][i[4]] for i in tpCandle],
            low=[df['low'][i[4]] for i in tpCandle],
            close=[df['close'][i[4]] for i in tpCandle],
            increasing={'line': {'color': 'black'}},
            decreasing={'line': {'color': 'black'}},
            hovertext=['('+str(i[2])+')'+str(round(i[6],2))+' '+str('Bid')+' '+'('+str(i[3])+')'+str(round(i[7],2))+' Ask' + '<br>' +i[11]+ str(i[2]-i[3]) for i in tpCandle], #i[11] + str(sum([i[10][x][2] for x in i[10]]))
            hoverlabel=dict(
                 bgcolor="black",
                 font=dict(color="white", size=10),
                 ),
            name='TRO' ),
        row=1, col=1)
        trcount+=1
        
    if len(troAbove) > 0:
        fig.add_trace(go.Candlestick(
            x=[df['time'][i[4]] for i in troAbove],
            open=[df['open'][i[4]] for i in troAbove],
            high=[df['high'][i[4]] for i in troAbove],
            low=[df['low'][i[4]] for i in troAbove],
            close=[df['close'][i[4]] for i in troAbove],
            increasing={'line': {'color': '#16FF32'}},
            decreasing={'line': {'color': '#16FF32'}},
            hovertext=['('+str(i[2])+')'+str(round(i[6],2))+' '+str('Bid')+' '+'('+str(i[3])+')'+str(round(i[7],2))+' Ask' + '<br>' +str(i[11])+ str(i[2]-i[3]) for i in troAbove], #i[11] + str(sum([i[10][x][2] for x in i[10]]))
            hoverlabel=dict(
                 bgcolor="#2CA02C",
                 font=dict(color="white", size=10),
                 ),
            name='TroBuyimbalance' ),
        row=1, col=1)
        trcount+=1
        
    
    if len(troBelow) > 0:
        fig.add_trace(go.Candlestick(
            x=[df['time'][i[4]] for i in troBelow],
            open=[df['open'][i[4]] for i in troBelow],
            high=[df['high'][i[4]] for i in troBelow],
            low=[df['low'][i[4]] for i in troBelow],
            close=[df['close'][i[4]] for i in troBelow],
            increasing={'line': {'color': '#F6222E'}},
            decreasing={'line': {'color': '#F6222E'}},
            hovertext=['('+str(i[2])+')'+str(round(i[6],2))+' '+str('Bid')+' '+'('+str(i[3])+')'+str(round(i[7],2))+' Ask' + '<br>' +str(i[11])+ str(i[2]-i[3]) for i in troBelow], #i[11] + str(sum([i[10][x][2] for x in i[10]]))
            hoverlabel=dict(
                 bgcolor="#F6222E",
                 font=dict(color="white", size=10),
                 ),
            name='TroSellimbalance' ),
        row=1, col=1)
        trcount+=1
    
    #for ttt in trends[0]:
        #fig.add_shape(ttt, row=1, col=1)
    

    #fig.add_trace(go.Scatter(x=df['time'], y=df['2ema'], mode='lines', name='2ema'))
    
    '''
    fig.add_trace(go.Scatter(x=df['time'], y= [sortadlist2[0][0]]*len(df['time']) ,
                             line_color='#0000FF',
                             text = 'Current Day POC',
                             textposition="bottom left",
                             showlegend=False,
                             visible=False,
                             mode= 'lines',
                            
                            ),
                  row=1, col=1
                 )
    '''
    if len(previousDay) > 0:
        if (abs(float(previousDay[2]) - df['1ema'][len(df)-1]) / ((float(previousDay[2]) + df['1ema'][len(df)-1]) / 2)) * 100 <= 0.15:
            fig.add_trace(go.Scatter(x=df['time'],
                                    y= [float(previousDay[2])]*len(df['time']) ,
                                    line_color='cyan',
                                    text = str(previousDay[2]),
                                    textposition="bottom left",
                                    name='Prev POC '+ str(previousDay[2]),
                                    showlegend=False,
                                    visible=False,
                                    mode= 'lines',
                                    
                                    ),
                        row=1, col=1
                        )
            trcount+=1

        if (abs(float(previousDay[0]) - df['1ema'][len(df)-1]) / ((float(previousDay[0]) + df['1ema'][len(df)-1]) / 2)) * 100 <= 0.15:
            fig.add_trace(go.Scatter(x=df['time'],
                                    y= [float(previousDay[0])]*len(df['time']) ,
                                    line_color='green',
                                    text = str(previousDay[0]),
                                    textposition="bottom left",
                                    name='Previous LVA '+ str(previousDay[0]),
                                    showlegend=False,
                                    visible=False,
                                    mode= 'lines',
                                    ),
                        )
            trcount+=1

        if (abs(float(previousDay[1]) - df['1ema'][len(df)-1]) / ((float(previousDay[1]) + df['1ema'][len(df)-1]) / 2)) * 100 <= 0.15:
            fig.add_trace(go.Scatter(x=df['time'],
                                    y= [float(previousDay[1])]*len(df['time']) ,
                                    line_color='purple',
                                    text = str(previousDay[1]),
                                    textposition="bottom left",
                                    name='Previous HVA '+ str(previousDay[1]),
                                    showlegend=False,
                                    visible=False,
                                    mode= 'lines',
                                    ),
                        )
            trcount+=1

    
    '''
    data =  [i[0] for i in sortadlist] #[i for i in df['close']]
    data.sort(reverse=True)
    differences = [abs(data[i + 1] - data[i]) for i in range(len(data) - 1)]
    average_difference = (sum(differences) / len(differences))
    cdata = find_clusters(data, average_difference)
    
    mazz = max([len(i) for i in cdata])
    for i in cdata:
        if len(i) >= clusterNum:
            
            
            bidCount = 0
            askCount = 0
            for x in sortadlist:
                if x[0] >= i[len(i)-1] and x[0] <= i[0]:
                    if x[3] == 'B':
                        bidCount+= x[1]
                    elif x[3] == 'A':
                        askCount+= x[1]

            if bidCount+askCount > 0:       
                askDec = round(askCount/(bidCount+askCount),2)
                bidDec = round(bidCount/(bidCount+askCount),2)
            else:
                askDec = 0
                bidDec = 0
            
            
            
            opac = round((len(i)/mazz)/2,2)
            fig.add_shape(type="rect",
                      y0=i[0], y1=i[len(i)-1], x0=-1, x1=len(df),
                      fillcolor="crimson" if askCount > bidCount else 'teal' if askCount < bidCount else 'gray',
                      opacity=opac)


            
            fig.add_trace(go.Scatter(x=df['time'],
                                 y= [i[0]]*len(df['time']) ,
                                 line_color='rgba(220,20,60,'+str(opac)+')' if askCount > bidCount else 'rgba(0,139,139,'+str(opac)+')' if askCount < bidCount else 'gray',
                                 text =str(i[0])+ ' (' + str(bidCount+askCount)+ ')' +' (' + str(len(i))+ ') Ask:('+ str(askDec) + ') '+str(askCount)+' | Bid: ('+ str(bidDec) +') '+str(bidCount),
                                 textposition="bottom left",
                                 name=str(i[0])+ ' (' + str(bidCount+askCount)+ ')' +' (' + str(len(i))+ ') Ask:('+ str(askDec) + ') '+str(askCount)+' | Bid: ('+ str(bidDec) +') '+str(bidCount),
                                 showlegend=False,
                                 mode= 'lines',
                                
                                ),
                      row=1, col=1)
            trcount+=1

            fig.add_trace(go.Scatter(x=df['time'],
                                 y= [i[len(i)-1]]*len(df['time']) ,
                                 line_color='rgba(220,20,60,'+str(opac)+')' if askCount > bidCount else 'rgba(0,139,139,'+str(opac)+')' if askCount < bidCount else 'gray',
                                 text = str(i[len(i)-1])+ ' (' + str(bidCount+askCount)+ ')' +' (' + str(len(i))+ ') Ask:('+ str(askDec) + ') '+str(askCount)+' | Bid: ('+ str(bidDec) +') '+str(bidCount),
                                 textposition="bottom left",
                                 name= str(i[len(i)-1])+' (' + str(bidCount+askCount)+ ')' + ' (' + str(len(i))+ ') Ask:('+ str(askDec) + ') '+str(askCount)+' | Bid: ('+ str(bidDec) +') '+str(bidCount),
                                 showlegend=False,
                                 mode= 'lines',
                                
                                ),
                      row=1, col=1)
            trcount+=1
    
    #df_dx = np.append(df_dx, df_dx[len(df_dx)-1])
    '''
    '''
    fig.add_trace(go.Scatter(x=df['time'],
                            y= [float(previousDay[3])]*len(df['time']) ,
                            line_color='black',
                            text = str(previousDay[3]),
                            textposition="bottom left",
                            name='Sydney Open',
                            showlegend=False,
                            visible=False,
                            mode= 'lines',
                            ))
    '''
    
    
        
            
        
    
    '''
    difList = [(i[2]-i[3],i[0]) for i in OptionTimeFrame]
    coll = [     'teal' if i[0] > 0
                else 'crimson' if i[0] < 0
                else 'gray' for i in difList]
    fig.add_trace(go.Bar(x=pd.Series([i[1] for i in difList]), y=pd.Series([i[0] for i in difList]), marker_color=coll), row=2, col=1)
    '''
    
    fig.add_trace(go.Bar(x=df['time'], y=df['Histogram'], marker_color='black'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['MACD'], marker_color='blue'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['Signal'], marker_color='red'), row=2, col=1)
    
    #fig.add_trace(go.Bar(x=pd.Series([i[0] for i in troInterval]), y=pd.Series([i[5] for i in troInterval]), marker_color='teal'), row=2, col=1)
    #fig.add_trace(go.Bar(x=pd.Series([i[0] for i in troInterval]), y=pd.Series([i[6] for i in troInterval]), marker_color='crimson'), row=2, col=1)
    
    '''
    if 'POCDistance' in df.columns:
        colors = ['maroon']
        for val in range(1,len(df['POCDistance'])):
            if df['POCDistance'][val] > 0:
                color = 'teal'
                if df['POCDistance'][val] > df['POCDistance'][val-1]:
                    color = '#54C4C1' 
            else:
                color = 'maroon'
                if df['POCDistance'][val] < df['POCDistance'][val-1]:
                    color='crimson' 
            colors.append(color)
        fig.add_trace(go.Bar(x=df['time'], y=df['POCDistance'], marker_color=colors), row=2, col=1)
    '''

    #fig.add_hline(y=0, row=3, col=1)
    #posti = pd.Series([i[0] if i[0] > 0 else 0  for i in difList]).rolling(9).mean()#sum([i[0] for i in difList if i[0] > 0])/len([i[0] for i in difList if i[0] > 0])
    #negati = pd.Series([i[0] if i[0] < 0 else 0 for i in difList]).rolling(9).mean()#sum([i[0] for i in difList if i[0] < 0])/len([i[0] for i in difList if i[0] < 0])
    
    #fig.add_trace(go.Scatter(x=pd.Series([i[0] for i in OptionTimeFrame]), y=posti, line=dict(color='teal'), mode='lines', name='Buy VMA'), row=3, col=1)
    #fig.add_trace(go.Scatter(x=pd.Series([i[0] for i in OptionTimeFrame]), y=negati, line=dict(color='crimson'), mode='lines', name='Sell VMA'), row=3, col=1)
    
    
    #df['Momentum'] = df['Momentum'].fillna(0) ['teal' if val > 0 else 'crimson' for val in df['Momentum']]
    '''
    colors = ['maroon']
    for val in range(1,len(df['Momentum'])):
        if df['Momentum'][val] > 0:
            color = 'teal'
            if df['Momentum'][val] > df['Momentum'][val-1]:
                color = '#54C4C1' 
        else:
            color = 'maroon'
            if df['Momentum'][val] < df['Momentum'][val-1]:
                color='crimson' 
        colors.append(color)
    fig.add_trace(go.Bar(x=df['time'], y=df['Momentum'], marker_color =colors ), row=2, col=1)
    '''
    '''
    coll = [     'teal' if i[2] > 0
                else 'crimson' if i[2] < 0
                else 'gray' for i in troInterval]
    fig.add_trace(go.Bar(x=pd.Series([i[0] for i in troInterval]), y=pd.Series([i[2] for i in troInterval]), marker_color=coll), row=4, col=1)
    
    coll2 = [     'crimson' if i[4] > 0
                else 'teal' if i[4] < 0
                else 'gray' for i in troInterval]
    fig.add_trace(go.Bar(x=pd.Series([i[0] for i in troInterval]), y=pd.Series([i[4] for i in troInterval]), marker_color=coll2), row=4, col=1)
    '''
    #fig.add_trace(go.Scatter(x=pd.Series([i[0] for i in troInterval]), y=pd.Series([i[1] for i in troInterval]), line=dict(color='teal'), mode='lines', name='Buy TRO'), row=4, col=1)
    #fig.add_trace(go.Scatter(x=pd.Series([i[0] for i in troInterval]), y=pd.Series([i[3] for i in troInterval]), line=dict(color='crimson'), mode='lines', name='Sell TRO'), row=4, col=1)
    
    '''
    if len(tpCandle) > 0:
        troRank = []
        for i in tpCandle:
            tobuyss =  sum([x[1] for x in [t for t in i[10] if t[3] == 'B']])
            tosellss = sum([x[1] for x in [t for t in i[10] if t[3] == 'A']])
            troRank.append([tobuyss+tosellss,i[4]])
            
        troRank = sorted(troRank, key=lambda x: x[0], reverse=True)
        
        for i in range(len(troRank)):
            fig.add_annotation(x=df['time'][troRank[i][1]], y=df['high'][troRank[i][1]],
                                   text='<b>' + '['+str(i)+', '+str(troRank[i][0])+']' + '</b>',
                                   showarrow=True,
                                   arrowhead=1,
                                   font=dict(
                    #family="Courier New, monospace",
                    size=10,
                    # color="#ffffff"
                ),)
    '''
        
    '''
    for trds in sortadlist[:1]:
        try:
            if str(trds[3]) == 'A':
                vallue = 'Sell'
                sidev = trds[0]
            elif str(trds[3]) == 'B':
                vallue = 'Buy'
                sidev = trds[0]
            else:
                vallue = 'Mid'
                sidev = df['open'][trds[7]]
            fig.add_annotation(x=df['time'][trds[7]], y=sidev,
                               text= str(trds[4]) + ' ' + str(trds[1]) + ' ' + vallue + ' '+ str(trds[0]) ,
                               showarrow=True,
                               arrowhead=4,
                               font=dict(
                #family="Courier New, monospace",
                size=8,
                # color="#ffffff"
            ),)
        except(KeyError):
            continue 
    '''
    
    
    for tmr in range(0,len(fig.data)): 
        fig.data[tmr].visible = True
    '''    
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
        steps=steps[6+trcount:]#[8::3]
    )]

    fig.update_layout(
        sliders=sliders
    )
    '''
    

    
    # Add a table in the second column
    transposed_data = list(zip(*troInterval[::-1]))
    default_color = "#EBF0F8"  # Default color for all cells
    defaultTextColor = 'black'
    #special_color = "#FFD700"  # Gold color for the highlighted cell
    
    buysideSpikes = find_spikes([i[2] for i in troInterval[::-1]])
    sellsideSpikes = find_spikes([i[4] for i in troInterval[::-1]])
    
    # Create a color matrix for the cells
    color_matrix = [[default_color for _ in range(len(transposed_data[0]))] for _ in range(len(transposed_data))]
    textColor_matrix = [[defaultTextColor for _ in range(len(transposed_data[0]))] for _ in range(len(transposed_data))]
    
    for b in buysideSpikes['high_spikes']:
        color_matrix[2][b[0]] = 'teal'
        #textColor_matrix[2][b[0]] = 'white'
    for b in buysideSpikes['low_spikes']:
        color_matrix[2][b[0]] = 'crimson'
        #textColor_matrix[2][b[0]] = 'white'

    for b in sellsideSpikes['high_spikes']:
        color_matrix[4][b[0]] = 'crimson'
        #textColor_matrix[4][b[0]] = 'white'
    for b in sellsideSpikes['low_spikes']:
        color_matrix[4][b[0]] = 'teal'
        #textColor_matrix[4][b[0]] = 'white'

    
    fig.add_trace(
        go.Table(
            header=dict(values=["Time", "Buyers", "Buyers Change", "Sellers", "Sellers Change",]),
            cells=dict(values=transposed_data, fill_color=color_matrix, font=dict(color=textColor_matrix)),  # Transpose data to fit the table
        ),
        row=2, col=2
    )
    
    '''
    for p in range(len(df)):
        if 'cross_above' in df.columns:
            if df['cross_above'][p]:
                fig.add_annotation(x=df['time'][p], y=df['close'][p],
                                   text='<b>' + 'Buy' + '</b>',
                                   showarrow=True,
                                   arrowhead=4,
                                   font=dict(
                    #family="Courier New, monospace",
                    size=8,
                    # color="#ffffff"
                ),)
         
        if 'cross_below' in df.columns:
            if df['cross_below'][p]:
                fig.add_annotation(x=df['time'][p], y=df['close'][p],
                                   text='<b>' + 'Sell' + '</b>',
                                   showarrow=True,
                                   arrowhead=4,
                                   font=dict(
                    #family="Courier New, monospace",
                    size=8,
                    # color="#ffffff"
                ),)
    
    '''
    for p in range(1, len(df)):  # Start from 1 to compare with the previous row
        if 'cross_above' in df.columns:
            # Check if the value of cross_above changed from the previous row
            if df['cross_above'][p] != df['cross_above'][p-1]:
                # Add annotation only if cross_above is True after the change
                if df['cross_above'][p]:
                    fig.add_annotation(x=df['time'][p], y=df['close'][p],
                                       text='<b>' + 'Buy' + '</b>',
                                       showarrow=True,
                                       arrowhead=4,
                                       font=dict(
                                           size=8,
                                       ),)
        
        if 'cross_below' in df.columns:
            # Check if the value of cross_above changed from the previous row
            if df['cross_below'][p] != df['cross_below'][p-1]:
                # Add annotation only if cross_above is True after the change
                if df['cross_below'][p]:
                    fig.add_annotation(x=df['time'][p], y=df['close'][p],
                                       text='<b>' + 'Sell' + '</b>',
                                       showarrow=True,
                                       arrowhead=4,
                                       font=dict(
                                           size=8,
                                       ),)


    
    
    
    fig.update_layout(height=890, xaxis_rangeslider_visible=False, showlegend=False)
    fig.update_xaxes(autorange="reversed", row=1, col=2)
    #fig.update_xaxes(autorange="reversed", row=1, col=3)
    #fig.update_layout(plot_bgcolor='gray')
    fig.update_layout(paper_bgcolor='#E5ECF6')
    #"paper_bgcolor": "rgba(0, 0, 0, 0)",

    
    
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=2)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=3, col=1)
    #fig.show(config={'modeBarButtonsToAdd': ['drawline']})
    return fig


def find_spikes(data, high_percentile=97, low_percentile=3):
    # Compute the high and low thresholds
    high_threshold = np.percentile(data, high_percentile)
    low_threshold = np.percentile(data, low_percentile)
    
    # Find and collect spikes
    spikes = {'high_spikes': [], 'low_spikes': []}
    for index, value in enumerate(data):
        if value > high_threshold:
            spikes['high_spikes'].append((index, value))
        elif value < low_threshold:
            spikes['low_spikes'].append((index, value))
    
    return spikes

def calculate_bollinger_bands(df):
   df['20sma'] = df['close'].rolling(window=20).mean()
   df['stddev'] = df['close'].rolling(window=20).std()
   df['lower_band'] = df['20sma'] - (2 * df['stddev'])
   df['upper_band'] = df['20sma'] + (2 * df['stddev'])

def calculate_keltner_channels(df):
   df['TR'] = abs(df['high'] - df['low'])
   df['ATR'] = df['TR'].rolling(window=20).mean()

   df['lower_keltner'] = df['20sma'] - (df['ATR'] * 1.5)
   df['upper_keltner'] = df['20sma'] + (df['ATR'] * 1.5)

def calculate_ttm_squeeze(df, n=13):
    '''
    df['20sma'] = df['close'].rolling(window=20).mean()
    highest = df['high'].rolling(window = 20).max()
    lowest = df['low'].rolling(window = 20).min()
    m1 = (highest + lowest)/2 
    df['Momentum'] = (df['close'] - (m1 + df['20sma'])/2)
    fit_y = np.array(range(0,20))
    df['Momentum'] = df['Momentum'].rolling(window = 20).apply(lambda x: np.polyfit(fit_y, x, 1)[0] * (20-1) + np.polyfit(fit_y, x, 1)[1], raw=True)
    
    '''
    #calculate_bollinger_bands(df)
    #calculate_keltner_channels(df)
    #df['Squeeze'] = (df['upper_band'] - df['lower_band']) - (df['upper_keltner'] - df['lower_keltner'])
    #df['Squeeze_On'] = df['Squeeze'] < 0
    #df['Momentum'] = df['close'] - df['close'].shift(20)
    df['20sma'] = df['close'].rolling(window=n).mean()
    highest = df['high'].rolling(window = n).max()
    lowest = df['low'].rolling(window = n).min()
    m1 = (highest + lowest)/2 
    df['Momentum'] = (df['close'] - (m1 + df['20sma'])/2)
    fit_y = np.array(range(0,n))
    df['Momentum'] = df['Momentum'].rolling(window = n).apply(lambda x: np.polyfit(fit_y, x, 1)[0] * (n-1) + np.polyfit(fit_y, x, 1)[1], raw=True)



def calculate_macd(df, short_window=12, long_window=26, signal_window=9, use_avg_price=True):
    """
    Calculate MACD, Signal line, and the histogram using open, high, low, and close prices.
    
    Parameters:
    - df (DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.
    - short_window (int): The short period for the MACD calculation. Default is 12.
    - long_window (int): The long period for the MACD calculation. Default is 26.
    - signal_window (int): The signal period for the MACD signal line. Default is 9.
    - use_avg_price (bool): Whether to use average of OHLC (open, high, low, close) prices instead of just close. Default is False.
    
    Returns:
    - DataFrame: DataFrame with MACD, Signal line, and the histogram.
    """
    
    if use_avg_price:
        df['avg_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        price_col = 'avg_price'
    else:
        price_col = 'close'
    
    # Calculate the short-term and long-term EMAs
    df['ema_short'] = df[price_col].ewm(span=short_window, adjust=False).mean()
    df['ema_long'] = df[price_col].ewm(span=long_window, adjust=False).mean()
    
    # Calculate the MACD line
    df['MACD'] = df['ema_short'] - df['ema_long']
    
    # Calculate the Signal line (9-period EMA of MACD line)
    df['Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    
    # Calculate the MACD histogram (difference between MACD line and Signal line)
    df['Histogram'] = df['MACD'] - df['Signal']
    
    # Separate positive and negative histogram bars
    #df['Positive_Histogram'] = df['Histogram'].apply(lambda x: x if x > 0 else 0)
    #df['Negative_Histogram'] = df['Histogram'].apply(lambda x: x if x < 0 else 0)
    
    # Clean up intermediate columns if needed
    df.drop(['ema_short', 'ema_long'], axis=1, inplace=True)
    
    #return df[['MACD', 'Signal', 'Histogram', 'Positive_Histogram', 'Negative_Histogram']]



from concurrent.futures import ThreadPoolExecutor    
def download_data(bucket_name, blob_name):
    blob = Blob(blob_name, bucket_name)
    return blob.download_as_text()    

#symbolNumList = ['118', '4358', '42012334', '392826', '393','163699', '935', '11232']
#symbolNameList = ['ES', 'NQ', 'YM','CL', 'GC', 'HG', 'NG', 'RTY']

symbolNumList = ['183748', '106364', '42006053', '258644', '393','163699', '923', '42018437', '4127886', '147644', '146415', '39545', '148217']
symbolNameList = ['ES', 'NQ', 'YM','CL', 'GC', 'HG', 'NG', 'RTY', 'PL', '6E', '6J', 'SI', '6A' ]

intList = ['1','2','3','4','5','6','10','15']

vaildClust = [str(i) for i in range(3,200)]

vaildTPO = [str(i) for i in range(10,500)]

gclient = storage.Client(project="stockapp-401615")
bucket = gclient.get_bucket("stockapp-storage")

styles = {
    'main_container': {
        'display': 'flex',
        'flexDirection': 'row',  # Align items in a row
        'justifyContent': 'space-around',  # Space between items
        'flexWrap': 'wrap',  # Wrap items if screen is too small
        #'marginTop': '20px',
        'background': '#E5ECF6',  # Soft light blue background
        'padding': '20px',
        #'borderRadius': '10px'  # Optional: adds rounded corners for better aesthetics
    },
    'sub_container': {
        'display': 'flex',
        'flexDirection': 'column',  # Align items in a column within each sub container
        'alignItems': 'center',
        'margin': '10px'
    },
    'input': {
        'width': '150px',
        'height': '35px',
        'marginBottom': '10px',
        'borderRadius': '5px',
        'border': '1px solid #ddd',
        'padding': '0 10px'
    },
    'button': {
        'width': '100px',
        'height': '35px',
        'borderRadius': '10px',
        'border': 'none',
        'color': 'white',
        'background': '#333333',  # Changed to a darker blue color
        'cursor': 'pointer'
    },
    'label': {
        'textAlign': 'center'
    }
}


#import pandas_ta as ta
#from collections import Counter
from google.api_core.exceptions import NotFound
from dash import Dash, dcc, html, Input, Output, callback, State
initial_inter = 300000  # Initial interval #210000#250000#80001
subsequent_inter = 40000  # Subsequent interval
app = Dash()
app.title = "Initial Title"
app.layout = html.Div([
    
    dcc.Graph(id='graph', config={'modeBarButtonsToAdd': ['drawline']}),
    dcc.Interval(
        id='interval',
        interval=initial_inter,
        n_intervals=0,
      ),
    html.Div([
        html.Div([
            dcc.Input(id='input-on-submit', type='text', style=styles['input']),
            html.Button('Submit', id='submit-val', n_clicks=0, style=styles['button']),
            html.Div(id='container-button-basic', children="Enter a symbol from 'ES', 'NQ', 'YM', 'CL', 'GC', 'HG', 'NG', 'RTY'", style=styles['label']),
        ], style=styles['sub_container']),
        dcc.Store(id='stkName-value'),
        
        html.Div([
            dcc.Input(id='input-on-interv', type='text', style=styles['input']),
            html.Button('Submit', id='submit-interv', n_clicks=0, style=styles['button']),
            html.Div(id='interv-button-basic',children="Enter interval from 5, 10, 15, 30", style=styles['label']),
        ], style=styles['sub_container']),
        dcc.Store(id='interv-value'),
        
        html.Div([
            dcc.Input(id='input-on-cluster', type='text', style=styles['input']),
            html.Button('Submit', id='submit-cluster', n_clicks=0, style=styles['button']),
            html.Div(id='cluster-button-basic',children="Adjust Buy/Sell Signal 3 - 200", style=styles['label']),
        ], style=styles['sub_container']),
        dcc.Store(id='cluster-value'),
        
        html.Div([
            dcc.Input(id='input-on-tpo', type='text', style=styles['input']),
            html.Button('Submit', id='submit-tpo', n_clicks=0, style=styles['button']),
            html.Div(id='tpo-button-basic', children="Enter a top ranked order number from 10 - 500", style=styles['label']),
        ], style=styles['sub_container']),
        dcc.Store(id='tpo-value'),
    ], style=styles['main_container']),
    
    
    dcc.Store(id='data-store'),
    dcc.Store(id='previous-interv'),
    dcc.Store(id='previous-stkName'),
    dcc.Store(id='previous-tpoNum'),
    dcc.Store(id='interval-time', data=initial_inter),
  
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
        return 'The input symbol '+str(value)+" is not accepted please try different symbol from  |'ES', 'NQ',  'YM',  'BTC', 'CL', 'GC'|", 'The input symbol was '+str(value)+" is not accepted please try different symbol  |'ESH4' 'NQH4' 'CLG4' 'GCG4' 'NGG4' 'HGH4' 'YMH4' 'BTCZ3' 'RTYH4'|  "

@callback(
    Output('interv-value', 'data'),
    Output('interv-button-basic', 'children'),
    Input('submit-interv', 'n_clicks'),
    State('input-on-interv', 'value'),
    prevent_initial_call=True
)
def update_interval(n_clicks, value):
    value = str(value)
    
    if value in intList:
        print('The input interval was "{}" '.format(value))
        return str(value), str(value), 
    else:
        return 'The input interval '+str(value)+" is not accepted please try different interval from  |'1' '2' '3' '5' '10' '15'|", 'The input interval '+str(value)+" is not accepted please try different interval from  |'1' '2' '3' '5' '10' '15'|"


@callback(
    Output('cluster-value', 'data'),
    Output('cluster-button-basic', 'children'),
    Input('submit-cluster', 'n_clicks'),
    State('input-on-cluster', 'value'),
    prevent_initial_call=True
)
def update_clusterNum(n_clicks, value):
    value = str(value)
    
    if value in vaildClust:
        print('The input Buy/Sell Signal number was "{}" '.format(value))
        return str(value), str(value), 
    else:
        return 'The Buy/Sell Signal '+str(value)+" is not accepted please try different number from  3 - 20", 'The Buy/Sell Signal '+str(value)+" is not accepted please try different number from  3 - 20"


@callback(
    Output('tpo-value', 'data'),
    Output('tpo-button-basic', 'children'),
    Input('submit-tpo', 'n_clicks'),
    State('input-on-tpo', 'value'),
    prevent_initial_call=True
)
def update_tpo(n_clicks, value):
    value = str(value)
    
    if value in vaildTPO:
        print('The input top rank order was "{}" '.format(value))
        return str(value), str(value), 
    else:
        return 'The input top rank order was '+str(value)+" is not accepted please try different number from  10 - 500", 'The input top rank order '+str(value)+" is not accepted please try different number from  10 - 500"




@callback(
    [Output('data-store', 'data'),
        Output('graph', 'figure'),
        Output('previous-stkName', 'data'),
        Output('previous-interv', 'data'),
        Output('previous-tpoNum', 'data'),
        Output('interval', 'interval')],
    [Input('interval', 'n_intervals')],
    [State('stkName-value', 'data'),
        State('interv-value', 'data'),
        State('data-store', 'data'),
        State('previous-stkName', 'data'),
        State('previous-interv', 'data'),
        State('cluster-value', 'data'),
        State('tpo-value', 'data'),
        State('previous-tpoNum', 'data'),
        State('interval-time', 'data'),
        
    ],
)
    
def update_graph_live(n_intervals, sname, interv, stored_data, previous_stkName, previous_interv, clustNum, tpoNum, previous_tpoNum ,interval_time): #interv
    
    #print(sname, interv, stored_data, previous_stkName)
    #print(interv)

    if sname in symbolNameList:
        stkName = sname
        symbolNum = symbolNumList[symbolNameList.index(stkName)]   
    else:
        stkName = 'NQ' 
        sname = 'NQ'
        symbolNum = symbolNumList[symbolNameList.index(stkName)]
        
    if interv not in intList:
        interv = '3'
        
    if clustNum not in vaildClust:
        clustNum = '50'
        
    if tpoNum not in vaildTPO:
        tpoNum = '100'
        
    if sname != previous_stkName or interv != previous_interv or tpoNum != previous_tpoNum:
        stored_data = None


    
        
        
    print('inFunction '+sname)	
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(download_data, bucket, 'FuturesOHLC' + str(symbolNum)),
            executor.submit(download_data, bucket, 'FuturesTrades' + str(symbolNum))
        ]
        
        # Wait for all downloads to complete
        FuturesOHLC, FuturesTrades = [future.result() for future in futures]
    
    # Process data with pandas directly
    FuturesOHLC = pd.read_csv(io.StringIO(FuturesOHLC), header=None)
    FuturesTrades = pd.read_csv(io.StringIO(FuturesTrades), header=None)
    
    '''
    blob = Blob('FuturesOHLC'+str(symbolNum), bucket) 
    FuturesOHLC = blob.download_as_text()
        

    csv_reader  = csv.reader(io.StringIO(FuturesOHLC))
    
    csv_rows = []
    for row in csv_reader:
        csv_rows.append(row)
        
    
    newOHLC = [i for i in csv_rows]
     
    
    aggs = [ ] 
    for i in FuturesOHLC.values.tolist():
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
    
    '''
    aggs = [ ] 
    for row in FuturesOHLC.itertuples(index=False):
        # Extract values from the row, where row[0] corresponds to the first column, row[1] to the second, etc.
        hourss = datetime.fromtimestamp(int(row[0]) // 1000000000).hour
        hourss = f"{hourss:02d}"  # Ensure it's a two-digit string
        minss = datetime.fromtimestamp(int(row[0]) // 1000000000).minute
        minss = f"{minss:02d}"  # Ensure it's a two-digit string
        
        # Construct the time string
        opttimeStamp = f"{hourss}:{minss}:00"
        
        # Append the transformed row data to the aggs list
        aggs.append([
            row[2] / 1e9,  # Convert the value at the third column (open)
            row[3] / 1e9,  # Convert the value at the fourth column (high)
            row[4] / 1e9,  # Convert the value at the fifth column (low)
            row[5] / 1e9,  # Convert the value at the sixth column (close)
            int(row[6]),   # Volume
            opttimeStamp,  # The formatted timestamp
            int(row[0]),   # Original timestamp
            int(row[1])    # Additional identifier or name
        ])
            
       
    df = pd.DataFrame(aggs, columns = ['open', 'high', 'low', 'close', 'volume', 'time', 'timestamp', 'name',])
    
    df['strTime'] = df['timestamp'].apply(lambda x: pd.Timestamp(int(x) // 10**9, unit='s', tz='EST') )
    
    df.set_index('strTime', inplace=True)
    df['volume'] = pd.to_numeric(df['volume'], downcast='integer')
    df_resampled = df.resample(interv+'min').agg({
        'timestamp': 'first',
        'name': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'time': 'first',
        'volume': 'sum'
    })
    
    df_resampled.reset_index(drop=True, inplace=True)
    
    df = df_resampled
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True) 

    
    
    

    vwap(df)
    ema(df)
    PPP(df)
    df['uppervwapAvg'] = df['STDEV_25'].cumsum() / (df.index + 1)
    df['lowervwapAvg'] = df['STDEV_N25'].cumsum() / (df.index + 1)
    df['vwapAvg'] = df['vwap'].cumsum() / (df.index + 1)

    
    
    

    '''
    blob = Blob('FuturesTrades'+str(symbolNum), bucket) 
    FuturesTrades = blob.download_as_text()
    
    
    csv_reader  = csv.reader(io.StringIO(FuturesTrades))
    
    csv_rows = []
    for row in csv_reader:
        csv_rows.append(row)
       

    #STrades = [i for i in csv_rows]
    start_time_itertuples = time.time()
    AllTrades = []
    for i in FuturesTrades.values.tolist():
        hourss = datetime.fromtimestamp(int(int(i[0])// 1000000000)).hour
        if hourss < 10:
            hourss = '0'+str(hourss)
        minss = datetime.fromtimestamp(int(int(i[0])// 1000000000)).minute
        if minss < 10:
            minss = '0'+str(minss)
        opttimeStamp = str(hourss) + ':' + str(minss) + ':00'
        AllTrades.append([int(i[1])/1e9, int(i[2]), int(i[0]), 0, i[3], opttimeStamp])
    time_itertuples = time.time() - start_time_itertuples
       
    #AllTrades = [i for i in AllTrades if i[1] > 1]
    '''

    AllTrades = []
    for row in FuturesTrades.itertuples(index=False):
        hourss = datetime.fromtimestamp(int(row[0]) // 1000000000).hour
        hourss = f"{hourss:02d}"  # Ensure two-digit format for hours
        minss = datetime.fromtimestamp(int(row[0]) // 1000000000).minute
        minss = f"{minss:02d}"  # Ensure two-digit format for minutes
        opttimeStamp = f"{hourss}:{minss}:00"
        AllTrades.append([int(row[1]) / 1e9, int(row[2]), int(row[0]), 0, row[3], opttimeStamp])
    
        
    hs = historV1(df,100,{},AllTrades,[])
    
    va = valueAreaV1(hs[0])
    
    

    x = np.array([i for i in range(len(df))])
    y = np.array([i for i in df['40ema']])
    
    

    # Simple interpolation of x and y
    f = interp1d(x, y)
    x_fake = np.arange(0.1, len(df)-1, 1)

    # derivative of y with respect to x
    df_dx = derivative(f, x_fake, dx=1e-6)
    df_dx = np.pad(df_dx, (1, 0), 'edge')
    #df['derivative'] = np.gradient(df['40ema'])
    
    #df['avg_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df[clustNum+'ema'] = df['close'].ewm(span=int(clustNum), adjust=False).mean()
    
    # Define window size and polynomial order
    window_size = 11 # Must be odd
    poly_order = 2
    # Apply Savitzky-Golay filter to compute the first derivative
    df['derivative'] = savgol_filter(df[clustNum+'ema'], window_length=window_size, polyorder=poly_order, deriv=1)
    
     
    mTrade = sorted(AllTrades, key=lambda d: d[1], reverse=True)
    
    [mTrade[i].insert(4,i) for i in range(len(mTrade))] 
    
    newwT = []
    for i in mTrade:
        newwT.append([i[0],i[1],i[2],i[5], i[4],i[3],i[6]])
    
    
    dtime = df['time'].dropna().values.tolist()
    dtimeEpoch = df['timestamp'].dropna().values.tolist()
    
    
    #tempTrades = [i for i in AllTrades]
    tempTrades = sorted(AllTrades, key=lambda d: d[6], reverse=False) 
    #tradeTimes = [i[6] for i in AllTrades]
    tradeEpoch = [i[2] for i in AllTrades]
    
    
    if stored_data is not None:
        print('NotNew')
        startIndex = next(iter(df.index[df['time'] == stored_data['timeFrame'][len(stored_data['timeFrame'])-1][0]]), None)#df['timestamp'].searchsorted(stored_data['timeFrame'][len(stored_data['timeFrame'])-1][9])
        timeDict = {}
        make = []
        for ttm in range(startIndex,len(dtimeEpoch)):
            
            make.append([dtimeEpoch[ttm],dtime[ttm],bisect.bisect_left(tradeEpoch, dtimeEpoch[ttm])])
            timeDict[dtime[ttm]] = [0,0,0]
            
        for tr in range(len(make)):
            if tr+1 < len(make):
                tempList = AllTrades[make[tr][2]:make[tr+1][2]]
            else:
                tempList = AllTrades[make[tr][2]:len(AllTrades)]
            for i in tempList:
                if i[5] == 'B':
                    timeDict[make[tr][1]][0] += i[1]
                elif i[5] == 'A':
                    timeDict[make[tr][1]][1] += i[1] 
                elif i[5] == 'N':
                    timeDict[make[tr][1]][2] += i[1]
            try:    
                timeDict[make[tr][1]] += [timeDict[make[tr][1]][0]/sum(timeDict[make[tr][1]]), timeDict[make[tr][1]][1]/sum(timeDict[make[tr][1]]), timeDict[make[tr][1]][2]/sum(timeDict[make[tr][1]])]   
            except(ZeroDivisionError):
                timeDict[make[tr][1]]  += [0,0,0] 
                
        timeFrame = [[i,'']+timeDict[i] for i in timeDict]
    
        for i in range(len(timeFrame)):
            timeFrame[i].append(dtimeEpoch[startIndex+i])
            
        for pott in timeFrame:
            #print(pott)
            pott.insert(4,df['timestamp'].searchsorted(pott[8]))
            
            
        stored_data['timeFrame'] = stored_data['timeFrame'][:len(stored_data['timeFrame'])-1] + timeFrame
        
        bful = []
        for it in range(len(make)):
            if it+1 < len(make):
                tempList = AllTrades[0:make[it+1][2]]
            else:
                tempList = AllTrades
            #print(make[0][2],make[it+1][2], len(tempList))
            nelist = sorted(tempList, key=lambda d: d[1], reverse=True)[:int(tpoNum)]
            
            timestamp_s = make[it][0] / 1_000_000_000
            new_timestamp_s = timestamp_s + (int(interv)*60)
            new_timestamp_ns = int(new_timestamp_s * 1_000_000_000)

            bful.append([make[it][1], sum([i[1] for i in nelist if i[5] == 'B']), sum([i[1] for i in nelist if i[5] == 'A']),  sum([i[1] for i in nelist if (i[2] >= make[it][0] and i[2] <= new_timestamp_ns) and i[5] == 'B']), sum([i[1] for i in nelist if (i[2] >= make[it][0] and i[2] <= new_timestamp_ns) and i[5] == 'A']) ])
            
            
  
        
        dst = [[bful[row][0], bful[row][1], 0, bful[row][2], 0, bful[row][3], bful[row][4]] for row in  range(len(bful))]
        
        stored_data['tro'] = stored_data['tro'][:len(stored_data['tro'])-1] + dst
        
        bolist = [0]
        for i in range(len(stored_data['tro'])-1):
            bolist.append(stored_data['tro'][i+1][1] - stored_data['tro'][i][1])
            
        solist = [0] 
        for i in range(len(stored_data['tro'])-1):
            solist.append(stored_data['tro'][i+1][3] - stored_data['tro'][i][3])
            
        newst = [[stored_data['tro'][i][0], stored_data['tro'][i][1], bolist[i], stored_data['tro'][i][3], solist[i], stored_data['tro'][i][5], stored_data['tro'][i][6]] for i in range(len(stored_data['tro']))]
        
        stored_data['tro'] = newst
            
    
    
    if stored_data is None:
        print('Newstored')
        timeDict = {}
        make = []
        for ttm in range(len(dtimeEpoch)):
            
            make.append([dtimeEpoch[ttm],dtime[ttm],bisect.bisect_left(tradeEpoch, dtimeEpoch[ttm])]) #min(range(len(tradeEpoch)), key=lambda i: abs(tradeEpoch[i] - dtimeEpoch[ttm]))
            timeDict[dtime[ttm]] = [0,0,0]
            
            
        
        for tr in range(len(make)):
            if tr+1 < len(make):
                tempList = AllTrades[make[tr][2]:make[tr+1][2]]
            else:
                tempList = AllTrades[make[tr][2]:len(AllTrades)]
            for i in tempList:
                if i[5] == 'B':
                    timeDict[make[tr][1]][0] += i[1]
                elif i[5] == 'A':
                    timeDict[make[tr][1]][1] += i[1] 
                elif i[5] == 'N':
                    timeDict[make[tr][1]][2] += i[1]
            try:    
                timeDict[make[tr][1]] += [timeDict[make[tr][1]][0]/sum(timeDict[make[tr][1]]), timeDict[make[tr][1]][1]/sum(timeDict[make[tr][1]]), timeDict[make[tr][1]][2]/sum(timeDict[make[tr][1]])]   
            except(ZeroDivisionError):
                timeDict[make[tr][1]]  += [0,0,0] 
    
                          
        timeFrame = [[i,'']+timeDict[i] for i in timeDict]
    
        for i in range(len(timeFrame)):
            timeFrame[i].append(dtimeEpoch[i])
            
        for pott in timeFrame:
            #print(pott)
            pott.insert(4,df['timestamp'].searchsorted(pott[8]))
            
        
        bful = []
        for it in range(len(make)):
            if it+1 < len(make):
                tempList = AllTrades[0:make[it+1][2]]
            else:
                tempList = AllTrades
            nelist = sorted(tempList, key=lambda d: d[1], reverse=True)[:int(tpoNum)]
            timestamp_s = make[it][0] / 1_000_000_000
            new_timestamp_s = timestamp_s + (int(interv)*60)
            new_timestamp_ns = int(new_timestamp_s * 1_000_000_000)

            bful.append([make[it][1], sum([i[1] for i in nelist if i[5] == 'B']), sum([i[1] for i in nelist if i[5] == 'A']),  sum([i[1] for i in nelist if (i[2] >= make[it][0] and i[2] <= new_timestamp_ns) and i[5] == 'B']), sum([i[1] for i in nelist if (i[2] >= make[it][0] and i[2] <= new_timestamp_ns) and i[5] == 'A']) ])
            
        bolist = [0]
        for i in range(len(bful)-1):
            bolist.append(bful[i+1][1] - bful[i][1])
            
        solist = [0]
        for i in range(len(bful)-1):
            solist.append(bful[i+1][2] - bful[i][2])
            #buyse/sellle
            
        
        dst = [[bful[row][0], bful[row][1], bolist[row], bful[row][2], solist[row], bful[row][3], bful[row][4]] for row in  range(len(bful))]
            
        stored_data = {'timeFrame': timeFrame, 'tro':dst} 
        
    
    
    
    
        
    #OptionTimeFrame = stored_data['timeFrame']   
    previous_stkName = sname
    previous_interv = interv
    previous_tpoNum = tpoNum

         
    
    #df['superTrend'] = ta.supertrend(df['high'], df['low'], df['close'], length=2, multiplier=1.8)['SUPERTd_2_1.8']
    #df['superTrend'][df['superTrend'] < 0] = 0
 
    blob = Blob('PrevDay', bucket) 
    PrevDay = blob.download_as_text()
        

    csv_reader  = csv.reader(io.StringIO(PrevDay))

    csv_rows = []
    for row in csv_reader:
        csv_rows.append(row)
        
    try:   
        previousDay = [csv_rows[[i[4] for i in csv_rows].index(symbolNum)][0], csv_rows[[i[4] for i in csv_rows].index(symbolNum)][1], csv_rows[[i[4] for i in csv_rows].index(symbolNum)][2], csv_rows[[i[4] for i in csv_rows].index(symbolNum)][6] ]
    except(ValueError):
        previousDay = []
    
    calculate_macd(df)
    try:
        
        blob = Blob('POCData'+str(symbolNum), bucket) 
        POCData = blob.download_as_text()
        csv_reader  = csv.reader(io.StringIO(POCData))

        csv_rows = []
        for row in csv_reader:
            csv_rows.append(row)
        
        LowVA = [float(i[0]) for i in csv_rows]
        HighVA = [float(i[1]) for i in csv_rows]
        POC = [float(i[2]) for i in csv_rows]
        pocc = [float(i[5]) for i in csv_rows]
        #if len(LowVA) > 0:
        if len(df) >= len(POC) and len(POC) > 0:
            df['LowVA'] = pd.Series(LowVA + [LowVA[len(LowVA)-1]]*(len(df)-len(LowVA)))
            df['HighVA'] = pd.Series(HighVA + [HighVA[len(HighVA)-1]]*(len(df)-len(HighVA)))
            df['POC']  = pd.Series(POC + [POC[len(POC)-1]]*(len(df)-len(POC)))
            df['POC2']  = pd.Series(pocc + [pocc[len(pocc)-1]]*(len(df)-len(pocc)))
            #df['POCDistance'] = (abs(df['1ema'] - df['POC']) / ((df['1ema']+ df['POC']) / 2)) * 100
            df['POCDistance'] = ((df['1ema'] - df['POC2']) / ((df['1ema'] + df['POC2']) / 2)) * 100
            
        
            
            df['cross_above'] = (df['1ema'] >= df['POC2']) & (df['derivative'] >= 0)# &  (df['MACD'] > df['Signal'])#(df['1ema'].shift(1) < df['POC2'].shift(1)) & 

            # Identify where cross below occurs (previous 3ema is above POC, current 3ema is below)
            df['cross_below'] =  (df['1ema'] <= df['POC2']) & (df['derivative'] <= 0)# & (df['Signal']  > df['MACD']) #(df['1ema'].shift(1) > df['POC2'].shift(1)) &
            
            # Get the indices where cross_above or cross_below happens
            #cross_above_indices = df[df['cross_above']].index
            #cross_below_indices = df[df['cross_below']].index
            
        '''
        blob = Blob('POCData'+str(symbolNum), bucket) 
        POCData = blob.download_as_text()
        csv_reader  = csv.reader(io.StringIO(POCData))   


        csv_rows = []
        for row in csv_reader:
            csv_rows.append(row) 
            
        for i in csv_rows:
            i[3] = int(float(i[3]))
        dfVA = pd.DataFrame(csv_rows, columns=['LowVA', 'HighVA', 'POC', 'Timestamp', 'Time'])

        # Step 3: Convert the 'Timestamp' to datetime format (nanoseconds format to datetime)
        dfVA['Timestamp'] = dfVA['Timestamp'] // 1_000_000_000
        dfVA['Timestamp'] = pd.to_datetime(dfVA['Timestamp'], unit='s')

        # Step 4: Set 'Timestamp' as the DataFrame index
        dfVA.set_index('Timestamp', inplace=True)

        # Step 5: Convert numerical columns to float
        dfVA['LowVA'] = dfVA['LowVA'].astype(float)
        dfVA['HighVA'] = dfVA['HighVA'].astype(float)
        dfVA['POC'] = dfVA['POC'].astype(float)

        # Step 6: Resample to 3-minute intervals and aggregate values
        resampled_dfVA = dfVA.resample(interv+'min').agg({
            'LowVA': 'last',  # Open price should be the first in the resampled period
            'HighVA': 'last',    # High price should be the maximum in the resampled period
            'POC': 'last',     # Low price should be the minimum in the resampled period
            'Time': 'first'   # Keep the first time for the resampled period
        })

        df['LowVA'] = pd.Series([i for i in resampled_dfVA['LowVA']])
        df['HighVA'] = pd.Series([i for i in resampled_dfVA['HighVA']])
        df['POC']  = pd.Series([i for i in resampled_dfVA['POC']])
        '''
    except(NotFound):
        pass
        
     
    try:
        mboString = str(round((abs(df['HighVA'][len(df)-1] - df['LowVA'][len(df)-1]) / ((df['HighVA'][len(df)-1] + df['LowVA'][len(df)-1]) / 2)) * 100,3))
    except(KeyError):
        mboString = ''

    #calculate_ttm_squeeze(df)
    
    
        
    if interval_time == initial_inter:
        interval_time = subsequent_inter
    
    if sname != previous_stkName or interv != previous_interv or tpoNum != previous_tpoNum:
        interval_time = initial_inter
    
    fg = plotChart(df, [hs[1],newwT[:int(tpoNum)]], va[0], va[1], x_fake, df_dx, mboString=mboString,  stockName=symbolNameList[symbolNumList.index(symbolNum)], previousDay=previousDay, pea=False,  OptionTimeFrame = stored_data['timeFrame'], clusterNum=int(clustNum), troInterval=stored_data['tro']) #trends=FindTrends(df,n=10)

    return stored_data, fg, previous_stkName, previous_interv, previous_tpoNum, interval_time

#[(i[2]-i[3],i[0]) for i in timeFrame ]
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8080)
    #app.run_server(debug=False, use_reloader=False)

'''
import time  
start_time = time.time()


end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
'''