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
#import yfinance as yf
#import dateutil.parser

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


def plotChart(df, lst2, num1, num2, x_fake, df_dx,  stockName='',   trends:list=[], pea:bool=False,  previousDay:list=[], OptionTimeFrame:list=[], clusterNum:int=5):
  
    
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
    

    buys = [abs(i[2]-i[3]) for i in OptionTimeFrame if i[2]-i[3] > 0 ]
    sells = [abs(i[2]-i[3]) for i in OptionTimeFrame if i[2]-i[3] < 0 ]

    
    
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
        

    fig = make_subplots(rows=3, cols=2, shared_xaxes=True, shared_yaxes=True,
                        specs=[[{}, {},],
                               [{"colspan": 1},{},],
                               [{"colspan": 1},{},]], #[{"colspan": 1},{},][{}, {}, ]'+ '<br>' +' ( Put:'+str(putDecHalf)+'('+str(NumPutHalf)+') | '+'Call:'+str(CallDecHalf)+'('+str(NumCallHalf)+') '
                        horizontal_spacing=0.02, vertical_spacing=0.03, subplot_titles=(stockName +' (Sell:'+str(putDec)+' ('+str(round(NumPut,2))+') | '+'Buy:'+str(CallDec)+' ('+str(round(NumCall,2))+') \n '+' (Sell:'+str(thputDec)+' ('+str(round(thNumPut,2))+') | '+'Buy:'+str(thCallDec)+' ('+str(round(thNumCall,2))+') \n '+strTrend + '('+str(average)+') '+ str(now)+ '  (Sell:'+str(sum(sells))+') (Buy:'+str(sum(buys))+') ', 'Volume Profile ' + str(datetime.now().time()) ), #,str(Ask)+'(Sell:'+str(dAsk)+') | '+str(Bid)+ '(Buy'+str(dBid)+') '
                         column_widths=[0.85,0.15], row_width=[0.15, 0.15, 0.70,] ) #,row_width=[0.30, 0.70,]

    
            
    
       
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
         row=3, col=1
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
        row=3, col=1
    )

    
    bms = pd.Series([i[2] for i in OptionTimeFrame]).rolling(9).mean()
    sms = pd.Series([i[3] for i in OptionTimeFrame]).rolling(9).mean()
    #xms = pd.Series([i[3]+i[2] for i in OptionTimeFrame]).rolling(4).mean()
    fig.add_trace(go.Scatter(x=pd.Series([i[0] for i in OptionTimeFrame]), y=bms, line=dict(color='teal'), mode='lines', name='Buy VMA'), row=3, col=1)
    fig.add_trace(go.Scatter(x=pd.Series([i[0] for i in OptionTimeFrame]), y=sms, line=dict(color='crimson'), mode='lines', name='Sell VMA'), row=3, col=1)

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
        colo.append([str(round(fk[0],7))+'A',fk[7],fk[8], fk[7]/(fk[7]+fk[8]+1)])
        colo.append([str(round(fk[0],7))+'B',fk[8],fk[7], fk[8]/(fk[7]+fk[8]+1)])
    fig.add_trace(
        go.Bar(
            x=pd.Series([i[1] for i in colo]),
            y=pd.Series([float(i[0][:len(i[0])-1]) for i in colo]),
            text=np.around(pd.Series([float(i[0][:len(i[0])-1]) for i in colo]), 6),
            textposition='auto',
            orientation='h',
            #width=0.2,
            marker_color=[     'teal' if 'B' in i[0] and i[3] < 0.65
                        else '#00FFFF' if 'B' in i[0] and i[3] >= 0.65
                        else 'crimson' if 'A' in i[0] and i[3] < 0.65
                        else 'pink' if 'A' in i[0] and i[3] >= 0.65
                        else i for i in colo],
            hovertext=pd.Series([i[0][:len(i[0])-1] + ' '+ str(round(i[1] / (i[1]+i[2]+1),2)) + ' '+ str(round(i[2]/ (i[1]+i[2]+1),2)) for i in colo])#pd.Series([str(round(i[7],3)) + ' ' + str(round(i[8],3))  + ' ' + str(round(i[9],3)) +' ' + str(round([i[7], i[8], i[9]][[i[7], i[8], i[9]].index(max([i[7], i[8], i[9]]))]/sum([i[7], i[8], i[9]]),2)) if sum([i[7], i[8], i[9]]) > 0 else '' for i in sortadlist2]),
        ),
        row=1, col=2
    )



    fig.add_trace(go.Scatter(x=[sortadlist2[0][1], sortadlist2[0][1]], y=[
                  num1, num2],  opacity=0.5), row=1, col=2)
    
    
    #fig.add_trace(go.Scatter(x=x_fake, y=df_dx, mode='lines',name='Derivative'), row=2, col=2)
    
    fig.add_trace(go.Scatter(x=df['time'], y=df['vwap'], mode='lines', name='VWAP'))
    

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
    

    fig.add_hline(y=df['close'][len(df)-1], row=1, col=2)
    
    
    #fig.add_hline(y=0, row=1, col=4)
    
 
    trcount = 0
    
    for trd in sortadlist:
        trd.append(df['timestamp'].searchsorted(trd[2])-1)
        


    putCand = [i for i in OptionTimeFrame if int(i[2]) > int(i[3]) if int(i[4]) < len(df)] # if int(i[4]) < len(df)
    callCand = [i for i in OptionTimeFrame if int(i[3]) > int(i[2]) if int(i[4]) < len(df)] # if int(i[4]) < len(df) +i[3]+i[5] +i[2]+i[5]
    MidCand = [i for i in OptionTimeFrame if int(i[3]) == int(i[2]) if int(i[4]) < len(df)]
    
    est_now = datetime.utcnow() + timedelta(hours=-4)
    start_time = est_now.replace(hour=8, minute=00, second=0, microsecond=0)
    end_time = est_now.replace(hour=17, minute=30, second=0, microsecond=0)
    
    # Check if the current time is between start and end times
    if start_time <= est_now <= end_time:
        ccheck = 0.58
    else:
       ccheck = 0.60
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
    
    if len(MidCand) > 0:
       fig.add_trace(go.Candlestick(
           x=[df['time'][i[4]] for i in MidCand],
           open=[df['open'][i[4]] for i in MidCand],
           high=[df['high'][i[4]] for i in MidCand],
           low=[df['low'][i[4]] for i in MidCand],
           close=[df['close'][i[4]] for i in MidCand],
           increasing={'line': {'color': 'gray'}},
           decreasing={'line': {'color': 'gray'}},
           hovertext=['('+str(i[2])+')'+str(round(i[6],2))+' '+str('Bid')+' '+'('+str(i[3])+')'+str(round(i[7],2))+' Ask' + '<br>' + str(i[2]-i[3])   for i in MidCand], #+ i[11] + str(sum([i[10][x][2] for x in i[10]]))
           hoverlabel=dict(
                bgcolor="gray",
                font=dict(color="black", size=8),
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
            hovertext=['('+str(i[2])+')'+str(round(i[6],2))+' '+str('Bid')+' '+'('+str(i[3])+')'+str(round(i[7],2))+' Ask' + '<br>' + str(i[2]-i[3]) for i in putCand], #i[11] + str(sum([i[10][x][2] for x in i[10]]))
            hoverlabel=dict(
                 bgcolor="teal",
                 font=dict(color="white", size=8),
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
            hovertext=['('+str(i[2])+')'+str(round(i[6],2))+' '+str('Bid')+' '+'('+str(i[3])+')'+str(round(i[7],2))+' Ask' + '<br>' +  str(i[2]-i[3]) for i in callCand], #i[11] + str(sum([i[10][x][2] for x in i[10]]))
            hoverlabel=dict(
                 bgcolor="pink",
                 font=dict(color="black", size=8),
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
            hovertext=['('+str(i[2])+')'+str(round(i[6],2))+' '+str('Bid')+' '+'('+str(i[3])+')'+str(round(i[7],2))+' Ask' + '<br>' + str(i[2]-i[3]) for i in indsAbove], #i[11] + str(sum([i[10][x][2] for x in i[10]]))
            hoverlabel=dict(
                 bgcolor="#00FFFF",
                 font=dict(color="black", size=8),
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
            hovertext=['('+str(i[2])+')'+str(round(i[6],2))+' '+str('Bid')+' '+'('+str(i[3])+')'+str(round(i[7],2))+' Ask' + '<br>' + str(i[2]-i[3]) for i in indsBelow], #i[11] + str(sum([i[10][x][2] for x in i[10]]))
            hoverlabel=dict(
                 bgcolor="#FF1493",
                 font=dict(color="white", size=8),
                 ),
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
        
    fig.add_trace(go.Scatter(x=df['time'],
                             y= [df['vwap'].mean()]*len(df['time']) ,
                             line_color='#FF1493',
                             text = str(df['vwap'].mean()),
                             textposition="bottom left",
                             name='Avg VWAP '+ str(df['vwap'].mean()),
                             showlegend=False,
                             visible=False,
                             mode= 'lines',
                            ),
                 )    
        
    fig.add_trace(go.Scatter(x=df['time'],
                             y= [df['STDEV_25'].mean()]*len(df['time']) ,
                             line_color='chartreuse',
                             text = str(df['STDEV_25'].mean()),
                             textposition="bottom left",
                             name='Avg UVWAP '+ str(df['STDEV_25'].mean()),
                             showlegend=False,
                             visible=False,
                             mode= 'lines',
                            ),
                 )
    
    fig.add_trace(go.Scatter(x=df['time'],
                             y= [df['STDEV_N25'].mean()]*len(df['time']) ,
                             line_color='chartreuse',
                             text = str(df['STDEV_N25'].mean()),
                             textposition="bottom left",
                             name='Avg LVWAP '+ str(df['STDEV_N25'].mean()),
                             showlegend=False,
                             visible=False,
                             mode= 'lines',
                            ),
                 )

    data = [i[0] for i in sortadlist]
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
            
            
            
            opac = round((len(i)/mazz)/1.3,2)
            fig.add_shape(type="rect",
                      y0=i[0], y1=i[len(i)-1], x0=-1, x1=len(df),
                      fillcolor="crimson" if askCount > bidCount else 'teal' if askCount < bidCount else 'gray',
                      opacity=opac)


            
            fig.add_trace(go.Scatter(x=df['time'],
                                 y= [i[0]]*len(df['time']) ,
                                 line_color='rgba(220,20,60,'+str(opac)+')' if askCount > bidCount else 'rgba(0,139,139,'+str(opac)+')' if askCount < bidCount else 'gray',
                                 text =str(i[0])+ ' (' + str(len(i))+ ') Ask:('+ str(askDec) + ') '+str(askCount)+' | Bid: ('+ str(bidDec) +') '+str(bidCount),
                                 textposition="bottom left",
                                 name=str(i[0])+ ' (' + str(len(i))+ ') Ask:('+ str(askDec) + ') '+str(askCount)+' | Bid: ('+ str(bidDec) +') '+str(bidCount),
                                 showlegend=False,
                                 mode= 'lines',
                                
                                ),
                      row=1, col=1)
            trcount+=1

            fig.add_trace(go.Scatter(x=df['time'],
                                 y= [i[len(i)-1]]*len(df['time']) ,
                                 line_color='rgba(220,20,60,'+str(opac)+')' if askCount > bidCount else 'rgba(0,139,139,'+str(opac)+')' if askCount < bidCount else 'gray',
                                 text = str(i[len(i)-1])+ ' (' + str(len(i))+ ') Ask:('+ str(askDec) + ') '+str(askCount)+' | Bid: ('+ str(bidDec) +') '+str(bidCount),
                                 textposition="bottom left",
                                 name= str(i[len(i)-1])+ ' (' + str(len(i))+ ') Ask:('+ str(askDec) + ') '+str(askCount)+' | Bid: ('+ str(bidDec) +') '+str(bidCount),
                                 showlegend=False,
                                 mode= 'lines',
                                
                                ),
                      row=1, col=1)
            trcount+=1
    
    #df_dx = np.append(df_dx, df_dx[len(df_dx)-1])
    difList = [(i[2]-i[3],i[0]) for i in OptionTimeFrame]
    coll = [     'teal' if i[0] > 0
                else 'crimson' if i[0] < 0
                else 'gray' for i in difList]
    fig.add_trace(go.Bar(x=pd.Series([i[1] for i in difList]), y=pd.Series([i[0] for i in difList]), marker_color=coll), row=2, col=1)
    '''
    
    posti = sum([i[0] for i in difList if i[0] > 0])/len([i[0] for i in difList if i[0] > 0])
    negati = sum([i[0] for i in difList if i[0] < 0])/len([i[0] for i in difList if i[0] < 0])

    fig.add_trace(go.Scatter(x=df['time'],
                             y= [posti]*len(df['time']) ,
                             line_color='teal',
                             text = str(posti),
                             textposition="bottom left",
                             name=str(posti),
                             showlegend=False,
                             mode= 'lines',
                            ),
                    row=2, col=1
                 )

    fig.add_trace(go.Scatter(x=df['time'],
                             y= [negati]*len(df['time']) ,
                             line_color='crimson',
                             text = str(negati),
                             textposition="bottom left",
                             name=str(negati),
                             showlegend=False,
                             mode= 'lines',
                            ),
                    row=2, col=1
                 )
   
    '''
    posti = pd.Series([i[0] if i[0] > 0 else 0  for i in difList]).rolling(9).mean()#sum([i[0] for i in difList if i[0] > 0])/len([i[0] for i in difList if i[0] > 0])
    negati = pd.Series([i[0] if i[0] < 0 else 0 for i in difList]).rolling(9).mean()#sum([i[0] for i in difList if i[0] < 0])/len([i[0] for i in difList if i[0] < 0])
    
    fig.add_trace(go.Scatter(x=pd.Series([i[0] for i in OptionTimeFrame]), y=posti, line=dict(color='teal'), mode='lines', name='Buy VMA'), row=2, col=1)
    fig.add_trace(go.Scatter(x=pd.Series([i[0] for i in OptionTimeFrame]), y=negati, line=dict(color='crimson'), mode='lines', name='Sell VMA'), row=2, col=1)



    for trds in sortadlist[:10]:
        try:
            if str(trds[3]) == 'A':
                vallue = 'Sell'
                sidev = trds[0]
            elif str(trds[3]) == 'B':
                vallue = 'BUY'
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
    
    
    fig.update_layout(height=880, xaxis_rangeslider_visible=False, showlegend=False)
    fig.update_xaxes(autorange="reversed", row=1, col=2)
    #fig.update_xaxes(autorange="reversed", row=1, col=3)
    #fig.update_layout(plot_bgcolor='gray')
    #fig.update_layout(paper_bgcolor='rgba(96, 95, 93, 0.5)')
    #"paper_bgcolor": "rgba(0, 0, 0, 0)",

    
    
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=2)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    #fig.update_xaxes(showticklabels=False, row=3, col=1)
    #fig.show(config={'modeBarButtonsToAdd': ['drawline']})
    return fig


symbolNumList = ['5602', '13743', '80420', '42009544', '200430', '669']
symbolNameList = ['ES', 'NQ',  'YM',  'BTC', 'CL', 'GC']

intList = ['1','2','3','4','5','6','10','15']

gclient = storage.Client(project="stockapp-401615")
bucket = gclient.get_bucket("stockapp-storage")

#import pandas_ta as ta
from collections import Counter
from dash import Dash, dcc, html, Input, Output, callback, State
inter = 40000#210000#250000#80001
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
    html.Div(id='container-button-basic',children="Enter a symbol from |'ES' 'NQ' 'GC' 'HG' 'YM' 'RTY' 'SI' 'CL' 'NG' | and submit"),
    dcc.Store(id='stkName-value'),
    
    html.Div(dcc.Input(id='input-on-interv', type='text')),
    html.Button('Submit', id='submit-interv', n_clicks=0),
    html.Div(id='interv-button-basic',children="Enter a symbol from |5 10 15 30 | and submit"),
    dcc.Store(id='interv-value'),
    
    dcc.Store(id='data-store'),
    dcc.Store(id='previous-interv'),
    dcc.Store(id='previous-stkName'),
    
    
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
        return 'The input symbol '+str(value)+" is not accepted please try different symbol from  |'ES' 'NQ' 'GC' 'HG' 'YM' 'RTY' 'SI' 'CL' 'NG'|", 'The input symbol was '+str(value)+" is not accepted please try different symbol  |'ESH4' 'NQH4' 'CLG4' 'GCG4' 'NGG4' 'HGH4' 'YMH4' 'BTCZ3' 'RTYH4'|  "

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
    [Output('data-store', 'data'),
        Output('graph', 'figure'),
        Output('previous-stkName', 'data'),
        Output('previous-interv', 'data')],
    [Input('interval', 'n_intervals')],
    [State('stkName-value', 'data'),
        State('interv-value', 'data'),
        State('data-store', 'data'),
        State('previous-stkName', 'data'),
        State('previous-interv', 'data')
    ],
)
    
def update_graph_live(n_intervals, sname, interv, stored_data, previous_stkName, previous_interv): #interv
    print('inFunction')	
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
        interv = '5'
        
    if stkName != previous_stkName:
        stored_data = None

    if interv != previous_interv:
        stored_data = None
    
    
        
    
    
    
    blob = Blob('FuturesOHLC'+str(symbolNum), bucket) 
    FuturesOHLC = blob.download_as_text()
        

    csv_reader  = csv.reader(io.StringIO(FuturesOHLC))
    
    csv_rows = []
    for row in csv_reader:
        csv_rows.append(row)
        
    
    
    aggs = [ ]  
    newOHLC = [i for i in csv_rows]

    for i in newOHLC:
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
    
    df['strTime'] = df['timestamp'].apply(lambda x: pd.Timestamp(int(x) // 10**9, unit='s', tz='EST') )
    
    df.set_index('strTime', inplace=True)
    df['volume'] = pd.to_numeric(df['volume'], downcast='integer')
    df_resampled = df.resample(interv+'T').agg({
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


    
    blob = Blob('FuturesTrades'+str(symbolNum), bucket) 
    FuturesTrades = blob.download_as_text()
    
    
    csv_reader  = csv.reader(io.StringIO(FuturesTrades))
    
    csv_rows = []
    for row in csv_reader:
        csv_rows.append(row)
       

    #STrades = [i for i in csv_rows]
    AllTrades = []
    for i in csv_rows:
        hourss = datetime.fromtimestamp(int(int(i[0])// 1000000000)).hour
        if hourss < 10:
            hourss = '0'+str(hourss)
        minss = datetime.fromtimestamp(int(int(i[0])// 1000000000)).minute
        if minss < 10:
            minss = '0'+str(minss)
        opttimeStamp = str(hourss) + ':' + str(minss) + ':00'
        AllTrades.append([int(i[1])/1e9, int(i[2]), int(i[0]), 0, i[3], opttimeStamp])
       
    #AllTrades = [i for i in AllTrades if i[1] > 1]
        
    hs = historV1(df,50,{},AllTrades,[])
    
    va = valueAreaV1(hs[0])
    
    

    x = np.array([i for i in range(len(df))])
    y = np.array([i for i in df['40ema']])
    
    

    # Simple interpolation of x and y
    f = interp1d(x, y)
    x_fake = np.arange(0.1, len(df)-1, 1)

    # derivative of y with respect to x
    df_dx = derivative(f, x_fake, dx=1e-6)



    mTrade = [i for i in AllTrades ]
    
     
    mTrade = sorted(mTrade, key=lambda d: d[1], reverse=True)
    
    [mTrade[i].insert(4,i) for i in range(len(mTrade))] 
    
    newwT = []
    for i in mTrade:
        newwT.append([i[0],i[1],i[2],i[5], i[4],i[3],i[6]])
    
    
    dtime = df['time'].dropna().values.tolist()
    dtimeEpoch = df['timestamp'].dropna().values.tolist()
    
    
    tempTrades = [i for i in AllTrades]
    tempTrades = sorted(tempTrades, key=lambda d: d[6], reverse=False) 
    tradeTimes = [i[6] for i in tempTrades]
    
    if stored_data is not None:
        print('here')
        timeDict = {}
        lastTime = stored_data['timeFrame'][len(stored_data['timeFrame'])-1][0]
        for ttm in dtime[dtime.index(lastTime):]:
            for tradMade in tempTrades[bisect.bisect_left(tradeTimes, ttm):]:
                if datetime.strptime(tradMade[6], "%H:%M:%S") > datetime.strptime(ttm, "%H:%M:%S") + timedelta(minutes=int(interv)):
                    try:
                        timeDict[ttm] += [timeDict[ttm][0]/sum(timeDict[ttm]), timeDict[ttm][1]/sum(timeDict[ttm]), timeDict[ttm][2]/sum(timeDict[ttm])]
                    except(KeyError,ZeroDivisionError):
                        timeDict[ttm] = [0,0,0]
                    break
                
                if ttm not in timeDict:
                    timeDict[ttm] = [0,0,0]
                if ttm in timeDict:
                    if tradMade[5] == 'B':
                        timeDict[ttm][0] += tradMade[1]#tradMade[0] * tradMade[1]
                    elif tradMade[5] == 'A':
                        timeDict[ttm][1] += tradMade[1]#tradMade[0] * tradMade[1] 
                    elif tradMade[5] == 'N':
                        timeDict[ttm][2] += tradMade[1]#tradMade[0] * tradMade[1] 
                        
    
        for i in timeDict:
            if len(timeDict[i]) == 3:
                try:
                    timeDict[i] += [timeDict[i][0]/sum(timeDict[i]), timeDict[i][1]/sum(timeDict[i]), timeDict[i][2]/sum(timeDict[i])]#
                except(ZeroDivisionError,KeyError):
                    timeDict[i] += [0, 0,0]
                    
        
                                    
        timeFrame = [[i,'']+timeDict[i] for i in timeDict]
    
        for i in range(len(timeFrame)):
            timeFrame[i].append(dtimeEpoch[dtime.index(timeFrame[i][0])])
            

        for pott in timeFrame:
            pott.insert(4,df['timestamp'].searchsorted(pott[8]))
            
        stored_data['timeFrame'] = stored_data['timeFrame'][:len(stored_data['timeFrame'])-1] + timeFrame
        #timeFrame = stored_data['timeFrame']
    
    if stored_data is None:
        print('Newstored')
        timeDict = {}
        for ttm in dtime:
            for tradMade in tempTrades[bisect.bisect_left(tradeTimes, ttm):]:
                if datetime.strptime(tradMade[6], "%H:%M:%S") > datetime.strptime(ttm, "%H:%M:%S") + timedelta(minutes=int(interv)):
                    try:
                        timeDict[ttm] += [timeDict[ttm][0]/sum(timeDict[ttm]), timeDict[ttm][1]/sum(timeDict[ttm]), timeDict[ttm][2]/sum(timeDict[ttm])]
                    except(KeyError,ZeroDivisionError):
                        timeDict[ttm] = [0,0,0]
                    break
                
                if ttm not in timeDict:
                    timeDict[ttm] = [0,0,0]
                if ttm in timeDict:
                    if tradMade[5] == 'B':
                        timeDict[ttm][0] += tradMade[1]#tradMade[0] * tradMade[1]
                    elif tradMade[5] == 'A':
                        timeDict[ttm][1] += tradMade[1]#tradMade[0] * tradMade[1] 
                    elif tradMade[5] == 'N':
                        timeDict[ttm][2] += tradMade[1]#tradMade[0] * tradMade[1] 
                    
    
        for i in timeDict:
            if len(timeDict[i]) == 3:
                try:
                    timeDict[i] += [timeDict[i][0]/sum(timeDict[i]), timeDict[i][1]/sum(timeDict[i]), timeDict[i][2]/sum(timeDict[i])]#
                except(ZeroDivisionError,KeyError):
                    timeDict[i] += [0, 0,0]
                    
        
                                    
        timeFrame = [[i,'']+timeDict[i] for i in timeDict]
    
        for i in range(len(timeFrame)):
            timeFrame[i].append(dtimeEpoch[i])
            
        for pott in timeFrame:
            pott.insert(4,df['timestamp'].searchsorted(pott[8]))
            
        stored_data = {'timeFrame': timeFrame} 
        
    #OptionTimeFrame = stored_data['timeFrame']   
    previous_stkName = sname
    previous_interv = interv

         
    
    #df['superTrend'] = ta.supertrend(df['high'], df['low'], df['close'], length=2, multiplier=1.8)['SUPERTd_2_1.8']
    #df['superTrend'][df['superTrend'] < 0] = 0
 
    blob = Blob('PrevDay', bucket) 
    PrevDay = blob.download_as_text()
        

    csv_reader  = csv.reader(io.StringIO(PrevDay))

    csv_rows = []
    for row in csv_reader:
        csv_rows.append(row)
        
    try:   
        previousDay = [csv_rows[[i[4] for i in csv_rows].index(symbolNum)][0] ,csv_rows[[i[4] for i in csv_rows].index(symbolNum)][1] ,csv_rows[[i[4] for i in csv_rows].index(symbolNum)][2]]
    except(ValueError):
        previousDay = []
    
    fg = plotChart(df, [hs[1],newwT[:100]], va[0], va[1], x_fake, df_dx,  stockName=symbolNameList[symbolNumList.index(symbolNum)], previousDay=previousDay, pea=False,  OptionTimeFrame = stored_data['timeFrame'], clusterNum=5) #trends=FindTrends(df,n=10)

    return stored_data, fg, previous_stkName, previous_interv

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