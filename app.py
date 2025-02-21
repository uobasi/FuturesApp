# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 03:50:19 2025

@author: uobas
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 09:35:54 2025

@author: UOBASUB
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:42:48 2025

@author: UOBASUB
"""

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
warnings.filterwarnings("ignore", category=RuntimeWarning)
import plotly.io as pio
pio.renderers.default='browser'
import bisect
from collections import defaultdict
from scipy.signal import savgol_filter
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter1d
from numpy.polynomial import Polynomial
from scipy.stats import percentileofscore
#import yfinance as yf
#import dateutil.parser


def calculate_tema(df, span):
    ema1 = df['close'].ewm(span=span, adjust=False).mean()
    ema2 = ema1.ewm(span=span, adjust=False).mean()
    ema3 = ema2.ewm(span=span, adjust=False).mean()
    
    tema = 3 * ema1 - 3 * ema2 + ema3
    return tema

def calculate_custom_ema(df, span):
    ema1 = df['close'].ewm(span=span, adjust=False).mean()
    ema2 = ema1.ewm(span=span, adjust=False).mean()
    ema3 = ema2.ewm(span=span, adjust=False).mean()
    ema4 = ema3.ewm(span=span, adjust=False).mean()
    ema5 = ema4.ewm(span=span, adjust=False).mean()
    ema6 = ema5.ewm(span=span, adjust=False).mean()
    ema7 = ema6.ewm(span=span, adjust=False).mean()
    
    # Custom formula for a higher-order EMA (you can customize this)
    custom_ema = 7 * ema1 - 21 * ema2 + 35 * ema3 - 35 * ema4 + 21 * ema5 - 7 * ema6 + ema7
    return custom_ema

def ema(df):
    df['30ema'] = df['close'].ewm(span=30, adjust=False).mean()
    df['9ema'] = df['close'].ewm(span=9, adjust=False).mean()
    df['21ema'] = df['close'].ewm(span=21, adjust=False).mean()
    df['40ema'] = df['close'].ewm(span=40, adjust=False).mean()
    #df['28ema'] = df['close'].ewm(span=28, adjust=False).mean()
    df['3ema'] = df['close'].ewm(span=3, adjust=False).mean()
    df['5ema'] = df['close'].ewm(span=5, adjust=False).mean()
    #df['50ema'] = df['close'].ewm(span=50, adjust=False).mean()
    #df['15ema'] = df['close'].ewm(span=15, adjust=False).mean()
    df['20ema'] = df['close'].ewm(span=20, adjust=False).mean()
    #df['10ema'] = df['close'].ewm(span=10, adjust=False).mean()
    #df['100ema'] = df['close'].ewm(span=100, adjust=False).mean()
    #df['150ema'] = df['close'].ewm(span=150, adjust=False).mean()
    #df['200ema'] = df['close'].ewm(span=200, adjust=False).mean()
    df['2ema'] = df['close'].ewm(span=2, adjust=False).mean()
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
    

def vwapCum(df):
    v = df['volume'].values
    h = df['high'].values
    l = df['low'].values
    # print(v)
    df['vwapCum'] = np.cumsum(v*(h+l)/2) / np.cumsum(v)
    df['volumeSumCum'] = df['volume'].cumsum()
    df['volume2SumCum'] = (v*((h+l)/2)*((h+l)/2)).cumsum()
    #df['disVWAP'] = (abs(df['close'] - df['vwap']) / ((df['close'] + df['vwap']) / 2)) * 100
    #df['disVWAPOpen'] = (abs(df['open'] - df['vwap']) / ((df['open'] + df['vwap']) / 2)) * 100
    #df['disEMAtoVWAP'] = ((df['close'].ewm(span=12, adjust=False).mean() - df['vwap'])/df['vwap']) * 100



def sigma(df):
    try:
        val = df.volume2Sum / df.volumeSum - df.vwap * df.vwap
    except(ZeroDivisionError):
        val = df.volume2Sum / (df.volumeSum+0.000000000001) - df.vwap * df.vwap
    return math.sqrt(val) if val >= 0 else val


def sigmaCum(df):
    try:
        val = df.volume2SumCum / df.volumeSumCum - df.vwapCum * df.vwapCum
    except(ZeroDivisionError):
        val = df.volume2SumCum / (df.volumeSumCum+0.000000000001) - df.vwapCum * df.vwapCum
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



def PPPCum(df):

    df['STDEV_TVCum'] = df.apply(sigmaCum, axis=1)
    stdev_multiple_0_25 = 0.25
    stdev_multiple_0 = 0.50
    stdev_multiple_0_75 = 0.75
    stdev_multiple_1 = 1
    stdev_multiple_1_25 = 1.25
    stdev_multiple_1_5 = 1.5
    stdev_multiple_1_75 = 1.75
    stdev_multiple_2 = 2.00
    stdev_multiple_2_25 = 2.25
    stdev_multiple_25 = 2.50
    
    df['STDEV_025Cum'] = df.vwapCum + stdev_multiple_0_25 * df['STDEV_TVCum']
    df['STDEV_N025Cum'] = df.vwapCum - stdev_multiple_0_25 * df['STDEV_TVCum']

    df['STDEV_0Cum'] = df.vwapCum + stdev_multiple_0 * df['STDEV_TVCum']
    df['STDEV_N0Cum'] = df.vwapCum - stdev_multiple_0 * df['STDEV_TVCum']
    
    df['STDEV_075Cum'] = df.vwapCum + stdev_multiple_0_75 * df['STDEV_TVCum']
    df['STDEV_N075Cum'] = df.vwapCum - stdev_multiple_0_75 * df['STDEV_TVCum']

    df['STDEV_1Cum'] = df.vwapCum + stdev_multiple_1 * df['STDEV_TVCum']
    df['STDEV_N1Cum'] = df.vwapCum - stdev_multiple_1 * df['STDEV_TVCum']
    
    df['STDEV_125Cum'] = df.vwapCum + stdev_multiple_1_25 * df['STDEV_TVCum']
    df['STDEV_N125Cum'] = df.vwapCum - stdev_multiple_1_25 * df['STDEV_TVCum']
    
    df['STDEV_15Cum'] = df.vwapCum + stdev_multiple_1_5 * df['STDEV_TVCum']
    df['STDEV_N15Cum'] = df.vwapCum - stdev_multiple_1_5 * df['STDEV_TVCum']
    
    df['STDEV_175Cum'] = df.vwapCum + stdev_multiple_1_75 * df['STDEV_TVCum']
    df['STDEV_N175Cum'] = df.vwapCum - stdev_multiple_1_75 * df['STDEV_TVCum']

    df['STDEV_2Cum'] = df.vwapCum + stdev_multiple_2 * df['STDEV_TVCum']
    df['STDEV_N2Cum'] = df.vwapCum - stdev_multiple_2 * df['STDEV_TVCum']
    
    df['STDEV_225Cum'] = df.vwapCum + stdev_multiple_2_25 * df['STDEV_TVCum']
    df['STDEV_N225Cum'] = df.vwapCum - stdev_multiple_2_25 * df['STDEV_TVCum']
    
    df['STDEV_25Cum'] = df.vwapCum + stdev_multiple_25 * df['STDEV_TVCum']
    df['STDEV_N25Cum'] = df.vwapCum - stdev_multiple_25 * df['STDEV_TVCum']



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
    pocT = max(dct, key=dct.get)
    
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
    
    return [cptemp,sortadlist,pocT]  


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
    print(lst)
    mkk = [i for i in lst if i[1] > 0]
    if len(mkk) == 0:
        mkk = lst
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

def valueAreaV3(lst):
    # Ensure list is not empty
    if not lst:
        return [None, None, None]

    # Filter out entries with zero volume
    mkk = [i for i in lst if i[1] > 0]
    if not mkk:
        mkk = lst

    # Assign indices for tracking
    for idx, item in enumerate(mkk):
        item[2] = idx

    # Total volume in mkk
    total_volume = sum([i[1] for i in mkk])
    if total_volume == 0:
        return [None, None, None]

    # Identify POC (Point of Control) by maximum volume
    poc_item = max(mkk, key=lambda x: x[1])
    pocIndex = poc_item[2]
    sPercent = total_volume * 0.70  # 70% of total volume
    accumulated_volume = poc_item[1]  # Start with POC volume

    # Initialize Value Area boundaries
    topIndex, dwnIndex = pocIndex, pocIndex

    # Expand the value area until 70% of volume is captured
    while accumulated_volume < sPercent:
        topVol = mkk[topIndex - 1][1] if topIndex > 0 else 0
        dwnVol = mkk[dwnIndex + 1][1] if dwnIndex < len(mkk) - 1 else 0

        # Add the larger volume to the total and adjust indices
        if topVol >= dwnVol:
            if topIndex > 0:
                topIndex -= 1
                accumulated_volume += topVol
        else:
            if dwnIndex < len(mkk) - 1:
                dwnIndex += 1
                accumulated_volume += dwnVol

        # Break if boundaries are fully expanded
        if topIndex == 0 and dwnIndex == len(mkk) - 1:
            break

    # Return Value Area Low, Value Area High, and POC
    return [mkk[topIndex][0], mkk[dwnIndex][0], poc_item[0]]


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

def plotChart(df, lst2, num1, num2, x_fake, df_dx,  stockName='', troPerCandle:list=[],   trends:list=[], pea:bool=False,  previousDay:list=[], OptionTimeFrame:list=[], clusterNum:int=5, troInterval:list=[]):
  
    notround = np.average(df_dx)
    average = round(np.average(df_dx), 3)
    now = round(df_dx[len(df_dx)-1], 3)
    if notround > 0:
        strTrend = "Uptrend"
    elif notround < 0:
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
    # Ratio : '+str(ratio)+' ' 
    tpString = ' (Buy:' + str(tobuys) + '('+str(round(tobuys/(tobuys+tosells),2))+') | '+ '(Sell:' + str(tosells) + '('+str(round(tosells/(tobuys+tosells),2))+'))'
    
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
                         horizontal_spacing=0.01, vertical_spacing=0.00, subplot_titles=(stockName + ' '+ '('+str(average)+') '+ str(now)+ ' '+ tpString, 'VP ' + str(datetime.now().time()) ), #' (Sell:'+str(putDec)+' ('+str(round(NumPut,2))+') | '+'Buy:'+str(CallDec)+' ('+str(round(NumCall,2))+') \n '+' (Sell:'+str(thputDec)+' ('+str(round(thNumPut,2))+') | '+'Buy:'+str(thCallDec)+' ('+str(round(thNumCall,2))+') \n '
                         column_widths=[0.80,0.20], row_width=[0.12, 0.15, 0.73,] ) #,row_width=[0.30, 0.70,]

    
            
    
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
    
    
    #if 'POC' in df.columns:
        #fig.add_trace(go.Scatter(x=df['time'], y=df['derivative_2'], mode='lines',name='Derivative_2'), row=2, col=1)
    #fig.add_trace(go.Scatter(x=df['time'], y=df['smoothed_derivative'], mode='lines',name='smoothed_derivative'), row=3, col=1)
    #fig.add_trace(go.Scatter(x=df['time'], y=df['positive_threshold'], mode='lines',name='positive_threshold'))
    #fig.add_trace(go.Scatter(x=df['time'], y=df['negative_threshold'], mode='lines',name='negative_threshold'))
    #fig.add_hline(y=70, row=2, col=1)
    #fig.add_hline(y=30, row=2, col=1)
    
    #fig.add_trace(go.Scatter(x=df['time'], y=df['polyfit_slope'], mode='lines',name='polyfit_slope'), row=3, col=1) 
    #fig.add_trace(go.Scatter(x=df['time'], y=df['slope_degrees'], mode='lines',name='slope_degrees'), row=3, col=1)
    #fig.add_trace(go.Scatter(x=df['time'], y=df['smoothed_derivative'], mode='lines',name='smoothed_derivative'), row=3, col=1)
    fig.add_trace(go.Bar(x=df['time'], y=df['percentile_Posdiff'], marker_color='teal'), row=3, col=1)
    fig.add_trace(go.Bar(x=df['time'], y=df['percentile_Negdiff'], marker_color='crimson'), row=3, col=1)
    
        #fig.add_trace(go.Scatter(x=df['time'], y=df['smoothed_derivative'], mode='lines',name='smoothed_derivative'), row=2, col=1)
        #fig.add_trace(go.Scatter(x=df['time'], y=df['filtfilt'], mode='lines',name='filtfilt'), row=2, col=1) 
        #fig.add_trace(go.Scatter(x=df['time'], y=df['lfilter'], mode='lines',name='lfilter'), row=2, col=1)
        #fig.add_trace(go.Scatter(x=df['time'], y=df['holt_trend'], mode='lines',name='holt_trend'), row=2, col=1)
        
    #fig.add_trace(go.Scatter(x=df['time'], y=df['rolling_imbalance'], mode='lines',name='rolling_imbalance'), row=3, col=1)
        
    #fig.add_trace(go.Scatter(x=df['time'], y=df['smoothed_1ema'], mode='lines',name='smoothed_1ema',marker_color='rgba(0,0,0)'))


        #fig.add_trace(go.Scatter(x=df['time'], y=df['close'].rolling(window=clusterNum).mean(), mode='lines',name=str(clusterNum)+'ema'), row=2, col=1)
        #fig.add_trace(go.Scatter(x=df['time'], y=df['lsfreal_time'], mode='lines',name='lsfreal_time'), row=2, col=1)
        #fig.add_trace(go.Scatter(x=df['time'], y=df['HighVA'], mode='lines', opacity=0.30, name='HighVA',marker_color='rgba(0,0,0)'), row=2, col=1)
        #fig.add_trace(go.Scatter(x=df['time'], y=df['LowVA'], mode='lines', opacity=0.30,name='LowVA',marker_color='rgba(0,0,0)'), row=2, col=1)


    fig.add_hline(y=0, row=3, col=1)

    fig.add_trace(go.Scatter(x=df['time'], y=df['vwap'], mode='lines', name='VWAP', line=dict(color='crimson')))
    #fig.add_trace(go.Scatter(x=df['time'], y=df['9ema'], mode='lines',name='9ema'))
    #fig.add_trace(go.Scatter(x=df['time'], y=df['20ema'], mode='lines',name='20ema'))
    #fig.add_trace(go.Scatter(x=df['time'], y=df['POC2'], mode='lines',name='POC2'))
    
    
    #if 'POC' in df.columns:
    #fig.add_trace(go.Scatter(x=df['time'], y=df['POC'], mode='lines',name='POC',opacity=0.80,marker_color='#0000FF'))
        #fig.add_trace(go.Scatter(x=df['time'], y=df['POC2'], mode='lines',name='POC2',opacity=0.80,marker_color='black'))
        #fig.add_trace(go.Scatter(x=df['time'], y=df['POC'].cumsum() / (df.index + 1), mode='lines', opacity=0.50, name='CUMPOC',marker_color='#0000FF'))
    #fig.add_trace(go.Scatter(x=df['time'], y=df['POC'], mode='lines', opacity=0.80, name='POC',marker_color='#0000FF'))
        #fig.add_trace(go.Scatter(x=df['time'], y=df['LowVA'], mode='lines', opacity=0.30,name='LowVA',marker_color='rgba(0,0,0)'))
      
    #fig.add_trace(go.Scatter(x=df['time'], y=df['100ema'], mode='lines', opacity=0.3, name='100ema', line=dict(color='black')))
    #fig.add_trace(go.Scatter(x=df['time'], y=df['150ema'], mode='lines', opacity=0.3, name='150ema', line=dict(color='black')))
    #fig.add_trace(go.Scatter(x=df['time'], y=df['200ema'], mode='lines', opacity=0.3, name='200emaa', line=dict(color='black')))
    
    #fig.add_trace(go.Scatter(x=df['time'], y=df['uppervwapAvg'], mode='lines', opacity=0.30,name='uppervwapAvg', ))
    #fig.add_trace(go.Scatter(x=df['time'], y=df['lowervwapAvg'], mode='lines',opacity=0.30,name='lowervwapAvg', ))
    #fig.add_trace(go.Scatter(x=df['time'], y=df['vwapAvg'], mode='lines', opacity=0.30,name='vwapAvg', ))
    
    
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_2'], mode='lines', opacity=0.1, name='UPPERVWAP2', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N2'], mode='lines', opacity=0.1, name='LOWERVWAP2', line=dict(color='black')))

    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_25'], mode='lines', opacity=0.15, name='UPPERVWAP2.5', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N25'], mode='lines', opacity=0.15, name='LOWERVWAP2.5', line=dict(color='black')))
   
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_1'], mode='lines', opacity=0.1, name='UPPERVWAP1', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N1'], mode='lines', opacity=0.1, name='LOWERVWAP1', line=dict(color='black')))
            
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_15'], mode='lines', opacity=0.1, name='UPPERVWAP1.5', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N15'], mode='lines', opacity=0.1, name='LOWERVWAP1.5', line=dict(color='black')))

    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_0'], mode='lines', opacity=0.1, name='UPPERVWAP0.5', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df['time'], y=df['STDEV_N0'], mode='lines', opacity=0.1, name='LOWERVWAP0.5', line=dict(color='black')))
    
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
        

    
    '''
    tpCandle =  sorted([i for i in OptionTimeFrame if len(i[10]) > 0 if int(i[4]) < len(df)], key=lambda x: sum([trt[1] for trt in x[10]]),reverse=True)[:8] 

    
    
    est_now = datetime.utcnow() + timedelta(hours=-4)
    start_time = est_now.replace(hour=8, minute=00, second=0, microsecond=0)
    end_time = est_now.replace(hour=17, minute=30, second=0, microsecond=0)
    
    # Check if the current time is between start and end times
    if start_time <= est_now <= end_time:
        ccheck = 0.64
    else:
    
    ccheck = 0.64
    indsAbove = [i for i in OptionTimeFrame if round(i[6],2) >= ccheck and int(i[4]) < len(df) and float(i[2]) >= (sum([i[2]+i[3] for i in OptionTimeFrame]) / len(OptionTimeFrame))]#  float(bms[i[4]])  # and int(i[4]) < len(df) [(len(df)-1,i[1]) if i[0] >= len(df) else i for i in [(int(i[10]),i[1]) for i in sord if i[11] == stockName and i[1] == 'AboveAsk(BUY)']]
    
    indsBelow = [i for i in OptionTimeFrame if round(i[7],2) >= ccheck and int(i[4]) < len(df) and float(i[3]) >= (sum([i[3]+i[2] for i in OptionTimeFrame]) / len(OptionTimeFrame))]#  float(sms[i[4]]) # and int(i[4]) < len(df) imbalance = [(len(df)-1,i[1]) if i[0] >= len(df) else i for i in [(i[10],i[1]) for i in sord if i[11] == stockName and i[13] == 'Imbalance' and i[1] != 'BelowBid(SELL)' and i[1] != 'AboveAsk(BUY)']]
    

    for i in OptionTimeFrame:
        mks = ''
        tobuyss =  sum([x[1] for x in [t for t in i[10] if t[3] == 'B']])
        tosellss = sum([x[1] for x in [t for t in i[10] if t[3] == 'A']])
        #lenbuys = len([t for t in i[10] if t[3] == 'B'])
        #lensells = len([t for t in i[10] if t[3] == 'A'])
        try:
            tpStrings = '(Sell:' + str(tosellss) + '('+str(round(tosellss/(tobuyss+tosellss),2))+') | '+ '(Buy:' + str(tobuyss) + '('+str(round(tobuyss/(tobuyss+tosellss),2))+')) ' + str(tobuyss-tosellss)+'<br>' #str(lenbuys+lensells) +
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
    
    '''     
    #textPerCandle = []
    ctn = 0
    for i in troPerCandle:
        mks = ''
        tobuyss =  sum([x[1] for x in [t for t in i[1] if t[5] == 'B']])
        tosellss = sum([x[1] for x in [t for t in i[1] if t[5] == 'A']])

        try:
            tpStrings = '(Sell:' + str(tosellss) + '('+str(round(tosellss/(tobuyss+tosellss),2))+') | '+ '(Buy:' + str(tobuyss) + '('+str(round(tobuyss/(tobuyss+tosellss),2))+')) '+'<br>' +'Top100OrdersPerCandle: '+ str(tobuyss-tosellss)+'<br>' #str(lenbuys+lensells) +
        except(ZeroDivisionError):
            tpStrings =' '
        
        
        for xp in i[1][:20]:
            #print(xp)
            try:
                taag = 'Buy' if xp[5] == 'B' else 'Sell' if xp[5] == 'A' else 'Mid'
                mks += str(xp[0]) + ' | ' + str(xp[1]) + ' ' + taag + ' ' + xp[6] + '<br>' 
            except(IndexError):
                pass
        
        OptionTimeFrame[ctn].append(mks + tpStrings)
        OptionTimeFrame[ctn].append([tobuyss,round(tobuyss/(tobuyss+tosellss),2),tosellss,round(tosellss/(tobuyss+tosellss),2)])
        #textPerCandle.append([ctn,mks + tpStrings])
        ctn+=1
        
    putCand = [i for i in OptionTimeFrame if int(i[2]) > int(i[3]) if int(i[4]) < len(df)] # if int(i[4]) < len(df)
    callCand = [i for i in OptionTimeFrame if int(i[3]) > int(i[2]) if int(i[4]) < len(df)] # if int(i[4]) < len(df) +i[3]+i[5] +i[2]+i[5]
    MidCand = [i for i in OptionTimeFrame if int(i[3]) == int(i[2]) if int(i[4]) < len(df)]
    
    putCandImb = [i for i in OptionTimeFrame if int(i[12][0]) > int(i[12][2]) and df['percentile_topBuys'][i[4]]> 96 and float(i[12][1]) >= 0.65] #float(i[4]) > 0.65 and df['topBuys'][i[0]] > df['topBuysAvg'][i[0]] and 
    callCandImb = [i for i in OptionTimeFrame if int(i[12][2]) > int(i[12][0])and  df['percentile_topSells'][i[4]]> 96 and float(i[12][3]) >= 0.65]
    
    #putCandImb = [i for i in OptionTimeFrame if int(i[12][0]) > int(i[12][2]) and float(i[12][1]) > 0.65 and int(i[4]) < len(df)]
    #callCandImb = [i for i in OptionTimeFrame if int(i[12][0]) > int(i[12][2]) and float(i[12][1]) > 0.65 and int(i[4]) < len(df)]
    
    if len(MidCand) > 0:
       fig.add_trace(go.Candlestick(
           x=[df['time'][i[4]] for i in MidCand],
           open=[df['open'][i[4]] for i in MidCand],
           high=[df['high'][i[4]] for i in MidCand],
           low=[df['low'][i[4]] for i in MidCand],
           close=[df['close'][i[4]] for i in MidCand],
           increasing={'line': {'color': 'gray'}},
           decreasing={'line': {'color': 'gray'}},
           hovertext=['('+str(i[2])+')'+str(round(i[6],2))+' '+str('Bid')+' '+'('+str(i[3])+')'+str(round(i[7],2))+' Ask' +  '<br>' +i[11]+ 'AllOrders: '+str(i[2]-i[3])   for i in MidCand], #+ i[11] + str(sum([i[10][x][2] for x in i[10]]))
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
            hovertext=['('+str(i[2])+')'+str(round(i[6],2))+' '+str('Bid')+' '+'('+str(i[3])+')'+str(round(i[7],2))+' Ask' + '<br>' +i[11]+ 'AllOrders: '+str(i[2]-i[3]) for i in putCand], #i[11] + str(sum([i[10][x][2] for x in i[10]]))
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
            hovertext=['('+str(i[2])+')'+str(round(i[6],2))+' '+str('Bid')+' '+'('+str(i[3])+')'+str(round(i[7],2))+' Ask' + '<br>' +i[11]+ 'AllOrders: '+str(i[2]-i[3]) for i in callCand], #i[11] + str(sum([i[10][x][2] for x in i[10]]))
            hoverlabel=dict(
                 bgcolor="pink",
                 font=dict(color="black", size=10),
                 ),
            name='' ),
        row=1, col=1)
        trcount+=1
    '''
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
    '''
    #tmpdict = find_spikes(df['weights'])
    if len(callCandImb) > 0:
        fig.add_trace(go.Candlestick(
            x=[df['time'][i[4]] for i in callCandImb],
            open=[df['open'][i[4]] for i in callCandImb],
            high=[df['high'][i[4]] for i in callCandImb],
            low=[df['low'][i[4]] for i in callCandImb],
            close=[df['close'][i[4]] for i in callCandImb],
            increasing={'line': {'color': 'crimson'}},
            decreasing={'line': {'color': 'crimson'}}, 
            hovertext=['('+str(OptionTimeFrame[i[4]][2])+')'+str(round(OptionTimeFrame[i[4]][6],2))+' '+str('Bid')+' '+'('+str(OptionTimeFrame[i[4]][3])+')'+str(round(OptionTimeFrame[i[4]][7],2))+' Ask' + '<br>' +str(OptionTimeFrame[i[4]][11])+ 'AllOrders: '+ str(OptionTimeFrame[i[4]][2]-OptionTimeFrame[i[4]][3]) for i in callCandImb],
            hoverlabel=dict(
                 bgcolor="crimson",
                 font=dict(color="white", size=10),
                 ),
            name='Sellimbalance' ),
        row=1, col=1)
        trcount+=1
        
    if len(putCandImb) > 0:
        fig.add_trace(go.Candlestick(
            x=[df['time'][i[4]] for i in putCandImb],
            open=[df['open'][i[4]] for i in putCandImb],
            high=[df['high'][i[4]] for i in putCandImb],
            low=[df['low'][i[4]] for i in putCandImb],
            close=[df['close'][i[4]] for i in putCandImb],
            increasing={'line': {'color': '#16FF32'}},
            decreasing={'line': {'color': '#16FF32'}},
            hovertext=['('+str(OptionTimeFrame[i[4]][2])+')'+str(round(OptionTimeFrame[i[4]][6],2))+' '+str('Bid')+' '+'('+str(OptionTimeFrame[i[4]][3])+')'+str(round(OptionTimeFrame[i[4]][7],2))+' Ask' + '<br>' +str(OptionTimeFrame[i[4]][11])+ 'AllOrders: '+ str(OptionTimeFrame[i[4]][2]-OptionTimeFrame[i[4]][3]) for i in putCandImb], 
            hoverlabel=dict(
                 bgcolor="#2CA02C",
                 font=dict(color="white", size=10),
                 ),
            name='BuyImbalance' ),
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
    if '19:00:00' in df['time'].values or '19:01:00' in df['time'].values:
        if '19:00:00' in df['time'].values:
            opstr = '19:00:00'
        elif '19:01:00' in df['time'].values:
            opstr = '19:01:00'
            
        fig.add_vline(x=df[df['time'] == opstr].index[0], line_width=2, line_dash="dash", line_color="green", annotation_text='Toyko Open', annotation_position='top right', row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df['time'],
                                y= [df['open'][df[df['time'] == '19:00:00'].index[0]]]*len(df['time']) ,
                                line_color='black',
                                text = str(df['open'][df[df['time'] == '19:00:00'].index[0]]),
                                textposition="bottom left",
                                name='Toyko Open',
                                showlegend=False,
                                visible=False,
                                mode= 'lines',
                                ))
        
        if '01:00:00' in df['time'].values:
            fig.add_vline(x=df[df['time'] == '01:00:00'].index[0], line_width=2, line_dash="dash", line_color="red", annotation_text='Sydney Close', annotation_position='top left', row=1, col=1)
            
            tempDf = df.loc[:df[df['time'] == '01:00:00'].index[0]]
            min_low = tempDf['low'].min()
            max_high = tempDf['high'].max()
            fig.add_trace(go.Scatter(x=df['time'],
                                    y= [min_low]*len(df['time']) ,
                                    line_color='black',
                                    text = str(min_low),
                                    textposition="bottom left",
                                    name='Sydney Low',
                                    showlegend=False,
                                    visible=False,
                                    mode= 'lines',
                                    ))
            
            fig.add_trace(go.Scatter(x=df['time'],
                                    y= [max_high]*len(df['time']) ,
                                    line_color='black',
                                    text = str(max_high),
                                    textposition="bottom left",
                                    name='Sydney High',
                                    showlegend=False,
                                    visible=False,
                                    mode= 'lines',
                                    ))
            
            fig.add_trace(go.Scatter(x=df['time'],
                                    y= [df['close'][df[df['time'] == '00:58:00'].index[0]]]*len(df['time']) ,
                                    line_color='black',
                                    text = str(df['close'][df[df['time'] == '00:58:00'].index[0]]),
                                    textposition="bottom left",
                                    name='Sydney Close',
                                    showlegend=False,
                                    visible=False,
                                    mode= 'lines',
                                    ))
            
            

        if '02:00:00' in df['time'].values:
            fig.add_vline(x=df[df['time'] == '02:00:00'].index[0], line_width=2, line_dash="dash", line_color="green", annotation_text='London Open', annotation_position='top right', row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df['time'],
                                    y= [df['open'][df[df['time'] == '02:00:00'].index[0]]]*len(df['time']) ,
                                    line_color='black',
                                    text = str(df['open'][df[df['time'] == '02:00:00'].index[0]]),
                                    textposition="bottom left",
                                    name='London Open',
                                    showlegend=False,
                                    visible=False,
                                    mode= 'lines',
                                    ))
            
    
        if '04:00:00' in df['time'].values:
            fig.add_vline(x=df[df['time'] == '04:00:00'].index[0], line_width=2, line_dash="dash", line_color="red", annotation_text='Toyko Close', annotation_position='top right', row=1, col=1)
            
            tempDf = df.loc[df[df['time'] == opstr].index[0]:df[df['time'] == '04:00:00'].index[0]]
            max_high = tempDf['high'].max()
            min_low = tempDf['low'].min()
            fig.add_trace(go.Scatter(x=df['time'],
                                    y= [max_high]*len(df['time']) ,
                                    line_color='black',
                                    text = str(max_high),
                                    textposition="bottom left",
                                    name='Toyko High',
                                    showlegend=False,
                                    visible=False,
                                    mode= 'lines',
                                    ))
            
            fig.add_trace(go.Scatter(x=df['time'],
                                    y= [min_low]*len(df['time']) ,
                                    line_color='black',
                                    text = str(min_low),
                                    textposition="bottom left",
                                    name='Toyko Low',
                                    showlegend=False,
                                    visible=False,
                                    mode= 'lines',
                                    ))
            
            
            fig.add_trace(go.Scatter(x=df['time'],
                                    y= [df['close'][df[df['time'] == '03:58:00'].index[0]]]*len(df['time']) ,
                                    line_color='black',
                                    text = str(df['close'][df[df['time'] == '03:58:00'].index[0]]),
                                    textposition="bottom left",
                                    name='Toyko Close',
                                    showlegend=False,
                                    visible=False,
                                    mode= 'lines',
                                    ))
            
            

            
        if '08:00:00' in df['time'].values:
            fig.add_vline(x=df[df['time'] == '08:00:00'].index[0], line_width=2, line_dash="dash", line_color="green", annotation_text='NewYork Open', annotation_position='top left', row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df['time'],
                                    y= [df['open'][df[df['time'] == '08:00:00'].index[0]]]*len(df['time']) ,
                                    line_color='black',
                                    text = str(df['open'][df[df['time'] == '08:00:00'].index[0]]),
                                    textposition="bottom left",
                                    name='NewYork Open',
                                    showlegend=False,
                                    visible=False,
                                    mode= 'lines',
                                    ))
            
    
        if '11:00:00' in df['time'].values:
            fig.add_vline(x=df[df['time'] == '11:00:00'].index[0], line_width=2, line_dash="dash", line_color="red", annotation_text='London Close', annotation_position='top left', row=1, col=1)
            
            tempDf = df.loc[df[df['time'] == '02:00:00'].index[0]:df[df['time'] == '11:00:00'].index[0]]
            max_high = tempDf['high'].max()
            min_low = tempDf['low'].min()
            fig.add_trace(go.Scatter(x=df['time'],
                                    y= [max_high]*len(df['time']) ,
                                    line_color='black',
                                    text = str(max_high),
                                    textposition="bottom left",
                                    name='London High',
                                    showlegend=False,
                                    visible=False,
                                    mode= 'lines',
                                    ))
            
            fig.add_trace(go.Scatter(x=df['time'],
                                    y= [min_low]*len(df['time']) ,
                                    line_color='black',
                                    text = str(min_low),
                                    textposition="bottom left",
                                    name='London Low',
                                    showlegend=False,
                                    visible=False,
                                    mode= 'lines',
                                    ))
            
            fig.add_trace(go.Scatter(x=df['time'],
                                    y= [df['close'][df[df['time'] == '10:58:00'].index[0]]]*len(df['time']) ,
                                    line_color='black',
                                    text = str(df['close'][df[df['time'] == '10:58:00'].index[0]]),
                                    textposition="bottom left",
                                    name='London Close',
                                    showlegend=False,
                                    visible=False,
                                    mode= 'lines',
                                    ))
            
            
    '''  

    '''      
    if '02:00:00' in df['time'].values:
         fig.add_vline(x=df[df['time'] == '02:00:00'].index[0], line_width=2, line_dash="dash", line_color="green", annotation_text='London Open', annotation_position='top right', row=1, col=1)
    
         
    if '04:00:00' in df['time'].values:
        fig.add_vline(x=df[df['time'] == '04:00:00'].index[0], line_width=2, line_dash="dash", line_color="red", annotation_text='Toyko Close', annotation_position='top right', row=1, col=1)
     
    
    if '08:00:00' in df['time'].values:
        fig.add_vline(x=df[df['time'] == '08:00:00'].index[0], line_width=2, line_dash="dash", line_color="green", annotation_text='NewYork Open', annotation_position='top left', row=1, col=1)
    '''    
    
    '''
    difList = [(i[2]-i[3],i[0]) for i in OptionTimeFrame]
    coll = [     'teal' if i[0] > 0
                else 'crimson' if i[0] < 0
                else 'gray' for i in difList]
    fig.add_trace(go.Bar(x=pd.Series([i[1] for i in difList]), y=pd.Series([i[0] for i in difList]), marker_color=coll), row=2, col=1)
    
    
    
    fig.add_trace(go.Bar(x=df['time'], y=df['Histogram'], marker_color='black'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['MACD'], marker_color='blue'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['Signal'], marker_color='red'), row=2, col=1)
    
    #fig.add_trace(go.Bar(x=pd.Series([i[0] for i in troInterval]), y=pd.Series([i[5] for i in troInterval]), marker_color='teal'), row=2, col=1)
    #fig.add_trace(go.Bar(x=pd.Series([i[0] for i in troInterval]), y=pd.Series([i[6] for i in troInterval]), marker_color='crimson'), row=2, col=1)
    
    
    coll = [    'teal' if i > 0
                else 'crimson' if i < 0
                else 'gray' for i in df['imbalance']]
    fig.add_trace(go.Bar(x=df['time'], y=df['imbalance'], marker_color=coll), row=3, col=1)
    
    
    #fig.add_trace(go.Bar(x=df['time'], y=pd.Series([i[5] for i in troInterval]), marker_color='teal'), row=3, col=1)
    #fig.add_trace(go.Bar(x=df['time'], y=pd.Series([i[6] for i in troInterval]), marker_color='crimson'), row=3, col=1)
    
    
    if 'smoothed_derivative' in df.columns:
        colors = ['maroon']
        for val in range(1,len(df['smoothed_derivative'])):
            if df['smoothed_derivative'][val] > 0:
                color = 'teal'
                if df['smoothed_derivative'][val] > df['smoothed_derivative'][val-1]:
                    color = '#54C4C1' 
            else:
                color = 'maroon'
                if df['smoothed_derivative'][val] < df['smoothed_derivative'][val-1]:
                    color='crimson' 
            colors.append(color)
        fig.add_trace(go.Bar(x=df['time'], y=df['smoothed_derivative'], marker_color=colors), row=3, col=1)
        
        

    if 'POCDistanceEMA' in df.columns:
        colors = ['maroon']
        for val in range(1,len(df['POCDistanceEMA'])):
            if df['POCDistanceEMA'][val] > 0:
                color = 'teal'
                if df['POCDistanceEMA'][val] > df['POCDistanceEMA'][val-1]:
                    color = '#54C4C1' 
            else:
                color = 'maroon'
                if df['POCDistanceEMA'][val] < df['POCDistanceEMA'][val-1]:
                    color='crimson' 
            colors.append(color)
        fig.add_trace(go.Bar(x=df['time'], y=df['POCDistanceEMA'], marker_color=colors), row=2, col=1)
    '''  
    fig.add_trace(go.Bar(x=df['time'], y=df['percentile_topBuys'], marker_color='teal'), row=2, col=1)
    fig.add_trace(go.Bar(x=df['time'], y=df['percentile_topSells'], marker_color='crimson'), row=2, col=1)
        
        

    
    #fig.add_trace(go.Bar(x=df['time'], y=pd.Series([i[2] for i in OptionTimeFrame]), marker_color='teal'), row=3, col=1)
    #fig.add_trace(go.Bar(x=df['time'], y=pd.Series([i[3] for i in OptionTimeFrame]), marker_color='crimson'), row=3, col=1)
        
    
    #fig.add_trace(go.Scatter(x=df['time'], y=df['kalman_velocity'], mode='lines',name='kalman_velocity'), row=3, col=1)
    #fig.add_trace(go.Scatter(x=df['time'], y=df['kalman_acceleration'], mode='lines',name='kalman_acceleration'), row=3, col=1)
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
    fig.add_trace(go.Bar(x=df['time'], y=df['Momentum'], marker_color =colors ), row=3, col=1)
    
    
    
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
            header=dict(values=["Time", "Buyers", "Buyers Change", "Sellers", "Sellers Change","Buyers per Interval", "Sellers per Interval"], font=dict(size=9)),
            cells=dict(values=transposed_data, fill_color=color_matrix, font=dict(color=textColor_matrix,size=9)),  # Transpose data to fit the table
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
    stillbuy = False
    stillsell = False

    if df['buy_signal'][0]:
        stillbuy = True
        fig.add_annotation(x=df['time'][0], y=df['close'][0],
                        text='<b>' + 'Buy' + '</b>',
                        showarrow=True,
                        arrowhead=4,
                        arrowcolor='green',
                        font=dict(
                            size=10,
                            color='green',
                        ),)

    if df['sell_signal'][0]:
        stillsell = True
        fig.add_annotation(x=df['time'][0], y=df['close'][0],
                        text='<b>' + 'Sell' + '</b>',
                        showarrow=True,
                        arrowhead=4,
                        arrowcolor='red',
                        font=dict(
                            size=10,
                            color='red',
                        ),)
        

    for p in range(1, len(df)):  # Start from 1 to compare with the previous row
        # Check if the value of cross_above changed from the previous row
        if df['buy_signal'][p] != df['buy_signal'][p-1] and not stillbuy :
            # Add 'Buy' only if cross_above is True after the change
            stillbuy = True
            stillsell = False
            if df['buy_signal'][p]:
                fig.add_annotation(x=df['time'][p], y=df['close'][p],
                                    text='<b>' + 'Buy' + '</b>',
                                    showarrow=True,
                                    arrowhead=4,
                                    arrowcolor='green',
                                    font=dict(
                                        size=12,
                                        color='green',
                                    ),)
        
        # Check if the value of cross_below changed from the previous row
        if df['sell_signal'][p] != df['sell_signal'][p-1] and not stillsell :
            # Add 'Sell' only if cross_below is True after the change
            stillsell = True
            stillbuy = False
            if df['sell_signal'][p]:
                fig.add_annotation(x=df['time'][p], y=df['close'][p],
                                    text='<b>' + 'Sell' + '</b>',
                                    showarrow=True,
                                    arrowhead=4,
                                    arrowcolor='red',
                                    font=dict(
                                        size=12,
                                        color='red'
                                    ),)
    
    
    '''
    stillbuy = False
    stillsell = False
    for p in range(1, len(df)):  # Start from 1 to compare with the previous row
        if 'cross_above' in df.columns:
            # Check if the value of cross_above changed from the previous row
            if df['cross_above'][p] != df['cross_above'][p-1] and not stillbuy :
                # Add annotation only if cross_above is True after the change
                stillbuy = True
                stillsell = False
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
            if df['cross_below'][p] != df['cross_below'][p-1]  and not stillsell :
                # Add annotation only if cross_above is True after the change
                stillsell = True
                stillbuy = False
                if df['cross_below'][p]:
                    fig.add_annotation(x=df['time'][p], y=df['close'][p],
                                       text='<b>' + 'Sell' + '</b>',
                                       showarrow=True,
                                       arrowhead=4,
                                       font=dict(
                                           size=8,
                                       ),)
    '''


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


def least_squares_filter(data, window_size, poly_order=2):
    """
    Applies a least squares polynomial fit to a moving window of the data.
    
    Parameters:
    data (pd.Series): The input data series.
    window_size (int): The number of data points in the moving window.
    poly_order (int): The order of the polynomial for curve fitting.
    
    Returns:
    pd.Series: The filtered data series.
    """
    half_window = window_size // 2
    filtered_data = []

    for i in range(len(data)):
        # Define the window boundaries
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)

        # Get the window of data
        window_data = data[start:end]
        x = np.arange(len(window_data))
        
        # Fit a polynomial of the specified order to the window
        coefficients = np.polyfit(x, window_data, poly_order)
        poly_func = np.poly1d(coefficients)
        
        # Estimate the current value using the polynomial fit
        filtered_value = poly_func(half_window if i >= half_window else i)
        filtered_data.append(filtered_value)

    return pd.Series(filtered_data, index=data.index)


def least_squares_filter_real_time(data, window_size, poly_order=2):
    filtered_data = []

    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        window_data = data[start:i + 1]
        x = np.arange(len(window_data))

        try:
            # Fit a polynomial to the trailing window
            coefficients = np.polyfit(x, window_data, poly_order)
            poly_func = np.poly1d(coefficients)
            filtered_value = poly_func(len(window_data) - 1)
        except np.linalg.LinAlgError:
            # Handle the error by falling back to a simple estimate
            filtered_value = window_data.iloc[-1]  # or use np.mean(window_data)

        filtered_data.append(filtered_value)

    return pd.Series(filtered_data, index=data.index)


def ewm_median(series, span):
    alpha = 2 / (span + 1)  # Exponential smoothing parameter
    weights = (1 - alpha) ** np.arange(len(series))[::-1]  # Reverse weights
    medians = []
    
    for i in range(len(series)):
        current_window = series.iloc[max(0, i - span + 1):i + 1]
        weighted_values = current_window * weights[-len(current_window):]
        weighted_values = weighted_values.dropna()
        medians.append(np.median(weighted_values))
    
    return pd.Series(medians, index=series.index)


from concurrent.futures import ThreadPoolExecutor    
def download_data(bucket_name, blob_name):
    blob = Blob(blob_name, bucket_name)
    return blob.download_as_text()   


def download_daily_data(bucket, stkName):
    """Download and process the daily data."""
    blob = Blob('Daily' + stkName, bucket)
    buffer = io.BytesIO()
    blob.download_to_file(buffer)
    buffer.seek(0)
    
    # Read CSV and keep only required columns
    columns_to_keep = ['open', 'high', 'low', 'close', 'volume']
    prevDf = pd.read_csv(buffer, usecols=columns_to_keep)
    
    return prevDf

def exponential_median(data, span):
    alpha = 2 / (span + 1)
    exp_median = []
    median_val = data[0]  # Initialize with the first value
    
    for val in data:
        median_val = (1 - alpha) * median_val + alpha * val
        exp_median.append(median_val)
    
    return exp_median


def mean_shift_filter(data, bandwidth, max_iterations=10, tolerance=1e-4):
    """
    Apply a mean shift filter to reduce noise in time series data.

    Parameters:
    - data: pd.Series - Input time series data (e.g., '1ema').
    - bandwidth: float - Size of the neighborhood for mean calculation.
    - max_iterations: int - Maximum number of iterations for the mean shift.
    - tolerance: float - Convergence tolerance for stopping the iterations.

    Returns:
    - pd.Series - Smoothed data.
    """
    smoothed = data.copy()
    for iteration in range(max_iterations):
        shifted = smoothed.copy()
        for i in range(len(data)):
            # Determine the neighborhood
            start = max(0, i - bandwidth)
            end = min(len(data), i + bandwidth)
            # Update value based on the mean of the neighborhood
            shifted.iloc[i] = smoothed.iloc[start:end].mean()
        
        # Check for convergence
        if np.abs(shifted - smoothed).max() < tolerance:
            break
        smoothed = shifted

    return smoothed


def mean_shift_filter_realtime(data, bandwidth, max_iterations=5, tolerance=1e-4):
    """
    Real-time Mean Shift Filter to smooth time series without relying on future data.

    Parameters:
    - data: pd.Series - Input time series data (e.g., '1ema').
    - bandwidth: int - Number of past points to consider for smoothing.
    - max_iterations: int - Maximum iterations for the mean shift process.
    - tolerance: float - Convergence tolerance for stopping the iterations.

    Returns:
    - pd.Series - Smoothed data.
    """
    smoothed = data.copy()
    for iteration in range(max_iterations):
        shifted = smoothed.copy()
        for i in range(len(data)):
            # Use only past and current points within the bandwidth
            start = max(0, i - bandwidth)
            window = smoothed.iloc[start:i + 1]  # Causal window
            # Update the current point based on the mean of the causal window
            shifted.iloc[i] = window.mean()
        
        # Check for convergence
        if np.abs(shifted - smoothed).max() < tolerance:
            break
        smoothed = shifted

    return smoothed

'''
#from sklearn.linear_model import LinearRegression
def linear_regression_smoothing(data, window_size):
    """
    Smooth a line using linear regression over a sliding window.
    
    Parameters:
    - data: pd.Series - The input data series.
    - window_size: int - The size of the sliding window.
    
    Returns:
    - pd.Series - Smoothed data.
    """
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(data)):
        # Define the window boundaries
        start = max(0, i - window_size + 1)
        end = min(len(data), i + half_window + 1)     
                               
        # Extract the windowed data
        x = np.arange(start, end).reshape(-1, 1)
        y = data[start:end].values
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(x, y)
        
        # Predict the value at the current index
        smoothed_value = model.predict([[i]])
        smoothed.append(smoothed_value[0])
    
    return pd.Series(smoothed, index=data.index)
'''
def random_walk_filter(data, alpha=0.5):
    """
    Apply a Random Walk Filter to smooth the data.

    Parameters:
    - data: pandas Series or numpy array of the original time-series data.
    - alpha: Smoothing factor, between 0 and 1.

    Returns:
    - Smoothed data as a pandas Series (if input is a pandas Series).
    """
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]  # Initialize with the first data point

    for t in range(1, len(data)):
        # Update rule for the random walk filter
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
    
    return pd.Series(smoothed, index=data.index) if isinstance(data, pd.Series) else smoothed

from pykalman import KalmanFilter

# Function to apply the Kalman filter
def apply_kalman_filter(data, transition_covariance, observation_covariance):
    """
    Applies Kalman Filter to smooth a time-series dataset.

    Parameters:
    - data (array-like): Time series to be smoothed (e.g., EMA or price).

    Returns:
    - np.array: Smoothed time series.
    """
    # Initialize the Kalman Filter
    kf = KalmanFilter(
        transition_matrices=[1],  # State transition matrix
        observation_matrices=[1],  # Observation matrix
        initial_state_mean=data[0],  # Initial state estimate
        initial_state_covariance=1,  # Initial covariance estimate
        observation_covariance=observation_covariance,  # Measurement noise covariance
        transition_covariance=transition_covariance  # Process noise covariance
    )

    # Use the Kalman filter to estimate the state
    state_means, _ = kf.filter(data)

    return state_means

def calculate_rsi(data, window=14):
    """
    Calculate the Relative Strength Index (RSI).
    
    Parameters:
        data (pd.Series): A pandas Series of prices (e.g., closing prices).
        window (int): The lookback period for RSI calculation (default is 14).
    
    Returns:
        pd.Series: The RSI values.
    """
    # Calculate price changes
    delta = data.diff()
    
    # Separate positive and negative gains
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Average gain and loss using exponential moving average
    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()
    
    # Compute relative strength (RS)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return pd.Series(rsi, index=data.index)


def compute_atr(df, period=14):
    """
    Compute the Average True Range (ATR) for a given DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame with 'high', 'low', and 'close' columns.
        period (int): The period over which to calculate the ATR (default: 14).

    Returns:
        pd.Series: A Pandas Series containing the ATR values.
    """
    # Calculate True Range (TR)
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift(1))
    df['low_close'] = abs(df['low'] - df['close'].shift(1))
    
    df['TR'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)

    # Calculate ATR
    atr = df['TR'].rolling(window=period, min_periods=1).mean()
    
    # Clean up temporary columns
    df.drop(['high_low', 'high_close', 'low_close', 'TR'], axis=1, inplace=True)
    
    return atr

def calculate_slope_to_row(index, values):
    x = np.arange(index + 1)  # Indices from 0 to the current row
    y = values[: index + 1]  # Values from 0 to the current row
    slope, _, _, _, _ = linregress(x, y)
    return round(math.degrees(math.atan(slope)), 3)

def calculate_polyfit_slope_to_row(index, values):
    try:
        x = np.arange(index + 1)
        y = values[: index + 1]
        poly = Polynomial.fit(x, y, 1)
        return round(poly.coef[1], 3)
    except np.linalg.LinAlgError:
        return 0.0 

def calculate_slope_rolling(index, values, window_size):
    """
    Calculate the slope using a rolling window.
    
    Parameters:
    - index: The current index in the DataFrame.
    - values: The column values to calculate the slope on.
    - window_size: The size of the rolling window.
    
    Returns:
    - The slope in degrees for the given rolling window.
    """
    start = max(0, index - window_size + 1)
    x = np.arange(start, index + 1)
    y = values[start: index + 1]
    slope, _, _, _, _ = linregress(x, y)
    return round(math.degrees(math.atan(slope)), 3)


def calculate_polyfit_slope_rolling(index, values, window_size):
    """
    Calculate the slope using a rolling window and Polynomial fit.
    
    Parameters:
    - index: The current index in the DataFrame.
    - values: The column values to calculate the slope on.
    - window_size: The size of the rolling window.
    
    Returns:
    - The slope from the Polynomial fit for the given rolling window.
    """
    try:
        start = max(0, index - window_size + 1)
        x = np.arange(start, index + 1)
        y = values[start: index + 1]
        poly = Polynomial.fit(x, y, 1)
        return round(poly.coef[1], 3)
    except np.linalg.LinAlgError:
        return 0.0
    
    
def calculate_slope_weighted(index, values, window_size):
    start = max(0, index - window_size + 1)
    x = np.arange(start, index + 1)
    y = values[start: index + 1]
    weights = np.exp(-0.1 * (x - x[-1]))  # Apply exponential decay weights
    slope, _, _, _, _ = linregress(x, y * weights)
    return round(math.degrees(math.atan(slope)), 3)


def calculate_polyfit_slope_weighted(index, values, window_size):
    try:
        start = max(0, index - window_size + 1)
        x = np.arange(start, index + 1)
        y = values[start: index + 1]
        weights = np.exp(-0.1 * (x - x[-1]))  # Apply exponential decay weights
        poly = Polynomial.fit(x, y * weights, 1)
        return round(poly.coef[1], 3)
    except np.linalg.LinAlgError:
        return 0.0

def vwapDistanceCheckBuy(df):
    if abs(df['vwapDistance']) <= 0.25:
        return df['smoothed_1ema'] > df['vwap']
    return True

def vwapDistanceCheckSell(df):
    if abs(df['vwapDistance']) <= 0.25:#0.09
        return df['smoothed_1ema'] < df['vwap']
    return True


def uppervwapDistanceCheckBuy(df):
    if abs(df['uppervwapDistance']) <= 0.25:
        return df['smoothed_1ema'] > df['uppervwapAvg']
    return True

def uppervwapDistanceCheckSell(df):
    if abs(df['uppervwapDistance']) <= 0.25:
        return df['smoothed_1ema'] < df['uppervwapAvg']
    return True

def lowervwapDistanceCheckBuy(df):
    if abs(df['lowervwapDistance']) <= 0.25:
        return df['smoothed_1ema'] > df['lowervwapAvg']
    return True

def lowervwapDistanceCheckSell(df):
    if abs(df['lowervwapDistance']) <= 0.25:
        return df['smoothed_1ema'] < df['lowervwapAvg'] 
    return True

def vwapAvgDistanceCheckBuy(df):
    if abs(df['vwapAvgDistance']) <= 0.25:
        return df['smoothed_1ema'] > df['vwapAvg']
    return True

def vwapAvgDistanceCheckSell(df):
    if abs(df['vwapAvgDistance']) <= 0.25:
        return df['smoothed_1ema'] < df['vwapAvg'] 
    return True

def double_exponential_smoothing(X, alpha, beta):
    """
    Applies double exponential smoothing (Holt's linear trend method) to a pandas Series.

    Parameters:
    - X (pandas.Series or pandas.DataFrame column): Time series data.
    - alpha (float): Smoothing factor for the level (0 < alpha < 1).
    - beta (float): Smoothing factor for the trend (0 < beta < 1).

    Returns:
    - numpy.ndarray: Smoothed series forecast.
    """
    # If X is a DataFrame column (Series), convert it to a NumPy array
    if isinstance(X, (pd.Series, pd.DataFrame)):
        # If X is a DataFrame, assume the first column is the time series data.
        X = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else X
        X = X.values

    # Ensure X is now a NumPy array
    X = np.asarray(X)
    
    # Check that X has at least 2 data points
    if X.shape[0] < 2:
        raise ValueError("Input series X must contain at least two elements.")
    
    # Initialize arrays for the smoothed values (S), level (A), and trend (B)
    S, A, B = [np.zeros(len(X)) for _ in range(3)]
    
    # Initial values: set initial level as the first observation,
    # and initial trend as the difference between the first two observations.
    S[0] = X[0]
    A[0] = X[0]
    B[0] = X[1] - X[0]
    
    # Main loop: update the level, trend, and forecast for each time step.
    for t in range(1, len(X)):
        A[t] = alpha * X[t] + (1 - alpha) * S[t - 1]
        B[t] = beta * (A[t] - A[t - 1]) + (1 - beta) * B[t - 1]
        S[t] = A[t] + B[t]
    
    return S
   
symbolNumList = ['5002', '42288528', '42002868', '37014', '1551','19222', '899', '42001620', '4127884', '5556', '42010915', '148071', '65', '42004880', '42002512']
symbolNameList = ['ES', 'NQ', 'YM','CL', 'GC', 'HG', 'NG', 'RTY', 'PL',  'SI', 'MBT', 'NIY', 'NKD', 'MET', 'UB']

intList = [str(i) for i in range(3,30)]

vaildClust = [str(i) for i in range(0,200)]

vaildTPO = [str(i) for i in range(1,500)]

covarianceList = [str(round(i, 2)) for i in [x * 0.01 for x in range(1, 1000)]]

gclient = storage.Client(project="stockapp-401615")
bucket = gclient.get_bucket("stockapp-storage")
#abg = download_data(bucket, 'DailyNQ')

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
#from filterpy.kalman import KalmanFilter
from google.api_core.exceptions import NotFound
from scipy.signal import filtfilt, butter, lfilter
from dash import Dash, dcc, html, Input, Output, callback, State
initial_inter = 1800000  # Initial interval #210000#250000#80001
subsequent_inter = 80000  # Subsequent interval
app = Dash()
app.title = "EnVisage"
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
            html.Div(id='container-button-basic', children="Enter a symbol from ES, NQ", style=styles['label']),
        ], style=styles['sub_container']),
        dcc.Store(id='stkName-value'),
        
        html.Div([
            dcc.Input(id='input-on-interv', type='text', style=styles['input']),
            html.Button('Submit', id='submit-interv', n_clicks=0, style=styles['button']),
            html.Div(id='interv-button-basic',children="Enter interval from 3-30, Default 10 mins", style=styles['label']),
        ], style=styles['sub_container']),
        dcc.Store(id='interv-value'),
        
        
    ], style=styles['main_container']),
    
    
    dcc.Store(id='data-store'),
    dcc.Store(id='previous-interv'),
    dcc.Store(id='previous-stkName'),
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
        return 'The input symbol '+str(value)+" is not accepted please try different symbol from  |'ES', 'NQ'|", 'The input symbol was '+str(value)+" is not accepted please try different symbol  |'ESH4' 'NQH4' 'CLG4' 'GCG4' 'NGG4' 'HGH4' 'YMH4' 'BTCZ3' 'RTYH4'|  "

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
        Output('previous-interv', 'data'),
        Output('interval', 'interval')],
    [Input('interval', 'n_intervals')],
    [State('stkName-value', 'data'),
        State('interv-value', 'data'),
        State('data-store', 'data'),
        State('previous-stkName', 'data'),
        State('previous-interv', 'data'),
        State('interval-time', 'data'),
        
    ],
)
    
def update_graph_live(n_intervals, sname, interv, stored_data, previous_stkName, previous_interv, interval_time): #interv
    
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
        interv = '10'
        
    clustNum = '20'
        
    tpoNum = '500'

    curvature = '0.6'
    
    curvatured2 = '0.7'

    
        
    if sname != previous_stkName or interv != previous_interv:
        stored_data = None
        


    
        
        
    print('inFunction '+sname)	
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        #if sname != previous_stkName:
        # Download everything when stock name changes
        futures = [
            executor.submit(download_data, bucket, 'FuturesOHLC' + str(symbolNum)),
            executor.submit(download_data, bucket, 'FuturesTrades' + str(symbolNum)),]
            #executor.submit(download_daily_data, bucket, stkName)]
        
        FuturesOHLC, FuturesTrades = [future.result() for future in futures] #, prevDf
        '''
        else:
            # Skip daily data when stock name is unchanged
            futures = [
                executor.submit(download_data, bucket, 'FuturesOHLC' + str(symbolNum)),
                executor.submit(download_data, bucket, 'FuturesTrades' + str(symbolNum))]
            
            FuturesOHLC, FuturesTrades = [future.result() for future in futures]
            prevDf = None  # No new daily data download
        '''
    
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
    # Apply TEMA calculation to the DataFrame
    if prevDf is not None:
        columns_to_keep = ['open', 'high', 'low', 'close', 'volume']
        prevDf_filtered = prevDf[columns_to_keep]
        #df_filtered = df[columns_to_keep]
        
        # Combine the two DataFrames row-wise
        #combined_df = pd.concat([prevDf_filtered, df_filtered], ignore_index=True)
        
        #vwapCum(combined_df)
        #PPPCum(combined_df)    
    '''

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
    
        
    hs = historV1(df,int(tpoNum),{},AllTrades,[])
    
    va = valueAreaV3(hs[0])
    
    df[clustNum+'ema'] = df['close'].ewm(span=int(clustNum), adjust=False).mean()

    x = np.array([i for i in range(len(df))])
    y = np.array([i for i in df[clustNum+'ema']])
    
    

    # Simple interpolation of x and y
    f = interp1d(x, y)
    x_fake = np.arange(0.1, len(df)-1, 1)

    # derivative of y with respect to x
    df_dx = derivative(f, x_fake, dx=1e-6)
    df_dx = np.pad(df_dx, (1, 0), 'edge')
    
    df['derivative'] = df_dx
    
    

    # Smooth the derivative using Gaussian filter
    df['smoothed_derivative'] = df['derivative']#gaussian_filter1d(df['derivative'], sigma=int(1)) #df['derivative'].ewm(span=int(5), adjust=False).mean()
    #df['derivative'] = df['derivative'].ewm(span=int(4), adjust=False).mean()
    #df['derivative'] = np.gradient(df[clustNum+'ema'])
    
    #df['avg_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    
    
    '''
    
    
    
    kf = KalmanFilter(dim_x=3, dim_z=1)

    # State transition matrix (F) for position, velocity, and acceleration
    dt = 1  # Time step
    kf.F = np.array([[1, dt, 0.5 * dt**2],
                     [0, 1, dt],
                     [0, 0, 1]])
    
    # Measurement function (H): We only observe the position (30 EMA)
    kf.H = np.array([[1, 0, 0]])
    
    # Covariance matrix (P): Initial uncertainty in the estimates
    kf.P *= 1000  # High initial uncertainty in all states
    
    # Measurement noise (R): Uncertainty in observations
    kf.R = np.array([[5]])
    
    # Process noise (Q): Uncertainty in the model dynamics
    kf.Q = np.array([[0.1, 0, 0],
                     [0, 0.1, 0],
                     [0, 0, 0.1]])
    
    # Initial state estimate: [position, velocity, acceleration]
    kf.x = np.array([[df[clustNum+'ema'].iloc[0]], [0], [0]])  # Start with initial position, zero velocity, and zero acceleration
    
    # Lists to store the estimates
    positions = []
    velocities = []
    accelerations = []
    
    # Iterate over each data point in '30ema'
    for emaa in df[clustNum+'ema']:
        kf.predict()  # Predict next state
        kf.update([emaa])  # Update state with new measurement (position)
        
        # Store the estimated position, velocity, and acceleration
        positions.append(kf.x[0][0])  # Position estimate (smoothed 30 EMA)
        velocities.append(kf.x[1][0])  # Velocity (first derivative) estimate
        accelerations.append(kf.x[2][0])  # Acceleration (second derivative) estimate
    
    # Add the estimates to the DataFrame
    df['kalman_position'] = positions
    df['kalman_velocity'] = velocities
    df['kalman_acceleration'] = accelerations
    
    df['kalman_velocity'] = df['kalman_velocity'].ewm(span=1, adjust=False).mean()
    '''
    
    '''
    order = 1     # Filter order
    cutoff = 0.04  # Cutoff frequency, adjust based on desired smoothness
    
    # Design a low-pass Butterworth filter
    b, a = butter(N=order, Wn=cutoff, btype='low')
    
    # Apply the filtfilt filter to df['30ema']
    df['filtfilt'] = filtfilt(b, a, df['close'])
    #df['lfilter'] = lfilter(b, a, df['close'])
    
    from scipy.signal import lfilter_zi

    # Get initial conditions for the filter
    zi = lfilter_zi(b, a) * df['close'].iloc[0]  # Scale initial conditions to match data
    
    # Apply lfilter with the initialized conditions
    df['lfilter'], _ = lfilter(b, a, df['close'], zi=zi)
    '''

    window_size = 3  # Define the window size
    poly_order = 1   # Polynomial order (e.g., 2 for quadratic fit)
    #df['lsfreal_time'] = least_squares_filter_real_time(df['close'], window_size, poly_order)
    df['lsf'] = least_squares_filter(df['close'], window_size, poly_order)
    #df['lsf'] = df['lsf'].ewm(span=int(1), adjust=False).mean()
    
    #df['lsfreal_time'] = df['lsfreal_time'].ewm(span=1, adjust=False).mean()

    mTrade = sorted(AllTrades, key=lambda d: d[1], reverse=True)
    
    [mTrade[i].insert(4,i) for i in range(len(mTrade))] 
    
    newwT = []
    for i in mTrade:
        newwT.append([i[0],i[1],i[2],i[5], i[4],i[3],i[6]])
    
    
    dtime = df['time'].dropna().values.tolist()
    dtimeEpoch = df['timestamp'].dropna().values.tolist()
    
    
    #tempTrades = [i for i in AllTrades]
    #tempTrades = sorted(AllTrades, key=lambda d: d[6], reverse=False) 
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
        
        troPerCandle = []
        for tr in range(len(make)):
            if tr+1 < len(make):
                tempList = AllTrades[make[tr][2]:make[tr+1][2]]
            else:
                tempList = AllTrades[make[tr][2]:len(AllTrades)]
                
            troPerCandle.append([make[tr][1],sorted(tempList, key=lambda d: d[1], reverse=True)[:int(100)]])
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
        stored_data['troPerCandle'] = stored_data['troPerCandle'][:len(stored_data['troPerCandle'])-1] + troPerCandle
        
        bful = []
        valist = []
        for it in range(len(make)):
            if it+1 < len(make):
                tempList = AllTrades[0:make[it+1][2]]
            else:
                tempList = AllTrades
                
            hstp = historV1(df[:startIndex+it],int(tpoNum),{}, tempList, [])
            vA = valueAreaV3(hstp[0])
            valist.append(vA  + [df['timestamp'][startIndex+it], df['time'][startIndex+it], hstp[2]])
            nelist = sorted(tempList, key=lambda d: d[1], reverse=True)[:int(100)]#tpoNum
            
            timestamp_s = make[it][0] / 1_000_000_000
            new_timestamp_s = timestamp_s + (int(interv)*60)
            new_timestamp_ns = int(new_timestamp_s * 1_000_000_000)

            bful.append([make[it][1], sum([i[1] for i in nelist if i[5] == 'B']), sum([i[1] for i in nelist if i[5] == 'A']),  sum([i[1] for i in nelist if (i[2] >= make[it][0] and i[2] <= new_timestamp_ns) and i[5] == 'B']), sum([i[1] for i in nelist if (i[2] >= make[it][0] and i[2] <= new_timestamp_ns) and i[5] == 'A']) ])
            
            
  
        
        dst = [[bful[row][0], bful[row][1], 0, bful[row][2], 0, bful[row][3], bful[row][4]] for row in  range(len(bful))]
        
        stored_data['tro'] = stored_data['tro'][:len(stored_data['tro'])-1] + dst
        stored_data['pdata'] = stored_data['pdata'][:len(stored_data['pdata'])-1] + valist

        
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
            
            
        troPerCandle = []
        for tr in range(len(make)):
            if tr+1 < len(make):
                #print(make[tr][2],make[tr+1][2])
                tempList = AllTrades[make[tr][2]:make[tr+1][2]]
            else:
                tempList = AllTrades[make[tr][2]:len(AllTrades)]
            
            #secList = sorted(tempList, key=lambda d: d[1], reverse=True)[:int(tpoNum)]
            troPerCandle.append([make[tr][1],sorted(tempList, key=lambda d: d[1], reverse=True)[:int(100)]])
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
        valist =[]
        for it in range(len(make)):
            if it+1 < len(make):
                tempList = AllTrades[0:make[it+1][2]]
            else:
                tempList = AllTrades
            
            temphs = historV1(df[:it+1],int(tpoNum),{}, tempList, [])
            vA = valueAreaV3(temphs[0])
            valist.append(vA  + [df['timestamp'][it], df['time'][it], temphs[2]])
            
            nelist = sorted(tempList, key=lambda d: d[1], reverse=True)[:int(100)]#tpoNum
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
            
        stored_data = {'timeFrame': timeFrame, 'tro':dst, 'pdata':valist, 'troPerCandle':troPerCandle} 
        
    
    
    topBuys = []
    topSells = []
    
    # Iterate through troPerCandle and compute values
    for i in stored_data['troPerCandle']:
        tobuyss = sum(x[1] for x in i[1] if x[5] == 'B')  # Sum buy orders
        tosellss = sum(x[1] for x in i[1] if x[5] == 'A')  # Sum sell orders
        
        topBuys.append(tobuyss)  # Store buy values
        topSells.append(tosellss)  # Store sell values
    
    # Add to the DataFrame
    df['topBuys'] = topBuys
    df['topSells'] = topSells
    df['topDiffNega'] = ((df['topBuys'] - df['topSells']).apply(lambda x: x if x < 0 else np.nan)).abs()
    df['topDiffPost'] = (df['topBuys'] - df['topSells']).apply(lambda x: x if x > 0 else np.nan)
    
    df['percentile_topBuys'] =  [percentileofscore(df['topBuys'][:i+1], df['topBuys'][i], kind='mean') for i in range(len(df))]
    df['percentile_topSells'] =  [percentileofscore(df['topSells'][:i+1], df['topSells'][i], kind='mean') for i in range(len(df))] 
    
    df['percentile_Posdiff'] =  [percentileofscore(df['topDiffPost'][:i+1].dropna(), df['topDiffPost'][i], kind='mean') if not np.isnan(df['topDiffPost'][i]) else None for i in range(len(df))]
    df['percentile_Negdiff'] =  [percentileofscore(df['topDiffNega'][:i+1].dropna(), df['topDiffNega'][i], kind='mean') if not np.isnan(df['topDiffNega'][i]) else None for i in range(len(df))]
    
        
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
        previousDay = [csv_rows[[i[4] for i in csv_rows].index(symbolNum)][0], csv_rows[[i[4] for i in csv_rows].index(symbolNum)][1], csv_rows[[i[4] for i in csv_rows].index(symbolNum)][2]]
    except(ValueError):
        previousDay = []
    
    '''
    df['total_buys'] =  [i[2] for i in stored_data['timeFrame']]
    df['total_sells'] = [i[3] for i in stored_data['timeFrame']]
    
    # Calculate imbalance
    df['imbalance'] = (df['total_buys'] - df['total_sells']) / (df['total_buys'] + df['total_sells'])
    
    window = 5  # Rolling window size in minutes
    df['rolling_buys'] = df['total_buys'].rolling(window=window).sum()
    df['rolling_sells'] = df['total_sells'].rolling(window=window).sum()
    
    df['rolling_imbalance'] = (df['rolling_buys'] - df['rolling_sells']) / (df['rolling_buys'] + df['rolling_sells'])
    '''
    #df['dominance'] = df['total_buys'] > df['total_sells']
    #df['ema_slope'] = df[clustNum+'ema'].rolling(window).apply(lambda x: linregress(range(len(x)), x).slope)
    # Apply Savitzky-Golay filter to compute the first derivative
    #try:
        
        #df['derivative_1'] = savgol_filter(df[clustNum+'ema'], window_length=int(curvature), polyorder=poly_order, deriv=1)
        #df['derivative_2'] = savgol_filter(df[clustNum+'ema'], window_length=int(curvatured2), polyorder=poly_order, deriv=2)
    #except(ValueError):
        #pass
    try:
        df['LowVA'] = pd.Series([i[0] for i in stored_data['pdata']])
        df['HighVA'] = pd.Series([i[1] for i in stored_data['pdata']])
        df['POC'] = pd.Series([i[2] for i in stored_data['pdata']])
        df['POC2'] = pd.Series([i[5] for i in stored_data['pdata']])
        df['weights'] = [i[2]-i[3] for i in stored_data['timeFrame']]
        '''
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
        if len(df) >= len(POC) and len(POC) > 0:
            df['LowVA'] = pd.Series(LowVA + [LowVA[len(LowVA)-1]]*(len(df)-len(LowVA)))
            df['HighVA'] = pd.Series(HighVA + [HighVA[len(HighVA)-1]]*(len(df)-len(HighVA)))
            df['POC']  = pd.Series(POC + [POC[len(POC)-1]]*(len(df)-len(POC)))
            '''
            #df['POC2']  = pd.Series(pocc + [pocc[len(pocc)-1]]*(len(df)-len(pocc)))
            #df['POCDistance'] = (abs(df['1ema'] - df['POC']) / ((df['1ema']+ df['POC']) / 2)) * 100
        # df['smoothed_derivative']#df['derivative_1']#((df['derivative_1'] - df['POC']) / ((df['derivative_1'] + df['POC']) / 2)) * 100
        
        #buffer = 0.002  # 0.2% buffer

        # Define the buffer zone
        #df['upper_buffer'] = df['POC'] * (1 + buffer)
        #df['lower_buffer'] = df['POC'] * (1 - buffer)
        
        #df['positive_mean'] = df['smoothed_derivative'].expanding().apply(lambda x: x[x > 0].mean(), raw=False)
        #df['negative_mean'] = df['smoothed_derivative'].expanding().apply(lambda x: x[x < 0].mean(), raw=False)
        
        df['smoothed_1ema'] = double_exponential_smoothing(df['1ema'], 0.5, 0.01)#apply_kalman_filter(df['1ema'], transition_covariance=float(curvature), observation_covariance=float(curvatured2))#random_walk_filter(df['1ema'], alpha=alpha)
        df['POCDistance'] = (df['smoothed_1ema'] - df['POC']) / df['POC'] * 100
        df['POCDistanceEMA'] = df['POCDistance']#((df['1ema'] - df['POC']) / ((df['1ema'] + df['POC']) / 2)) * 100
        df['vwapDistance'] = (df['smoothed_1ema'] - df['vwap']) / df['vwap'] * 100
        df['uppervwapDistance'] = (df['smoothed_1ema'] - df['uppervwapAvg']) / df['uppervwapAvg'] * 100
        df['lowervwapDistance'] = (df['smoothed_1ema'] - df['lowervwapAvg']) / df['lowervwapAvg'] * 100
        df['vwapAvgDistance'] = (df['smoothed_1ema'] - df['vwapAvg']) / df['vwapAvg'] * 100
        #df['POCDistanceEMA'] = df['POCDistanceEMA'].ewm(span=2, adjust=False).mean()#gaussian_filter1d(df['POCDistanceEMA'], sigma=int(1))##
        #df['POCDistanceEMA'] = exponential_median(df['POCDistanceEMA'].values, span=2)
        
        #df['positive_meanEma'] = df['POCDistanceEMA'].expanding().apply(lambda x: x[x > 0].mean(), raw=False)
        #df['negative_meanEma'] = df['POCDistanceEMA'].expanding().apply(lambda x: x[x < 0].mean(), raw=False)
        
        #df['positive_medianEma'] = df['POCDistanceEMA'].expanding().apply(lambda x: np.median(x[x > 0]), raw=False)
        #df['negative_medianEma'] = df['POCDistanceEMA'].expanding().apply(lambda x: np.median(x[x < 0]), raw=False)
        
        #positive_values = df['POCDistanceEMA'].apply(lambda x: x if x > 0 else None)
        #negative_values = df['POCDistanceEMA'].apply(lambda x: x if x < 0 else None)
        
        # Calculate EMA separately for positive and negative values
        #df['positive_emaEmaRoll'] = positive_values.ewm(span=30, adjust=False).mean()
        #df['negative_emaEmaRoll'] = negative_values.ewm(span=30, adjust=False).mean()
        '''
        df['rolling_std'] = df['close'].rolling(window=150, min_periods=1).std()
        df['rollingPositive_std'] = positive_values.std()
        df['rollingNegative_std'] = negative_values.std()


        df['positive_dynamicEma'] = df['positive_meanEma'] + df['rolling_std']
        df['negative_dynamicEma'] = df['negative_meanEma'] - df['rolling_std']
        
        df['positive_emaEmaRoll_median'] = ewm_median(positive_values, span=20)
        df['negative_emaEmaRoll_median'] = ewm_median(negative_values, span=20)
        
        
        df['total_buys'] =  [i[2] for i in stored_data['timeFrame']]
        df['total_sells'] = [i[3] for i in stored_data['timeFrame']]
        
        # Calculate imbalance
        df['imbalance'] = (df['total_buys'] - df['total_sells']) / (df['total_buys'] + df['total_sells'])
        
        df['rolling_buys'] = df['total_buys'].rolling(window=20).sum()
        df['rolling_sells'] = df['total_sells'].rolling(window=20).sum()
        
        df['rolling_imbalance'] = (df['rolling_buys'] - df['rolling_sells']) / (df['rolling_buys'] + df['rolling_sells'])
        

        rolling_window = 30
        df['positive_percentile'] = df['POCDistanceEMA'].rolling(window=rolling_window, min_periods=1).apply(
            lambda x: np.percentile(x[x > 0], 75) if len(x[x > 0]) > 0 else np.nan)
        df['negative_percentile'] = df['POCDistanceEMA'].rolling(window=rolling_window, min_periods=1).apply(
            lambda x: np.percentile(x[x < 0], 25) if len(x[x < 0]) > 0 else np.nan)
        
        positive_values = df['POCDistanceEMA'][df['POCDistanceEMA'] > 0]
        negative_values = df['POCDistanceEMA'][df['POCDistanceEMA'] < 0]
        
        # Calculate the 75th percentile for positive values
        positive_percentile = np.percentile(positive_values, 15) if len(positive_values) > 0 else np.nan
        
        # Calculate the 25th percentile for negative values
        negative_percentile = np.percentile(negative_values, 15) if len(negative_values) > 0 else np.nan
        
        # Assign the computed percentiles to the dataframe
        df['positive_percentile'] = positive_percentile
        df['negative_percentile'] = negative_percentile
        '''
        #df['smoothed_1ema'] = linear_regression_smoothing(df['1ema'],window_size=8)#mean_shift_filter(df['close'], bandwidth=3, max_iterations=5)
        #alpha = 0.6  # Smoothing factor
        #df['momentum'] = df['smoothed_1ema'].diff() 
        #df['RSI'] = calculate_rsi(df['close'])
        #df['divergence'] = (df['RSI'].diff() * df['close'].diff()) < 0
        
        

        df['slope_degrees'] = [calculate_slope_rolling(i, df['smoothed_1ema'].values, int(15)) for i in range(len(df))]
        df['polyfit_slope'] = [calculate_polyfit_slope_rolling(i, df['smoothed_1ema'].values, int(15)) for i in range(len(df))]
        #df['hybrid'] = [calculate_hybrid_slope(i, df['smoothed_1ema'].values, int(30)) for i in range(len(df))]
        
        slope = str(df['slope_degrees'].iloc[-1]) + ' ' + str(df['polyfit_slope'].iloc[-1])
        
        #df['atr'] = compute_atr(df) #period=int(clustNum)
        #df['positive_threshold'] = df['POC'] + 1.2 * df['atr']
        #df['negative_threshold'] = df['POC'] - 1.2 * df['atr']
        
        
        #df['atr_multiplier'] = 1.3 + (df['atr'] / df['atr'].mean()) * 0.5
        #df['positive_threshold'] = df['POC'] + df['atr_multiplier'] * df['atr']
        #df['negative_threshold'] = df['POC'] - df['atr_multiplier'] * df['atr']
        #& (df['POCDistanceEMA'] > df['positive_meanEma']) & (df['smoothed_derivative'] > 0)
        #(df['POCDistanceEMA'] < df['negative_meanEma']) & (df['smoothed_derivative'] < 0)  &
        
                
        df['vwap_signalBuy'] = df.apply(vwapDistanceCheckBuy, axis=1)
        df['vwap_signalSell'] = df.apply(vwapDistanceCheckSell, axis=1)
        
        df['uppervwap_signalBuy'] = df.apply(uppervwapDistanceCheckBuy, axis=1)
        df['uppervwap_signalSell'] = df.apply(uppervwapDistanceCheckSell, axis=1) 
        
        df['lowervwap_signalBuy'] = df.apply(lowervwapDistanceCheckBuy, axis=1)
        df['lowervwap_signalSell'] = df.apply(lowervwapDistanceCheckSell, axis=1) 
        
        df['vwapAvg_signalBuy'] = df.apply(vwapAvgDistanceCheckBuy, axis=1)
        df['vwapAvg_signalSell'] = df.apply(vwapAvgDistanceCheckSell, axis=1)  
        
        #df['buy_signal'] = (df['POCDistanceEMA'].abs() <= 0.021) & (df['smoothed_derivative'] > 0) & ((df['polyfit_slope'] > 0) | (df['slope_degrees'] > 0))#(df['smoothed_1ema'] >= df['POC']) & (df['POCDistanceEMA'] > 0.048) & (df['smoothed_derivative'] > 0)& ((df['polyfit_slope'] > 0) | (df['slope_degrees'] > 0)) & (df['vwap_signalBuy'])#0.03 0.0183& (df['smoothed_derivative'] > 0) & (df['POCDistanceEMA'] > 0.01)#(df['momentum'] > 0) #& (df['1ema'] >= df['vwap']) #& (df['2ema'] >= df['POC'])#(df['derivative_1'] > 0) (df['lsf'] >= df['POC']) #(df['1ema'] > df['POC2']) &  #& (df['holt_winters'] >= df['POC2'])# &  (df['derivative_1'] >= df['kalman_velocity'])# &  (df['derivative_1'] >= df['derivative_2']) )# & (df['1ema'].shift(1) >= df['POC2'].shift(1)) # &  (df['MACD'] > df['Signal'])#(df['1ema'].shift(1) < df['POC2'].shift(1)) & 
        
        # Identify where cross below occurs (previous 3ema is above POC, current 3ema is below)
        #df['sell_signal'] = (df['POCDistanceEMA'].abs() <= 0.021) & (df['smoothed_derivative'] < 0) & ((df['polyfit_slope'] < 0) | (df['slope_degrees'] < 0))#(df['smoothed_1ema'] <= df['POC'])  & (df['POCDistanceEMA'] < -0.048) & (df['smoothed_derivative'] < 0)&  ((df['polyfit_slope'] < 0) | (df['slope_degrees'] < 0)) & (df['vwap_signalSell']) #-0.03 -0.0183& (df['smoothed_derivative'] < 0) & (df['POCDistanceEMA'] < -0.01)#&  (df['momentum'] < 0)  #& (df['1ema'] <= df['vwap']) #& (df['2ema'] <= df['POC'])#(df['derivative_1'] < 0) (df['lsf'] <= df['POC']) #(df['1ema'] < df['POC2']) &    #& (df['holt_winters'] <= df['POC2'])# & (df['derivative_1'] <= 0) & (df['derivative_1'] <= df['kalman_velocity'])# )# & (df['1ema'].shift(1) <= df['POC2'].shift(1)) # & (df['Signal']  > df['MACD']) #(df['1ema'].shift(1) > df['POC2'].shift(1)) &

        #df['buy_signal'] = (df['cross_above']) #& (df['smoothed_1ema'] >= df['positive_threshold'])# & (df['smoothed_derivative'] > df['positive_mean']) & (df['POCDistanceEMA'] > df['positive_meanEma'])# & (df['POCDistanceEMA'] > df['positive_percentile'])# & (df['rolling_imbalance'] > 0)#& (df['rolling_imbalance'] > 0) #&   (df['rolling_imbalance'] >=  rollingThres)# & (df['POCDistance'] <= thresholdTwo))
        #df['sell_signal'] = (df['cross_below']) #& (df['smoothed_1ema'] <= df['negative_threshold'])# & (df['smoothed_derivative'] < df['negative_mean']) & (df['POCDistanceEMA'] < df['positive_meanEma'])# & (df['POCDistanceEMA'] < df['negative_percentile'])# & (df['rolling_imbalance'] < 0)#& (df['rolling_imbalance'] < 0) #& (df['rolling_imbalance'] <= -rollingThres)# & (df['POCDistance'] >= -thresholdTwo))
        
    except(NotFound):
        pass
    '''
    df['stillbuy'] = False
    df['stillsell'] = False
    
    # Initialize tracking variables
    stillbuy = False
    stillsell = False
    
    # Iterate through the DataFrame rows
    for p in range(len(df)):
        # Check if buy_signal is triggered
        if df['buy_signal'][p]:
            stillbuy = True
            stillsell = False  # Reset stillsell when a buy is triggered
    
        # Check if sell_signal is triggered
        if df['sell_signal'][p]:
            stillsell = True
            stillbuy = False  # Reset stillbuy when a sell is triggered
    
        # Update the tracking columns
        df.at[p, 'stillbuy'] = stillbuy
        df.at[p, 'stillsell'] = stillsell

    # Define conditions for ending a buy or sell
    df['endBuy'] = (df['stillbuy']) & (df['smoothed_1ema'] <= df['POC']) & (df['POCDistanceEMA'] < -0.048) & (df['smoothed_derivative'] < 0) & ((df['polyfit_slope'] < 0) | (df['slope_degrees'] < 0)) & (df['vwap_signalSell'])
    
    df['endSell'] = (df['sell_signal']) & (df['smoothed_1ema'] >= df['POC']) & (df['POCDistanceEMA'] > 0.048) & (df['smoothed_derivative'] > 0) & ((df['polyfit_slope'] > 0) | (df['slope_degrees'] > 0)) & (df['vwap_signalBuy'])
    
    # Initialize new tracking signals for forced cross changes
    #df['cross_above'] = False
    #df['cross_below'] = False
    
    # Handle transitions based on endBuy and endSell
    for p in range(len(df)):
        if df['endBuy'][p]:  # When endBuy is triggered
            df.at[p, 'sell_signal'] = True  # Start a sell signal
            #df.at[p, 'cross_below'] = True  # Manually trigger a new cross_below
            #df.at[p, 'stillbuy'] = False  # End stillbuy
            #df.at[p, 'stillsell'] = True  # Start stillsell
    
        if df['endSell'][p]:  # When endSell is triggered
            df.at[p, 'buy_signal'] = True  # Start a buy signal
            #df.at[p, 'cross_above'] = True  # Manually trigger a new cross_above
            #df.at[p, 'stillsell'] = False  # End stillsell
            #df.at[p, 'stillbuy'] = True  # Start stillbuy
        #try:
        #mboString = '('+str(round(df['positive_mean'].iloc[-1], 3)) + ' | ' + str(round(df['negative_mean'].iloc[-1], 3))+') --' + ' ('+str(round(df['positive_meanEma'].iloc[-1], 3)) + ' | ' + str(round(df['negative_meanEma'].iloc[-1], 3))+') '+slope#str(round((abs(df['HighVA'][len(df)-1] - df['LowVA'][len(df)-1]) / ((df['HighVA'][len(df)-1] + df['LowVA'][len(df)-1]) / 2)) * 100,3))
    #except(KeyError):
    '''
    df['stillbuy'] = False
    df['stillsell'] = False
    df['buy_signal'] = False
    df['sell_signal'] = False
    
    # Initialize tracking variables
    stillbuy = False
    stillsell = False
    
    for p in range(len(df)):
        # Initial trade entry conditions (fixed for better execution) not stillsell and
        if not stillsell and ((abs(df.at[p, 'POCDistanceEMA']) <= 0.021)  & (df.at[p, 'smoothed_derivative'] > 0) & ((df.at[p, 'polyfit_slope'] > 0) | (df.at[p, 'slope_degrees'] > 0))):
            df.at[p, 'buy_signal'] = True
            stillbuy = True
            stillsell = False  
    #not stillbuy and
        if not stillbuy and ((abs(df.at[p, 'POCDistanceEMA']) <= 0.021) & (df.at[p, 'smoothed_derivative'] < 0) & ((df.at[p, 'polyfit_slope'] < 0) | (df.at[p, 'slope_degrees'] < 0))):
            df.at[p, 'sell_signal'] = True
            stillsell = True
            stillbuy = False  
    
        # Exit condition for stillbuy → Trigger a sell
        if (
            stillbuy and 
            (df.at[p, 'smoothed_1ema'] <= df.at[p, 'POC']) and 
            (df.at[p, 'POCDistanceEMA'] < -0.048) and 
            (df.at[p, 'smoothed_derivative'] < 0) and 
            ((df.at[p, 'polyfit_slope'] < 0) | (df.at[p, 'slope_degrees'] < 0)) and 
            (df.at[p, 'vwap_signalSell']) and
            (df.at[p, 'uppervwap_signalSell']) and
            (df.at[p, 'lowervwap_signalSell']) and
            (df.at[p, 'vwapAvg_signalSell']) 
        ):
            df.at[p, 'sell_signal'] = True  # Trigger sell
            stillbuy = False  # Stop buy tracking
            stillsell = True  # Start sell tracking
    
        # Exit condition for stillsell → Trigger a buy
        if (
            stillsell and 
            (df.at[p, 'smoothed_1ema'] >= df.at[p, 'POC']) and 
            (df.at[p, 'POCDistanceEMA'] > 0.048) and 
            (df.at[p, 'smoothed_derivative'] > 0) and 
            ((df.at[p, 'polyfit_slope'] > 0) | (df.at[p, 'slope_degrees'] > 0)) and 
            (df.at[p, 'vwap_signalBuy']) and
            (df.at[p, 'uppervwap_signalBuy']) and
            (df.at[p, 'lowervwap_signalBuy']) and
            (df.at[p, 'vwapAvg_signalBuy'])
        ):
            df.at[p, 'buy_signal'] = True  # Trigger buy
            stillsell = False  # Stop sell tracking
            stillbuy = True  # Start buy tracking
            
            
        if (
            not stillsell and not stillbuy and 
            (df.at[p, 'smoothed_1ema'] >= df.at[p, 'POC']) and 
            (df.at[p, 'POCDistanceEMA'] > 0.048) and 
            (df.at[p, 'smoothed_derivative'] > 0) and 
            ((df.at[p, 'polyfit_slope'] > 0) | (df.at[p, 'slope_degrees'] > 0)) and 
            (df.at[p, 'vwap_signalBuy']) and
            (df.at[p, 'uppervwap_signalBuy']) and
            (df.at[p, 'lowervwap_signalBuy'])and
            (df.at[p, 'vwapAvg_signalBuy']) 
        ):
            df.at[p, 'buy_signal'] = True  # Trigger buy
            stillsell = False  # Stop sell tracking
            stillbuy = True  # Start buy tracking
            
        if (
            not stillsell and not stillbuy and 
            (df.at[p, 'smoothed_1ema'] <= df.at[p, 'POC']) and 
            (df.at[p, 'POCDistanceEMA'] < -0.048) and 
            (df.at[p, 'smoothed_derivative'] < 0) and 
            ((df.at[p, 'polyfit_slope'] < 0) | (df.at[p, 'slope_degrees'] < 0)) and 
            (df.at[p, 'vwap_signalSell']) and
            (df.at[p, 'uppervwap_signalSell'])and
            (df.at[p, 'lowervwap_signalSell']) and
            (df.at[p, 'vwapAvg_signalSell']) 
        ):
            df.at[p, 'sell_signal'] = True  # Trigger sell
            stillbuy = False  # Stop buy tracking
            stillsell = True  # Start sell tracking
            
    
        # Update tracking columns
        df.at[p, 'stillbuy'] = stillbuy
        df.at[p, 'stillsell'] = stillsell
    


    #calculate_ttm_squeeze(df)
    
    
        
    if interval_time == initial_inter:
        interval_time = subsequent_inter
    
    if sname != previous_stkName or interv != previous_interv:
        interval_time = initial_inter
        
    
    
    fg = plotChart(df, [hs[1],newwT[:int(100)]], va[0], va[1], x_fake, df_dx, troPerCandle=stored_data['troPerCandle'] , stockName=symbolNameList[symbolNumList.index(symbolNum)], previousDay=previousDay, pea=False,  OptionTimeFrame = stored_data['timeFrame'], clusterNum=int(clustNum), troInterval=stored_data['tro']) #trends=FindTrends(df,n=10)
 
    return stored_data, fg, previous_stkName, previous_interv, interval_time

#[(i[2]-i[3],i[0]) for i in timeFrame ]valist.append(vA  + [df['timestamp'][it], df['time'][it], temphs[2]])
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8080)
    #app.run_server(debug=False, use_reloader=False)
