from matplotlib import pyplot as plt
import numpy as np
import math
import pandas as pd
from scipy.stats import norm

df = pd.DataFrame()
stocks = []

def plotData(interval):
    df.fillna(method='ffill',inplace=True)
    df1 = df
    if(interval=='W' or interval=='M'):
        df1 = df1.groupby(pd.DatetimeIndex(df1.Date).to_period(interval)).nth(0)
    print(df1)
    for i in range(5):
        fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
        axes = [ax1,ax2,ax3,ax4]
        for j in range(4):
            axes[j].plot(df1.loc[:,stocks[4*i+j+1]])
            axes[j].set_xticks(np.arange(0,len(df1),len(df1)/4),labels=df1.loc[np.arange(0,len(df1),len(df1)/4),"Date"])
        plt.show()
    if(interval=='W' or interval=='M'):
        df1 = df1.set_index('Date')
    df1 = 252*(df1.iloc[:,1:].pct_change())
    df1.fillna(method='ffill',inplace=True)
    df1.fillna(method='bfill',inplace=True)
    m = np.mean(df1,axis=0)
    v = np.cov(np.array(df1))
    df1 = df1.iloc[1:,:]
    for i in range(5):
        fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
        axes = [ax1,ax2,ax3,ax4]
        for j in range(4):
            mu = m[4*i+j+1]
            vr = math.sqrt(v[4*i+j+1][4*i+j+1])
            axes[j].hist([(x-mu)/vr for x in df1[stocks[4*i+j+1]]],color='y',density = True,bins=50)
            axes[j].plot(np.linspace(-3,3,5000),[norm.pdf(x) for x in np.linspace(-3,3,5000)],color='r')
        plt.show()
        



def solve(filename):
    global df
    df = pd.read_csv(filename)
    global stocks
    stocks = df.columns[1:]
    for interval in ['W','M']:
        plotData(interval)
    
solve("bsedata1.csv")