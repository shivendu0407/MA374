from matplotlib import pyplot as plt
import numpy as np
import math
import pandas as pd
from datetime import datetime
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



def newtonRaphson(S,K,r,tau,sig,p):
    if(tau==0):
        return -1
    prev = -1
    nmax = 0
    while(nmax<25):
        if(sig==0):
            return -1
        nmax += 1
        prev = sig
        d1 = (1/(sig*math.sqrt(tau)))*(math.log(S/K) + (r + 0.5*sig*sig)*tau)
        d2 = d1 - sig*math.sqrt(tau)
        sig = sig - ((S*norm.cdf(d1)-K*math.exp(-r*tau)*norm.cdf(d2)-p)/(S*norm.pdf(d1)*math.sqrt(tau)))
        if(np.isnan(d1) or np.isnan(d2) or np.isnan(sig) or sig<1e-2):
            return -1
        if(abs(sig-prev)<1e-4):
            return sig
        if(abs(sig-prev)>2):
            return -1
    return -1


def getHistVolatility(df,timePeriod,startDate,stock):
    arr = []
    for i in range(len(df)):
        if((startDate - datetime.strptime(df["Date"][i],"%d-%b-%y")).days<=timePeriod):
            arr.append(df[stock][i])
    X = []
    for i in range(len(arr)-1):
        X.append(math.log(arr[i+1]/arr[i]))
    if(len(X)==0):
        return -1
    return math.sqrt(252)*np.nanstd(X)
    
        
        

companies = ["NSE","AsianPaint","ICICIBANK","TECHM","WIPRO","UPL"]
stocks = ["^NSEI","ASIANPAINT.NS","ICICIBANK.NS","TECHM.NS","WIPRO.NS","UPL.NS"]
for j in range(6):
    df = pd.DataFrame()
    if(j==0):
        df = pd.read_csv("NIFTYoptiondata.csv")
    else:
        df = pd.read_excel("stockoptiondata.xlsx",sheet_name=companies[j])
    idx = np.random.randint(0,len(df),10000)
    
    callPrices = []
    putPrices = []
    days = []
    strikes = []
    
    for i in idx:
        callPrices.append(df.loc[i,"Call Price"])
        putPrices.append(df.loc[i,"Put Price"])
        days.append((datetime.strptime(df.loc[i,"Maturity"],"%d-%b-%y")-datetime.strptime(df.loc[i,"Date"],"%d-%b-%y")).days)
        strikes.append(df.loc[i,"Strike Price"])
        
    ax = plt.axes(projection="3d")
    ax.scatter3D(days,strikes,callPrices)
    plt.show()

    ax = plt.axes(projection="3d")
    ax.scatter3D(days,strikes,putPrices)
    plt.show()
    
    df1 = pd.read_csv("nsedata1.csv").loc[:,["Date",stocks[j]]]
    df = pd.merge(df,df1,on="Date")
    
    idx = np.random.randint(0,len(df),5000)
    impliedVols = []
    impliedVol1 = []
    histVols = []
    strikes = []
    days = []
    
    for i in idx:
        price = df.loc[i,"Call Price"]
        S = df.loc[i,stocks[j]]
        K = df.loc[i,"Strike Price"]
        tau = (datetime.strptime(df.loc[i,"Maturity"],"%d-%b-%y")-datetime.strptime(df.loc[i,"Date"],"%d-%b-%y")).days
        impVol = newtonRaphson(S,K,0.05,tau/252,0.3,price)
        if(impVol!=-1):
            impliedVols.append(impVol)
            strikes.append(K)
            days.append(tau)
            if(np.random.uniform()<0.3):
                histVol = getHistVolatility(df1,tau,datetime.strptime(df["Date"][i],"%d-%b-%y"),stocks[j])
                histVols.append(histVol)
                impliedVol1.append(impVol)
    print(len(impliedVols))    
    ax = plt.axes(projection="3d")
    ax.scatter3D(days,strikes,impliedVols)
    plt.show()
    
    plt.scatter(impliedVol1,histVols)
    plt.axis("square")
    plt.show()