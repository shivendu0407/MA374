import numpy as np
from matplotlib import pyplot as plt
import math
import pandas as pd

df = pd.read_csv("stocks.csv")


companies = ["AAPL","AMZN","META","GOOG","IBM","INTC","MSFT","NFLX","NKE","TSLA"]
dat = np.transpose(np.array(12*df.loc[:,companies].pct_change().loc[1:,]))
C = np.cov(dat)
m = (np.mean(dat,axis=1))
Cinv = np.linalg.inv(C)
u = [1,1,1,1,1,1,1,1,1,1]
ut = np.transpose(u)
uCinv = u@Cinv
mCinv = np.matmul(m,Cinv)
mt = np.transpose(m)
otherRisk = []
markowitzRisk = []
muRf = 0.05

wmin =(u@Cinv)/(uCinv@ut)
stdmin = (wmin@C@np.transpose(wmin))**0.5
retmin = mt@wmin


otherRet = np.linspace(0,retmin,10000)
otherWeights = []
markowitzRet = np.linspace(retmin,1,10000)
markowitzWeights = []

def getWeights(mu):
    w = ((np.linalg.det([[1,uCinv@mt],[mu,mCinv@mt]])*(uCinv)) + (np.linalg.det([[uCinv@ut,1],[mCinv@ut,mu]])*(mCinv)))/(np.linalg.det([[uCinv@ut,uCinv@mt],[mCinv@ut,mCinv@mt]]))
    return w
def getRisk(w):
    return (w@C@np.transpose(w))**0.5

def markowitzBulletLine():
    global otherRisk
    global markowitzRisk
    
    for x in otherRet:
        otherWeights.append(getWeights(x))
    otherRisk = [getRisk(x) for x in otherWeights]
    
    for x in markowitzRet:
        markowitzWeights.append(getWeights(x))
    markowitzRisk = [getRisk(x) for x in markowitzWeights]
    
    plt.style.use("seaborn")
    plt.plot(otherRisk,otherRet)
    plt.plot(markowitzRisk,markowitzRet)
    plt.title("Markowitz efficient frontier")
    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.legend(["Other part","Markowitz frontier"])
    plt.show()
    
def computeMarketReturn(x,muM,sigM):
    return (muRf + (((muM-muRf)/(sigM))*x))
    
def capitalSecurityMarketLine():
    sig = np.linspace(0,3,10000)
    plt.style.use("seaborn")
    gamma = (np.subtract(m,np.dot(0.05,u)) @ Cinv @ np.transpose(u) )
    wM = (np.subtract(m,np.dot(0.05,u))@Cinv)/gamma
    muM = mt@wM
    sigM = (wM@C@np.transpose(wM))**0.5
    print("Market Portfolio :\n","Weights = ",wM,"\nreturn = ",muM,"\nrisk = ",sigM)
    plt.plot(sig,[computeMarketReturn(x,muM,sigM) for x in sig ])
    plt.plot(otherRisk,otherRet)
    plt.plot(markowitzRisk,markowitzRet)
    plt.title("Capital Market Line")
    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.legend(["Capital Market Line","Other part","Markowitz frontier"])
    plt.show() 
    beta = np.linspace(-1,1,5000)
    plt.plot(beta,[(muRf + (muM-muRf)*x) for x in beta])
    plt.xlabel("beta")
    plt.ylabel("return")
    plt.title("Security Market Line")
    plt.show()
    
markowitzBulletLine()
capitalSecurityMarketLine()