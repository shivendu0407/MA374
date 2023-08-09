import numpy as np
from matplotlib import pyplot as plt
import math
import pandas as pd

def solve(filename,companies):
    df = pd.read_csv(filename)
    df = (df.loc[:,companies].pct_change()).loc[1:,:]
    df.fillna(0,inplace = True)
    datX = (df.shape[0]/5)*np.transpose(np.array(df))   
    dat = datX[1:]
    C = np.cov(dat)
    m = (np.mean(dat,axis=1))
    Cinv = np.linalg.inv(C)
    u = [1 for i in range(20)]
    ut = np.transpose(u)
    uCinv = u@Cinv
    mCinv = np.matmul(m,Cinv)
    mt = np.transpose(m)
    muRf = 0.05

    wmin =(u@Cinv)/(uCinv@ut)
    stdmin = (wmin@C@np.transpose(wmin))**0.5
    retmin = mt@wmin


    otherRet = np.linspace(-1,retmin,10000)
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
        plt.title("Markowitz efficient frontier for {a}".format(a=filename))
        plt.xlabel("Risk")
        plt.ylabel("Return")
        plt.legend(["Other part","Markowitz frontier"])
        plt.show()
        
    def computeMarketReturn(x,muM,sigM):
        return (muRf + (((muM-muRf)/(sigM))*x))
        
    def capitalSecurityMarketLine():
        sig = np.linspace(0,0.25,10000)
        plt.style.use("seaborn")
        gamma = (np.subtract(m,np.dot(0.05,u)) @ Cinv @ np.transpose(u) )
        wM = (np.subtract(m,np.dot(0.05,u))@Cinv)/gamma
        muM = np.mean(datX[0])
        sigM = (np.var(datX[0]))**0.5
        print("Market Portfolio :\n","return = ",muM,"\nrisk = ",sigM)
        plt.plot(sig,[computeMarketReturn(x,muM,sigM) for x in sig ])
        plt.title("Capital Market Line for {a}".format(a = filename))
        plt.xlabel("Risk")
        plt.ylabel("Return")
        plt.legend(["Capital Market Line","Other part","Markowitz frontier"])
        plt.show() 
        betas = []
        for i in range(20):
            t = (np.cov(datX[0],datX[i+1]))[1][0]/(sigM**2)
            print("\n Calculated beta for",companies[i+1]," = ",t)
            betas.append(t)
        beta = np.linspace(-1,2,5000)
        plt.plot(beta,[(muRf + (muM-muRf)*x) for x in beta])
        plt.scatter(betas[:10],m[:10],color = "green")
        plt.scatter(betas[10:],m[10:],color = "pink")
        plt.xlabel("beta")
        plt.ylabel("return")
        plt.title("Security Market Line for {a}".format(a=filename))
        plt.legend(["Security Market Line","Index Stocks","Non Index Stocks"])
        plt.show()
        print("Index Return : ",muM)
        print("Index Risk : ",sigM)
        print("Equation of CML : y = {m}*x + {c}".format(m = (muM-muRf)/sigM,c = muRf))
        print("Equation of SML : y = {m}*x + {c}".format(m = (muM-muRf),c = muRf))
        
        
    markowitzBulletLine()
    capitalSecurityMarketLine()

print("For Bse:\n")
solve("bsedata1.csv",["^BSESN","INFY.BO","RELIANCE.BO","HDFCBANK.BO","AXISBANK.BO","BHARTIARTL.BO","ICICIBANK.BO","VEDL.BO","CIPLA.BO","BAJAJ-AUTO.BO","NESTLEIND.BO","TITAN.BO","IDBI.BO","TVSMOTOR.BO","BOSCHLTD.BO","GAIL.BO","YESBANK.BO","VOLTAS.BO","MARICO.BO","UNIONBANK.BO","BERGEPAINT.BO"])
print("\n\nFor Nse:\n")
solve("nsedata1.csv",["^NSEI","UPL.NS","ICICIBANK.NS","WIPRO.NS","TECHM.NS","ASIANPAINT.NS","EICHERMOT.NS","NTPC.NS","LT.NS","COALINDIA.NS","GRASIM.NS","BHEL.NS","DABUR.NS","COLPAL.NS","MRF.NS","GAIL.NS","HAVELLS.NS","POONAWALLA.NS","BOSCHLTD.NS","UNIONBANK.NS","GODREJCP.NS"])