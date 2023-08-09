import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.interpolate import interp1d

C = [[0.005,-0.010,0.004],[-0.010,0.040,-0.002],[0.004,-0.002,0.023]]
Cinv = np.linalg.inv(C)
m  = [0.1,0.2,0.15]
u = [1,1,1]
ut = np.transpose(u)
uCinv = u@Cinv
mCinv = np.matmul(m,Cinv)
mt = np.transpose(m)
otherRisk = []
markowitzRisk = []
muRf = 0.10




wmin =(u@Cinv)/(uCinv@ut)
stdmin = (wmin@C@np.transpose(wmin))**0.5
retmin = mt@wmin

otherRet = np.linspace(-0.1,retmin,5000)
otherWeights = []
markowitzRet = np.linspace(retmin,0.3,10000)
markowitzWeights = []

print("Minimum risk = ",stdmin)
print("Return = ",mt@wmin)





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
    
    myFile = open("markowitz.csv","w")
    myFile.write("w1,w2,w3,return,risk\n")
    
    for i in np.random.randint(0,10000,10):
        print("Weights = ",markowitzWeights[i],"Return = ",markowitzRet[i],"Risk = ",markowitzRisk[i])
        myFile.write("{a},{b},{c},{d},{e}\n".format(a = markowitzWeights[i][0],b = markowitzWeights[i][1],c = markowitzWeights[i][2],d = markowitzRet[i],e=markowitzRisk[i]))
    myFile.close()    
    t1 = interp1d(otherRisk,otherRet)(0.15)
    t2 = interp1d(markowitzRisk,markowitzRet)(0.15)
    print("\n\nMinimum Return for risk = 15%: ",t1*100,"%")
    print("Weights for minimum return = ",getWeights(t1))
    print("\nMaximum Return for risk = 15%: ",t2*100,"%")
    print("Weights for maximum return = ",getWeights(t2))
        
    weights18 = getWeights(0.18)
    print("\n\nFor return = 18%:\nWeights = ",weights18,"\nRisk = ",getRisk(weights18)*100,"%")
        

def computeMarketReturn(x,muM,sigM):
    return (muRf + (((muM-muRf)/(sigM))*x))
    
 
def capitalMarketLine():
    sig = np.linspace(0,0.5,10000)
    plt.style.use("seaborn")
    gamma = (np.subtract(m,np.dot(0.1,u)) @ Cinv @ np.transpose(u) )
    wM = (np.subtract(m,np.dot(0.1,u))@Cinv)/gamma
    muM = mt@wM
    sigM = (wM@C@np.transpose(wM))**0.5
    print("\nReturn for Market Portfolio = ",muM,"\nRisk for market portfolio = ",sigM)
    plt.plot(sig,[computeMarketReturn(x,muM,sigM) for x in sig ])
    plt.plot(otherRisk,otherRet)
    plt.plot(markowitzRisk,markowitzRet)
    plt.title("Capital Market Line")
    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.legend(["Capital Market Line","Other part","Markowitz frontier"])
    plt.show()

    for risk in [0.10,0.25]:
        print("\nFor risk = ",100*risk,"%")
        mu = computeMarketReturn(risk,muM,sigM)
        wtRiskFree = (mu-muM)/(muRf - muM)
        wtRisky  = (1-wtRiskFree)*wM
        print("Return = ",mu)
        print("Risk Free Weight = ",wtRiskFree)
        print("Risky Weights = ",wtRisky)

    
    
markowitzBulletLine()
capitalMarketLine()