import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.interpolate import interp1d



def get_eqn(x, y):
  slope, intercept = [], []
  for i in range(len(x) - 1):
    x1, x2 = x[i], x[i + 1]
    y1, y2 = y[i], y[i + 1]
    slope.append((y2 - y1)/(x2 - x1))
    intercept.append(y1 - slope[-1]*x1)

  return sum(slope)/len(slope), sum(intercept)/len(intercept)

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

otherRet = np.linspace(0,retmin,5000)
otherWeights = []
markowitzRet = np.linspace(retmin,0.5,10000)
markowitzWeights = []

def getWeights(mu):
    w = ((np.linalg.det([[1,uCinv@mt],[mu,mCinv@mt]])*(uCinv)) + (np.linalg.det([[uCinv@ut,1],[mCinv@ut,mu]])*(mCinv)))/(np.linalg.det([[uCinv@ut,uCinv@mt],[mCinv@ut,mCinv@mt]]))
    return w
def getRisk(w):
    return (w@C@np.transpose(w))**0.5

def markowitzBulletLine():
    global otherRisk
    global markowitzRisk

    Ret2 = []
    
    
    for x in markowitzRet:
        w = getWeights(x)
        if(all(w>=0)):
            markowitzWeights.append(w)
            Ret2.append(x)
    markowitzRisk = [getRisk(x) for x in markowitzWeights]

    
    plt.style.use("seaborn")
    plt.plot(markowitzRisk,Ret2,color = "maroon")
    plt.title("Markowitz efficient frontier")
    plt.xlabel("Risk")
    plt.ylabel("Return")
    feasible_mu = []
    feasible_sig = []
    a = np.linspace(0,1,100)
    b = np.linspace(0,1,100)

    noa_mu = []
    noa_sig = []
    nob_mu = []
    nob_sig = []
    noc_mu = []
    noc_sig = []



    for w1 in a:
        t1 = np.array([0,w1,1-w1])
        t2 = np.array([w1,0,1-w1])
        t3 = np.array([w1,1-w1,0])
        noa_mu.append(t1@mt)
        nob_mu.append(t2@mt)
        noc_mu.append(t3@mt)
        noa_sig.append(getRisk(t1))
        nob_sig.append(getRisk(t2))
        noc_sig.append(getRisk(t3))




    for w1 in a:
        for w2 in b:
            if((w1+w2)<=1):
                t1 = np.array([w1,w2,1-w1-w2])
                t2 = np.array([1-w1-w2,w1,w2])
                t3 = np.array([w1,1-w1-w2,w2])
                feasible_mu.append(t1@mt)
                feasible_mu.append(t2@mt)
                feasible_mu.append(t3@mt)
                feasible_sig.append(getRisk(t1))
                feasible_sig.append(getRisk(t2))
                feasible_sig.append(getRisk(t3))

    
    plt.scatter(feasible_sig,feasible_mu,color = "lightgoldenrodyellow",alpha = 0.5)
    plt.plot(noa_sig,noa_mu)
    plt.plot(nob_sig,nob_mu)
    plt.plot(noc_sig,noc_mu)
    #print(np.shape(feasible_mu),np.shape(feasible_sig))
    plt.legend(["Markowitz frontier without short selling","feasible region without short selling","without stock 1","without stock 2","withput stock 3"])
    plt.show()
    
        
    weights18 = getWeights(0.18)
    print("\n\nFor return = 18%:\nWeights = ",weights18,"\nRisk = ",getRisk(weights18)*100,"%")


def plotWeights():
    returns = np.linspace(0,0.3,5000)
    w1 = []
    w2 = []
    w3 = []
    for x in returns:
        w = getWeights(x)
        w1.append(w[0])
        w2.append(w[1])
        w3.append(w[2])
        
    plt.style.use("seaborn")
    plt.plot(w1,w2)
    plt.xlabel("w1")
    plt.ylabel("w2")
    plt.title("Weights on minimum variance curve w1vsw2")
    plt.show()
    plt.plot(w2,w3)
    plt.xlabel("w2")
    plt.ylabel("w3")
    plt.title("Weights on minimum variance curve w2vsw3")
    plt.show()
    plt.plot(w3,w1)
    plt.title("Weights on minimum variance curve w3vsw1")
    plt.xlabel("w3")
    plt.ylabel("w1")
    plt.show()
    
    m, c = get_eqn(w1, w2)
    print("Eqn of line w1 vs w2 is:")
    print("w2 = {:.2f} w1 + {:.2f}".format(m, c))
    m, c = get_eqn(w2, w3)
    print("Eqn of line w2 vs w3 is:")
    print("w3 = {:.2f} w2 + {:.2f}".format(m, c))
    m, c = get_eqn(w3, w1)
    print("Eqn of line w3 vs w1 is:")
    print("w1 = {:.2f} w3 + {:.2f}".format(m, c))
    
        
markowitzBulletLine()
plotWeights()