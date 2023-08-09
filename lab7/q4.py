from matplotlib import pyplot as plt
plt.style.use("seaborn")
import numpy as np
import math
import pandas as pd
from scipy.stats import norm


def bsmOption(t,s,k,sig,r,T):
	if(T==t):
		return [max(0,s-k),max(0,k-s)]
	d1 = (1/(sig*((T-t)**0.5)))*(math.log(s/k) + ((r + (sig*sig/2))*(T-t)))
	d2 = d1 - sig*((T-t)**0.5)
	c = (s*norm.cdf(d1) - (k*math.exp(-r*(T-t))*norm.cdf(d2)))
	p = (c - s + (k*math.exp(-r*(T-t))))
	return [c,p]

def tabulate(a,b,calls,puts):
    print("\n\nTable of prices with respect to different values of {k}:".format(k=a))
    df = pd.DataFrame()
    r = [0,100,500,1000,1500,2000,2500,3500,4000,4500]
    #print(r)
    df[a] = b[r]
    df["C(t,s)"] = np.array(calls)[r]
    df["P(t,s)"] = np.array(puts)[r]
    print(df)


calls = []
puts = []
for s in [0.4,0.8,1.2,1.6]:
	callPrices = []
	putPrices = []
	for k in np.linspace(0.001,2,5000):
		t1 = bsmOption(0,s,k,0.6,0.05,1)
		callPrices.append(t1[0])
		putPrices.append(t1[1])
	calls.append(callPrices)
	puts.append(putPrices)
for i in range(4):	
	plt.plot(np.linspace(0.001,2,5000),calls[i])
plt.xlabel("k")
plt.ylabel("C(t,s)")
plt.legend(["s=0.4","s=0.8","s=1.2","s=1.6"])
plt.title("Variation of call prices with k")
plt.show()


for i in range(4):	
	plt.plot(np.linspace(0.001,2,5000),puts[i])
plt.xlabel("k")
plt.ylabel("P(t,s)")
plt.legend(["s=0.4","s=0.8","s=1.2","s=1.6"])
plt.title("Variation of put prices with k")
plt.show()

calls = []
puts = []
for t in [0,0.2,0.4,0.6,0.8,1]:
	callPrices = []
	putPrices = []
	for k in np.linspace(0.001,2,5000):
		t1 = bsmOption(t,1,k,0.6,0.05,1)
		callPrices.append(t1[0])
		putPrices.append(t1[1])
	calls.append(callPrices)
	puts.append(putPrices)
for i in range(6):	
	plt.plot(np.linspace(0.001,2,5000),calls[i])
plt.xlabel("k")
plt.ylabel("C(t,s)")
plt.legend(["t = 0","t = 0.2","t = 0.4","t = 0.6","t = 0.8","t = 1"])
plt.title("Variation of call prices with k")
plt.show()
tabulate("k",np.linspace(0.001,2,5000),calls[0],puts[0])

for i in range(4):	
	plt.plot(np.linspace(0.001,2,5000),puts[i])
plt.xlabel("k")
plt.ylabel("P(t,s)")
plt.legend(["t = 0","t = 0.2","t = 0.4","t = 0.6","t = 0.8","t = 1"])
plt.title("Variation of put prices with k")
plt.show()



calls = []
puts = []
for s in [0.4,0.8,1.2,1.6]:
	callPrices = []
	putPrices = []
	for sig in np.linspace(0.001,1,5000):
		t1 = bsmOption(0,s,1,sig,0.05,1)
		callPrices.append(t1[0])
		putPrices.append(t1[1])
	calls.append(callPrices)
	puts.append(putPrices)
for i in range(4):	
	plt.plot(np.linspace(0.001,1,5000),calls[i])
plt.xlabel("k")
plt.ylabel("C(t,s)")
plt.legend(["s=0.4","s=0.8","s=1.2","s=1.6"])
plt.title("Variation of call prices with volatility")
plt.show()


for i in range(4):	
	plt.plot(np.linspace(0.001,1,5000),puts[i])
plt.xlabel("k")
plt.ylabel("P(t,s)")
plt.legend(["s=0.4","s=0.8","s=1.2","s=1.6"])
plt.title("Variation of put prices with volatility")
plt.show()

calls = []
puts = []
for t in [0,0.2,0.4,0.6,0.8,1]:
	callPrices = []
	putPrices = []
	for sig in np.linspace(0.001,1,5000):
		t1 = bsmOption(t,1,1,sig,0.05,1)
		callPrices.append(t1[0])
		putPrices.append(t1[1])
	calls.append(callPrices)
	puts.append(putPrices)
for i in range(6):	
	plt.plot(np.linspace(0.001,1,5000),calls[i])
plt.xlabel("volatility")
plt.ylabel("C(t,s)")
plt.legend(["t = 0","t = 0.2","t = 0.4","t = 0.6","t = 0.8","t = 1"])
plt.title("Variation of call prices with volatility")
plt.show()
tabulate("volatility",np.linspace(0.001,1,5000),calls[0],puts[0])

for i in range(4):	
	plt.plot(np.linspace(0.001,1,5000),puts[i])
plt.xlabel("volatility")
plt.ylabel("P(t,s)")
plt.legend(["t = 0","t = 0.2","t = 0.4","t = 0.6","t = 0.8","t = 1"])
plt.title("Variation of put prices with volatility")
plt.show()

calls = []
puts = []
for s in [0.4,0.8,1.2,1.6]:
	callPrices = []
	putPrices = []
	for r in np.linspace(0.001,1,5000):
		t1 = bsmOption(0,s,1,0.6,r,1)
		callPrices.append(t1[0])
		putPrices.append(t1[1])
	calls.append(callPrices)
	puts.append(putPrices)
for i in range(4):	
	plt.plot(np.linspace(0.001,1,5000),calls[i])
plt.xlabel("r")
plt.ylabel("C(t,s)")
plt.legend(["s=0.4","s=0.8","s=1.2","s=1.6"])
plt.title("Variation of call prices with r")
plt.show()

for i in range(4):	
	plt.plot(np.linspace(0.001,1,5000),puts[i])
plt.xlabel("r")
plt.ylabel("P(t,s)")
plt.legend(["s=0.4","s=0.8","s=1.2","s=1.6"])
plt.title("Variation of put prices with r")
plt.show()

def plotFig(calls,puts,T,S,a,b):
	ax = plt.axes(projection="3d")
	ax.plot_surface(T,S,calls)
	ax.set_xlabel(a)
	ax.set_ylabel(b)
	ax.set_zlabel("C(t,s)")
	ax.set_title("C(t,s) vs {a} and {b}".format(a=a,b=b))
	plt.show()
	ax = plt.axes(projection="3d")
	ax.plot_surface(T,S,puts)
	ax.set_xlabel(a)
	ax.set_ylabel(b)
	ax.set_zlabel("P(t,s)")
	ax.set_title("P(t,s) vs {a} and {b}".format(a=a,b=b))
	plt.show()


calls = []
puts = []
for t in [0,0.2,0.4,0.6,0.8,1]:
	callPrices = []
	putPrices = []
	for r in np.linspace(0.001,1,5000):
		t1 = bsmOption(t,1,1,0.6,r,1)
		callPrices.append(t1[0])
		putPrices.append(t1[1])
	calls.append(callPrices)
	puts.append(putPrices)
for i in range(6):	
	plt.plot(np.linspace(0.001,1,5000),calls[i])
plt.xlabel("r")
plt.ylabel("C(t,s)")
plt.legend(["t = 0","t = 0.2","t = 0.4","t = 0.6","t = 0.8","t = 1"])
plt.title("Variation of call prices with r")
plt.show()
tabulate("r",np.linspace(0.001,1,5000),calls[0],puts[0])

for i in range(4):	
	plt.plot(np.linspace(0.001,1,5000),puts[i])
plt.xlabel("r")
plt.ylabel("P(t,s)")
plt.legend(["t = 0","t = 0.2","t = 0.4","t = 0.6","t = 0.8","t = 1"])
plt.title("Variation of put prices with r")
plt.show()

calls = []
puts = []
T,K = np.meshgrid(np.linspace(0,1,50),np.linspace(0.001,2,100))
calls = np.empty(np.shape(T))
puts = np.empty(np.shape(T))
for i in range(np.shape(T)[0]):
	for j in range(np.shape(T)[1]):
		t = bsmOption(T[i,j],1,K[i,j],0.6,0.05,1)
		calls[i,j] = t[0]
		puts[i,j] = t[1]
plotFig(calls,puts,T,K,"t","k")

calls = []
puts = []
T,sig = np.meshgrid(np.linspace(0,1,50),np.linspace(0.001,1,50))
calls = np.empty(np.shape(T))
puts = np.empty(np.shape(T))
for i in range(np.shape(T)[0]):
	for j in range(np.shape(T)[1]):
		t = bsmOption(T[i,j],1,1,sig[i,j],0.05,1)
		calls[i,j] = t[0]
		puts[i,j] = t[1]
plotFig(calls,puts,T,sig,"t","volatility")

calls = []
puts = []
T,r = np.meshgrid(np.linspace(0,1,50),np.linspace(0.001,1,50))
calls = np.empty(np.shape(T))
puts = np.empty(np.shape(T))
for i in range(np.shape(T)[0]):
	for j in range(np.shape(T)[1]):
		t = bsmOption(T[i,j],1,1,0.6,r[i,j],1)
		calls[i,j] = t[0]
		puts[i,j] = t[1]
plotFig(calls,puts,T,r,"t","r")

calls = []
puts = []
T,r = np.meshgrid(np.linspace(0,1,50),np.linspace(1,5,100))
calls = np.empty(np.shape(T))
puts = np.empty(np.shape(T))
for i in range(np.shape(T)[0]):
	for j in range(np.shape(T)[1]):
		t = bsmOption(T[i,j],1,1,0.6,0.05,r[i,j])
		calls[i,j] = t[0]
		puts[i,j] = t[1]
plotFig(calls,puts,T,r,"t","T")

calls = []
puts = []
S,K = np.meshgrid(np.linspace(0.001,2,100),np.linspace(0.001,2,100))
calls = np.empty(np.shape(S))
puts = np.empty(np.shape(S))
for i in range(np.shape(S)[0]):
	for j in range(np.shape(S)[1]):
		t = bsmOption(0,S[i,j],K[i,j],0.6,0.05,1)
		calls[i,j] = t[0]
		puts[i,j] = t[1]
plotFig(calls,puts,S,K,"s","k")

calls = []
puts = []
S,sig = np.meshgrid(np.linspace(0.001,2,100),np.linspace(0.001,1,50))
calls = np.empty(np.shape(S))
puts = np.empty(np.shape(S))
for i in range(np.shape(S)[0]):
	for j in range(np.shape(S)[1]):
		t = bsmOption(0,S[i,j],1,sig[i,j],0.05,1)
		calls[i,j] = t[0]
		puts[i,j] = t[1]
plotFig(calls,puts,S,sig,"s","volatility")

calls = []
puts = []
S,r = np.meshgrid(np.linspace(0.001,2,100),np.linspace(0.001,1,50))
calls = np.empty(np.shape(S))
puts = np.empty(np.shape(S))
for i in range(np.shape(S)[0]):
	for j in range(np.shape(S)[1]):
		t = bsmOption(0,S[i,j],1,0.6,r[i,j],1)
		calls[i,j] = t[0]
		puts[i,j] = t[1]
plotFig(calls,puts,S,r,"s","r")

calls = []
puts = []
S,T = np.meshgrid(np.linspace(0.001,2,100),np.linspace(0,1,50))
calls = np.empty(np.shape(S))
puts = np.empty(np.shape(S))
for i in range(np.shape(S)[0]):
	for j in range(np.shape(S)[1]):
		t = bsmOption(0,S[i,j],1,0.6,0.05,T[i,j])
		calls[i,j] = t[0]
		puts[i,j] = t[1]
plotFig(calls,puts,S,T,"s","T")

calls = []
puts = []
K,sig = np.meshgrid(np.linspace(0.001,2,100),np.linspace(0.001,1,50))
calls = np.empty(np.shape(S))
puts = np.empty(np.shape(S))
for i in range(np.shape(S)[0]):
	for j in range(np.shape(S)[1]):
		t = bsmOption(0,1,K[i,j],sig[i,j],0.05,1)
		calls[i,j] = t[0]
		puts[i,j] = t[1]
plotFig(calls,puts,K,sig,"k","volatility")

calls = []
puts = []
K,r = np.meshgrid(np.linspace(0.001,2,100),np.linspace(0.001,1,50))
calls = np.empty(np.shape(S))
puts = np.empty(np.shape(S))
for i in range(np.shape(S)[0]):
	for j in range(np.shape(S)[1]):
		t = bsmOption(0,1,K[i,j],0.6,r[i,j],1)
		calls[i,j] = t[0]
		puts[i,j] = t[1]
plotFig(calls,puts,K,r,"k","r")

calls = []
puts = []
K,T = np.meshgrid(np.linspace(0.001,2,100),np.linspace(0,1,50))
calls = np.empty(np.shape(S))
puts = np.empty(np.shape(S))
for i in range(np.shape(S)[0]):
	for j in range(np.shape(S)[1]):
		t = bsmOption(0,1,K[i,j],0.6,0.05,T[i,j])
		calls[i,j] = t[0]
		puts[i,j] = t[1]
plotFig(calls,puts,K,T,"k","T")

calls = []
puts = []
sig,r = np.meshgrid(np.linspace(0.001,1,100),np.linspace(0.001,1,50))
calls = np.empty(np.shape(S))
puts = np.empty(np.shape(S))
for i in range(np.shape(S)[0]):
	for j in range(np.shape(S)[1]):
		t = bsmOption(0,1,1,sig[i,j],r[i,j],1)
		calls[i,j] = t[0]
		puts[i,j] = t[1]
plotFig(calls,puts,sig,r,"volatility","r")

calls = []
puts = []
sig,T = np.meshgrid(np.linspace(0.001,1,100),np.linspace(0,1,50))
calls = np.empty(np.shape(S))
puts = np.empty(np.shape(S))
for i in range(np.shape(S)[0]):
	for j in range(np.shape(S)[1]):
		t = bsmOption(0,1,1,sig[i,j],0.05,T[i,j])
		calls[i,j] = t[0]
		puts[i,j] = t[1]
plotFig(calls,puts,sig,T,"volatility","T")

calls = []
puts = []
r,T = np.meshgrid(np.linspace(0.001,1,100),np.linspace(0,1,50))
calls = np.empty(np.shape(S))
puts = np.empty(np.shape(S))
for i in range(np.shape(S)[0]):
	for j in range(np.shape(S)[1]):
		t = bsmOption(0,1,1,0.6,r[i,j],T[i,j])
		calls[i,j] = t[0]
		puts[i,j] = t[1]
plotFig(calls,puts,r,T,"r","T")