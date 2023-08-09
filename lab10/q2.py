from matplotlib import pyplot as plt
import numpy as np
import math



def boxMuller():
	y1 = np.random.uniform()
	y2 = np.random.uniform()
	return 0.5*(math.cos(2*math.pi*(1-y2))*math.sqrt(-2*math.log(1-y1)) + math.cos(2*math.pi*(y2))*math.sqrt(-2*math.log(y1)))

def generatePath(s0,mu,sig,t,n=10000,method = ""):
	times = np.linspace(0,t,n)
	prices = []
	prices.append(s0)
	delta = times[1]-times[0]
	current = s0
	for i in range(1,len(times)):
		if(method!="c"): 
			current = current*math.exp(((np.random.normal()*(delta**0.5))*(sig)) + ((mu-(sig*sig/2))*delta))
		elif(method=="c"):
			u = boxMuller()
			current = current*math.exp(((u*(delta**0.5))*(sig)) + ((mu-(sig*sig/2))*delta))
		prices.append(current)
	return prices

def getOptionPrice(K,s0,mu,sig,t,n=10000,N = 2000,method = ""):
	prices = []
	prices1 = []
	for i in range(N):
		x = generatePath(s0,mu,sig,t,n,method)
		prices.append(max(0,math.exp(-mu*(t))*(np.mean(x)-K)))
		prices1.append(max(0,math.exp(-mu*(t))*(-np.mean(x)+K)))
	if(method=="d"):
		Y = []
		for i in range(N):
			Y.append(np.random.normal())
		c1 = -np.cov(prices,Y)[1][0]/np.var(Y)
		c2 = -np.cov(prices1,Y)[1][0]/np.var(Y)
		mu = np.mean(Y)
		for i in range(N):
			prices[i] = prices[i] + (c1*(Y[i]-mu))
			prices1[i] = prices1[i] +(c2*(Y[i]-mu))
	return [np.mean(prices),N*np.var(prices)/(N-1),np.mean(prices1),N*np.var(prices1)/(N-1)]

print("\n\nUsing antithetic variables method of variance reduction:")
for K in [105,110,90]:
	print("\n\nFor K = ",K)
	m = getOptionPrice(K,100,0.05,0.2,0.5,126,1000,"c")
	print("Calculated Price of Asian call option = ",m[0])
	print("Variance of the estimate = ",m[1])
	print("\nCalculated Price of Asian put option = ",m[2])
	print("Variance of the estimate = ",m[3])
 
print("\n\nUsing control variate method of variance reduction")
for K in [105,110,90]:
	print("\n\nFor K = ",K)
	m = getOptionPrice(K,100,0.05,0.2,0.5,126,1000,"d")
	print("Calculated Price of Asian call option = ",m[0])
	print("Variance of the estimate = ",m[1])
	print("\nCalculated Price of Asian put option = ",m[2])
	print("Variance of the estimate = ",m[3])
 
 
for K in [105,110,90]:
	print("\n\nFor K = ",K)
	m = getOptionPrice(K,100,0.05,0.2,0.5,126,1000,"c")
	print("Calculated Price of Asian call option = ",m[0])
	print("Variance of the estimate = ",m[1])
	print("\nCalculated Price of Asian put option = ",m[2])
	print("Variance of the estimate = ",m[3])

callPrices = []
putPrices = []
k = np.arange(50,151,2)
for K in k:
	price = getOptionPrice(K,100,0.05,0.2,0.5,126,1000,"c")
	callPrices.append(price[0])
	putPrices.append(price[2])
plt.plot(k,callPrices)
plt.plot(k,putPrices)
plt.legend(["Call Option","Put Option"])
plt.xlabel("K")
plt.ylabel("Asian Option Price")
plt.title("Asian Option vs K")
plt.show()


callPrices = []
putPrices = []
R = np.linspace(0.001,1,100)
for r in R:   
	price = getOptionPrice(105,100,r,0.2,0.5,126,1000,"c")
	callPrices.append(price[0])
	putPrices.append(price[2])
plt.plot(R,callPrices)
plt.plot(R,putPrices)
plt.legend(["Call Option","Put Option"])
plt.xlabel("r")
plt.ylabel("Asian Option Price")
plt.title("Asian Option vs r")
plt.show()

callPrices = []
putPrices = []
SIG = np.linspace(0.001,1,100)
for sig in SIG:
	price = (getOptionPrice(105,100,0.05,sig,0.5,126,1000,"c"))
	callPrices.append(price[0])
	putPrices.append(price[2])
plt.plot(SIG,callPrices)
plt.plot(SIG,putPrices)
plt.legend(["Call Option","Put Option"])
plt.xlabel("volatility")
plt.ylabel("Asian Option Price")
plt.title("Asian Option vs volatility")
plt.show()

callPrices = []
putPrices = []
S = np.arange(50,151)
for s in S:
	price = getOptionPrice(105,s,0.05,0.2,0.5,126,1000,"c")
	callPrices.append(price[0])
	putPrices.append(price[2])
plt.plot(S,callPrices)
plt.plot(S,putPrices)
plt.legend(["Call Option","Put Option"])
plt.xlabel("S")
plt.ylabel("Asian Option Price")
plt.title("Asian Option vs S0")
plt.show()

callPrices = []
putPrices = []
T = np.linspace(2/252,1)
for t in T:
	price = getOptionPrice(105,100,0.05,0.2,t,int(252*t),1000,"c")
	callPrices.append(price[0])
	putPrices.append(price[2])
plt.plot(T,callPrices)
plt.plot(T,putPrices)
plt.legend(["Call Option","Put Option"])
plt.xlabel("T")
plt.ylabel("Asian Option Price")
plt.title("Asian Option vs T")
plt.show()