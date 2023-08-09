import math
import numpy as np
from matplotlib import pyplot as plt


def binomialModel(M,S0,K,T,r,sigma):
	delta = T/M
	u = d = 0
	u = math.exp((sigma*math.sqrt(delta))+delta*(r-(sigma*sigma)/2))
	d = math.exp((-sigma*math.sqrt(delta))+delta*(r-(sigma*sigma)/2))
	p = (math.exp(r*delta)-d)/(u-d)
	q = (u-math.exp(r*delta))/(u-d)
	priceCall = []
	pricePut = []
	current = S0*(u**M)
	for i in range(M+1):
		priceCall.append(max(0,current-K))
		pricePut.append(max(0,K-current))
		current = current*d/u
	for i in range(M):
		for j in range(M-i):
			priceStock = max(S0*((u)**(M-i-j-1))*((d)**j)-K,0)
			priceStock1 = max(-S0*((u)**(M-i-j-1))*((d)**j)+(K),0)
			priceCall[j] = max(priceStock,math.exp(-r*delta)*(p*priceCall[j] + q*priceCall[j+1]))
			pricePut[j] = max(priceStock1,math.exp(-r*delta)*(p*pricePut[j] + q*pricePut[j+1]))
	return [priceCall[0],pricePut[0]]



S0 = 100
K = 100
T = 1
M = 100
r = 0.08
sigma = 0.2

x = binomialModel(M,S0,K,T,r,sigma)
print("Price of American call option = ",x[0])
print("Price of American put option = ",x[1])


plt.style.use("ggplot")
calls = []
puts = []
for S in np.arange(0,205,5):
	x = binomialModel(M,S,K,T,r,sigma)
	calls.append(x[0])
	puts.append(x[1])
plt.plot(np.arange(0,205,5),calls)
plt.plot(np.arange(0,205,5),puts)
plt.legend(["Call Option","Put Option"])
plt.xlabel("S0")
plt.ylabel("Price")
plt.title("Price  of American Call and Put Option vs S0")
plt.show()


calls = []
puts = []
for k in np.arange(0,205,5):
	x = binomialModel(M,S0,k,T,r,sigma)
	calls.append(x[0])
	puts.append(x[1])
plt.plot(np.arange(0,205,5),calls)
plt.plot(np.arange(0,205,5),puts)
plt.legend(["Call Option","Put Option"])
plt.xlabel("K")
plt.ylabel("Price")
plt.title("Price  of American Call and Put Option vs K")
plt.show()



calls = []
puts = []
for R in np.linspace(0,0.2,100):
	x = binomialModel(M,S0,K,T,R,sigma)
	calls.append(x[0])
	puts.append(x[1])
plt.plot(np.linspace(0,0.2,100),calls)
plt.plot(np.linspace(0,0.2,100),puts)
plt.legend(["Call Option","Put Option"])
plt.xlabel("r")
plt.ylabel("Price")
plt.title("Price  of American Call and Put Option vs r")
plt.show()


calls = []
puts = []
for sig in np.linspace(0.1,0.3,100):
	x = binomialModel(M,S0,K,T,r,sig)
	calls.append(x[0])
	puts.append(x[1])
plt.xlabel("sigma")
plt.ylabel("Price")
plt.title("Price  of American Call Option vs sigma")
plt.plot(np.linspace(0.1,0.5,100),calls)
plt.show()
plt.xlabel("sigma")
plt.ylabel("Price")
plt.title("Price  of American Put Option vs sigma")
plt.plot(np.linspace(0.1,0.5,100),puts)
plt.show()

for K in [95,100,105]:
	calls = []
	puts = []
	for m in np.arange(1,201):
		x = binomialModel(m,S0,K,T,r,sigma)
		calls.append(x[0])
		puts.append(x[1])
	plt.xlabel("M")
	plt.ylabel("Price")
	plt.title("Price  of American Call Option vs M and K = {K}".format(K = K))
	plt.plot(np.arange(1,201),calls)
	plt.show()
	plt.xlabel("M")
	plt.ylabel("Price")
	plt.title("Price  of American Put Option vs N and K = {K}".format(K = K))
	plt.plot(np.arange(1,201),puts)
	plt.show()	
