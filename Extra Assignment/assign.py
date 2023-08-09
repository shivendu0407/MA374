from matplotlib import pyplot as plt
plt.style.use("dark_background")
import numpy as np
import math
from scipy.stats import norm

def binaryOption(S0,r,t,T,sig,Q):
	if(T==t):
		return [Q,0]
	tau = T - t
	d1 = (1/(sig*(tau**0.5)))*(math.log(S0/Q) + (r-0.5*sig*sig)*tau)
	return [Q*math.exp(-tau)*norm.cdf(d1),Q*math.exp(-tau)*norm.cdf(-d1)]

def binaryOption1(t,S0,Q,sig,r,T):
	return binaryOption(S0,r,t,T,sig,Q)

print(binaryOption(100,0.1,0,1,0.15,90))

callPrices = []
putPrices = []
P = np.arange(50,150)
for S0 in P:
	u = binaryOption(S0,0.1,0,1,0.15,90)
	callPrices.append(u[0])
	putPrices.append(u[1])

fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
ax1.plot(P,callPrices)
ax2.plot(P,putPrices)
ax1.set_title("Variation of Binary Call Option with S0")
ax1.set_xlabel("S0")
ax1.set_ylabel("Price")
ax2.set_title("Variation of Binary Put Option with S0")
ax2.set_xlabel("S0")
ax2.set_ylabel("Price")
plt.show()

callPrices = []
putPrices = []
P = np.arange(50,150)
for K in P:
	u = binaryOption(100,0.1,0,1,0.15,K)
	callPrices.append(u[0])
	putPrices.append(u[1])

fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
ax1.plot(P,callPrices)
ax2.plot(P,putPrices)
ax1.set_title("Variation of Binary Call Option with S0")
ax1.set_xlabel("S0")
ax1.set_ylabel("Price")
ax2.set_title("Variation of Binary Put Option with S0")
ax2.set_xlabel("S0")
ax2.set_ylabel("Price")
plt.show()

callPrices = []
putPrices = []
P = np.linspace(0,1,5000)
for r in P:
	u = binaryOption(100,r,0,1,0.15,90)
	callPrices.append(u[0])
	putPrices.append(u[1])

fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
ax1.plot(P,callPrices)
ax2.plot(P,putPrices)
ax1.set_title("Variation of Binary Call Option with S0")
ax1.set_xlabel("S0")
ax1.set_ylabel("Price")
ax2.set_title("Variation of Binary Put Option with S0")
ax2.set_xlabel("S0")
ax2.set_ylabel("Price")
plt.show()

callPrices = []
putPrices = []
P = np.linspace(0.01,1,5000)
for sig in P:
	u = binaryOption(100,0.1,0,1,sig,90)
	callPrices.append(u[0])
	putPrices.append(u[1])

fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
ax1.plot(P,callPrices)
ax2.plot(P,putPrices)
ax1.set_title("Variation of Binary Call Option with S0")
ax1.set_xlabel("S0")
ax1.set_ylabel("Price")
ax2.set_title("Variation of Binary Put Option with S0")
ax2.set_xlabel("S0")
ax2.set_ylabel("Price")
plt.show()

callPrices = []
putPrices = []
P = np.linspace(0,1,1000)
for t in P:
	u = binaryOption(100,0.1,t,1,0.15,90)
	callPrices.append(u[0])
	putPrices.append(u[1])

fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
ax1.plot(P,callPrices)
ax2.plot(P,putPrices)
ax1.set_title("Variation of Binary Call Option with S0")
ax1.set_xlabel("S0")
ax1.set_ylabel("Price")
ax2.set_title("Variation of Binary Put Option with S0")
ax2.set_xlabel("S0")
ax2.set_ylabel("Price")
plt.show()

callPrices = []
putPrices = []
P = np.linspace(0,3,1000)
for T in P:
	u = binaryOption(100,0.1,0,T,0.15,90)
	callPrices.append(u[0])
	putPrices.append(u[1])
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
ax1.plot(P,callPrices)
ax2.plot(P,putPrices)
ax1.set_title("Variation of Binary Call Option with S0")
ax1.set_xlabel("S0")
ax1.set_ylabel("Price")
ax2.set_title("Variation of Binary Put Option with S0")
ax2.set_xlabel("S0")
ax2.set_ylabel("Price")
plt.show()

def plotFig(calls,puts,T,S,a,b):
	fig,(ax1,ax2) = plt.subplots(nrows=1,ncols = 2,subplot_kw = {"projection": "3d"})
	ax1.plot_surface(T,S,calls,cmap='viridis')
	ax2.plot_surface(T,S,puts,cmap='viridis')
	ax1.set_xlabel(a)
	ax1.set_ylabel(b)
	ax1.set_zlabel("C(t,s)")
	ax1.set_title("C(t,s) vs {a} and {b}".format(a=a,b=b))
	ax2.set_xlabel(a)
	ax2.set_ylabel(b)
	ax2.set_zlabel("P(t,s)")
	ax2.set_title("P(t,s) vs {a} and {b}".format(a=a,b=b))
	plt.show()


calls = []
puts = []
T,K = np.meshgrid(np.linspace(0,1,50),np.linspace(0.001,2,100))
calls = np.empty(np.shape(T))
puts = np.empty(np.shape(T))
for i in range(np.shape(T)[0]):
	for j in range(np.shape(T)[1]):
		t = binaryOption1(T[i,j],1,K[i,j],0.6,0.05,1)
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
		t = binaryOption1(T[i,j],1,1,sig[i,j],0.05,1)
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
		t = binaryOption1(T[i,j],1,1,0.6,r[i,j],1)
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
		t = binaryOption1(T[i,j],1,1,0.6,0.05,r[i,j])
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
		t = binaryOption1(0,S[i,j],K[i,j],0.6,0.05,1)
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
		t = binaryOption1(0,S[i,j],1,sig[i,j],0.05,1)
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
		t = binaryOption1(0,S[i,j],1,0.6,r[i,j],1)
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
		t = binaryOption1(0,S[i,j],1,0.6,0.05,T[i,j])
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
		t = binaryOption1(0,1,K[i,j],sig[i,j],0.05,1)
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
		t = binaryOption1(0,1,K[i,j],0.6,r[i,j],1)
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
		t = binaryOption1(0,1,K[i,j],0.6,0.05,T[i,j])
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
		t = binaryOption1(0,1,1,sig[i,j],r[i,j],1)
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
		t = binaryOption1(0,1,1,sig[i,j],0.05,T[i,j])
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
		t = binaryOption1(0,1,1,0.6,r[i,j],T[i,j])
		calls[i,j] = t[0]
		puts[i,j] = t[1]
plotFig(calls,puts,r,T,"r","T")