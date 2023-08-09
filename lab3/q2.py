import math
import numpy as np
from matplotlib import pyplot as plt
import time


def printPrice(optionPrices,i):
	for j in range(2**i):
		currentPath = ""
		for t in range(i):
			if(j&(1<<(i-1-t))):
				currentPath += "T"
			else:
				currentPath += "H"
		print("Path = ",currentPath,"Price = ",optionPrices[j])

def binomialModel(M,S0,T,r,sigma):
	delta = T/M
	u = math.exp((sigma*math.sqrt(delta))+delta*(r-(sigma*sigma)/2))
	d = math.exp((-sigma*math.sqrt(delta))+delta*(r-(sigma*sigma)/2))
	p = (math.exp(r*delta)-d)/(u-d)
	q = (u-math.exp(r*delta))/(u-d)
	optionPrices = []
	for i in range(2**M):
		tot = S0;
		currentPrice = S0
		for j in range(M):
			if(i&(1<<(M-1-j))):
				currentPrice = currentPrice*d
			else:
				currentPrice = currentPrice*u
			tot = max(tot,currentPrice)
		optionPrices.append(tot-currentPrice)
	for i in range(M):
		if(M==5):
			print("\n\nOption Prices at stage ",M-i)
			printPrice(optionPrices,M-i)
		for j in range(2**(M-1-i)):
			optionPrices[j] = math.exp(-r*delta)*(p*optionPrices[2*j] + q*optionPrices[2*j+1])
	return optionPrices[0]

S0 = 100
T = 1
r = 0.08
sigma = 0.2

for M in [5,10,25]:
	print("\n\nFor M = ",M)
	init = time.time()
	print("Option Price at time 0 = ",binomialModel(M,S0,T,r,sigma))
	print("\nComputation Time = ",time.time()-init,"seconds")