import math
import numpy as np
from matplotlib import pyplot as plt
def binomialModel(M):
	S0 = 100
	K = 105
	T = 5
	r = 0.05
	sigma = 0.3
	delta = T/M
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
			priceCall[j] = math.exp(-r*delta)*(p*priceCall[j] + q*priceCall[j+1])
			pricePut[j] = math.exp(-r*delta)*(p*pricePut[j] + q*pricePut[j+1])
	return [priceCall[0],pricePut[0]]

def solve(stepSize):
	plt.style.use("ggplot")
	steps = np.arange(1,400,stepSize)
	priceCall = []
	pricePut = []
	for M in steps:
		x = binomialModel(M)
		priceCall.append(x[0])
		pricePut.append(x[1])
	plt.xlabel("Number of Steps")
	plt.ylabel("Price")
	plt.title("Price of Call option vs M with steps of {x}".format(x=stepSize))
	plt.plot(steps,priceCall)
	plt.show()
	plt.xlabel("Number of Steps")
	plt.ylabel("Price")
	plt.title("Price of Put option vs M with steps of {x}".format(x=stepSize))
	plt.plot(steps,pricePut)
	plt.show()

for stepSize in [1,5]:
	solve(stepSize)