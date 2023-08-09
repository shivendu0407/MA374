import math
import time


S0 = 100
T = 1
M = 25
r = 0.08
sigma = 0.2
cache = []
for i in range(M+1):
	cache.append(dict())
delta = T/M
u = math.exp((sigma*math.sqrt(delta))+delta*(r-(sigma*sigma)/2))
d = math.exp((-sigma*math.sqrt(delta))+delta*(r-(sigma*sigma)/2))
p = (math.exp(r*delta)-d)/(u-d)
q = (u-math.exp(r*delta))/(u-d)

def binomialModelMarkov(S,currentMax,step):
	if((S,currentMax) in cache[step]):
		return cache[step][(S,currentMax)]
	val = 0;
	if(step==M):
		val = currentMax - S
	else:
		up = binomialModelMarkov(S*u,max(currentMax,S*u),step+1)
		down = binomialModelMarkov(S*d,currentMax,step+1)
		val = math.exp(-r*delta)*(p*up + q*down)
	cache[step][(S,currentMax)] = val
	return val;



def displayPrice(cache):
    for i in range(6):
        print("\nFor t = ",i)
        print("{0:15s} {1:15s} {2:15s}".format("Stock Price","Max Price","Option Price"))
        for k,v in cache[i].items():
            print("{0:3.10f}    {1:3.10f}      {2:3.7f}".format(k[0],k[1],v))

for m in [5,10,25,26,27,50]:
	print("\n\nFor  m = ",m)
	M = m
	delta = T/M
	u = math.exp((sigma*math.sqrt(delta))+delta*(r-(sigma*sigma)/2))
	d = math.exp((-sigma*math.sqrt(delta))+delta*(r-(sigma*sigma)/2))
	p = (math.exp(r*delta)-d)/(u-d)
	q = (u-math.exp(r*delta))/(u-d)
	cache.clear()
	for i in range(M+1):
		cache.append(dict())
	init = time.time()
	print("Cost of Lookback option = ",binomialModelMarkov(S0,S0,0))
	print("\nComputation time = ",time.time()-init,"seconds")
	if(M==5):
		displayPrice(cache)