import math
import time


S0 = 100
T = 1
M = 50
r = 0.08
K = 100
sigma = 0.2
cache = []
delta = T/M
u = math.exp((sigma*math.sqrt(delta))+delta*(r-(sigma*sigma)/2))
d = math.exp((-sigma*math.sqrt(delta))+delta*(r-(sigma*sigma)/2))
p = (math.exp(r*delta)-d)/(u-d)
q = (u-math.exp(r*delta))/(u-d)

def binomialModelMarkov(S,step):
	if S in cache[step]:
		return cache[step][S]
	val = 0;
	if(step==M):
		val = max(S-K,0)
	else:
		up = binomialModelMarkov(S*u,step+1)
		down = binomialModelMarkov(S*d,step+1)
		val = math.exp(-r*delta)*(p*up + q*down)
	cache[step][S] = val
	return val;


for m in [5,10,25,50,100,500]:
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
	print("Cost of european call option = ",binomialModelMarkov(S0,0))
	print("\nComputation time = ",time.time()-init,"seconds")
	