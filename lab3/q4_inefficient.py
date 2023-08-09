import math
import time


S0 = 100
T = 1
M = 50
r = 0.08
K = 100
sigma = 0.2
delta = T/M
u = math.exp((sigma*math.sqrt(delta))+delta*(r-(sigma*sigma)/2))
d = math.exp((-sigma*math.sqrt(delta))+delta*(r-(sigma*sigma)/2))
p = (math.exp(r*delta)-d)/(u-d)
q = (u-math.exp(r*delta))/(u-d)

def binomialModel(S,step):
	val = 0
	if(step==M):
		val = max(S-K,0)
	else:
		up = binomialModel(S*u,step+1)
		down = binomialModel(S*d,step+1)
		val = math.exp(-r*delta)*(p*up + q*down)
	return val;


for m in [5,10,25]:
	print("\n\nFor  m = ",m)
	M = m
	delta = T/M
	u = math.exp((sigma*math.sqrt(delta))+delta*(r-(sigma*sigma)/2))
	d = math.exp((-sigma*math.sqrt(delta))+delta*(r-(sigma*sigma)/2))
	p = (math.exp(r*delta)-d)/(u-d)
	q = (u-math.exp(r*delta))/(u-d)
	init = time.time()
	print("Cost of european call option = ",binomialModel(S0,0))
	print("\nComputation time = ",time.time()-init,"seconds")
	