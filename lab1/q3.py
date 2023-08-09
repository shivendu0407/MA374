import math

def createOutput(N,priceCall,pricePut):
	heads = N
	tails = 0
	f = open("Price_{x}.csv".format(x = N),"w")
	f.write("Up Steps,Down Steps,Call Price,Put Price\n")
	for i in range(N+1):
		f.write("{a},{b},{c},{d}\n".format(a = heads,b = tails,c=priceCall[i],d=pricePut[i]))
		heads-=1
		tails+=1
	f.close()
	print("Price_{x}.csv created containing table for time {y}".format(x=N,y=0.25*N))
def solve(M = 20):
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
		if M-i-1 in [0,2,4,6,12,18]:
			print("Call option prices at t = ",0.25*(M-i-1),priceCall[:M-i])
			print("Put option prices at t = ",0.25*(M-i-1),pricePut[:M-i])
			createOutput(M-i-1,priceCall[:M-i],pricePut[:M-i])
			print("\n")
solve()