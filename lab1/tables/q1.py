import math
def binomialModel(M):
	print("\nFor M = ",M)
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
	print("d = ",d,",u = ",u,",e^(rt) = ",math.exp(r*delta))
	if(0<d and d<math.exp(-r*delta) and u>math.exp(r*delta)):
		print("No arbitrarge condition satisfied")
	else:
		print("No arbitrarge condition not satisfied")
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
	print("Price of Call option = "  ,priceCall[0])
	print("Price of Put option = "  ,pricePut[0])
for M in [1,5,10,20,50,100,200,400]:
	binomialModel(M)