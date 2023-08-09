from matplotlib  import pyplot as plt
import numpy as np
import math

def CIR(beta,mu,sig,r0,T,t=0):
	gamma = math.sqrt(beta*beta + 2*sig*sig)
	B = 2*(math.exp(gamma*(T-t))-1)/(((gamma+beta)*(math.exp(gamma*(T-t))-1)) + 2*gamma)
	A = (2*gamma*math.exp((beta+gamma)*0.5*(T-t))/(((gamma+beta)*(math.exp(gamma*(T-t))-1)) + 2*gamma))**(2*beta*mu/(sig*sig))
	return ((B*r0) - math.log(A))/(T-t)

yields = []
t = np.linspace(0.1,10,num=10,endpoint='False')
for T in  t:
	yields.append(CIR(0.02,0.7,0.02,0.1,T))
plt.plot(t,yields,marker='o')
plt.title("Term Structure for set 1")
plt.xlabel("Time to maturity")
plt.ylabel("Yield")
plt.show()

yields = []
for T in  t:
	yields.append(CIR(0.7,0.1,0.3,0.2,T))
plt.plot(t,yields,marker='o')
plt.title("Term Structure for set 2")
plt.xlabel("Time to maturity")
plt.ylabel("Yield")
plt.show()

yields = []
for T in  t:
	yields.append(CIR(0.06,0.09,0.5,0.02,T))
plt.plot(t,yields,marker='o')
plt.title("Term Structure for set 3")
plt.xlabel("Time to maturity")
plt.ylabel("Yield")
plt.show()

r = 0.1
leg = []
for i in range(10):
	yields = []
	t = np.linspace(0.1,600,num=600,endpoint='False')
	for T in  t:
		yields.append(CIR(0.02,0.7,0.02,r,T))
	plt.plot(t,yields)
	leg.append("r={r}".format(r=round(r,1)))
	r += 0.1
plt.title("Yield vs Maturity for set 1")
plt.xlabel("Time to maturity")
plt.ylabel("Yield")
plt.legend(leg)
plt.show()

