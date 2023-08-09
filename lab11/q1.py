from matplotlib  import pyplot as plt
import numpy as np
import math

def vasicek(beta,mu,sig,r0,T,t=0):

	B = (1-math.exp(-beta*(T-t)))/beta
	A = math.exp((((B-T+t)*(beta*beta*mu - (sig*sig/2)))/(beta**2)) - ((sig*sig*B*B)/(4*beta)))
	return ((B*r0) - math.log(A))/(T-t)

yields = []
t = np.linspace(0.1,10,num=10,endpoint='False')
for T in  t:
	yields.append(vasicek(5.9,0.2,0.3,0.1,T))
plt.plot(t,yields,marker='o')
plt.title("Term Structure for set 1")
plt.xlabel("Time to maturity")
plt.ylabel("Yield")
plt.show()

yields = []
for T in  t:
	yields.append(vasicek(3.9,0.1,0.3,0.2,T))
plt.plot(t,yields,marker='o')
plt.title("Term Structure for set 2")
plt.xlabel("Time to maturity")
plt.ylabel("Yield")
plt.show()

yields = []
for T in  t:
	yields.append(vasicek(0.1,0.4,0.11,0.1,T))
plt.plot(t,yields,marker='o')
plt.title("Term Structure for set 3")
plt.xlabel("Time to maturity")
plt.ylabel("Yield")
plt.show()

leg = []
r = 0.1
for i in range(10):
	yields = []
	t = np.linspace(10,500,num=500,endpoint='False')
	for T in  t:
		yields.append(vasicek(5.9,0.2,0.3,r,T))
	plt.plot(t,yields)
	leg.append("r={r}".format(r=round(r,1)))
	r += 0.1
plt.title("Yield vs Maturity for set 1")
plt.xlabel("Time to maturity")
plt.ylabel("Yield")
plt.legend(leg)
plt.show()

leg = []
r = 0.1
for i in range(10):
	yields = []
	t = np.linspace(10,500,num=500,endpoint='False')
	for T in  t:
		yields.append(vasicek(3.9,0.1,0.3,r,T))
	plt.plot(t,yields)
	leg.append("r={r}".format(r=round(r,1)))
	r += 0.1
plt.title("Yield vs Maturity for set 2")
plt.xlabel("Time to maturity")
plt.ylabel("Yield")
plt.legend(leg)
plt.show()

leg = []
r = 0.1
for i in range(10):
	yields = []
	t = np.linspace(10,500,num=500,endpoint='False')
	for T in  t:
		yields.append(vasicek(0.1,0.4,0.11,r,T))
	plt.plot(t,yields)
	leg.append("r={r}".format(r=round(r,1)))
	r += 0.1
plt.title("Yield vs Maturity for set 3")
plt.xlabel("Time to maturity")
plt.ylabel("Yield")
plt.legend(leg)
plt.show()
