from matplotlib import pyplot as plt
plt.style.use("dark_background")
import numpy as np
import math

def vasicek(beta,mu,T,r,sig):
	if(T==0):
		return 0
	B = (1/beta)*(1-math.exp(-beta*T))
	A = ((B-T)*(beta*beta*mu-0.5*sig*sig)/(beta*beta)) - (B*B*sig*sig/(4*beta))
	return (1/T)*(-A + r*B)

yields = []
X = np.arange(1,11)
for T in np.arange(1,11):
	yields.append(vasicek(5.9,0.2,T,0.1,0.3))
plt.plot(X,yields,marker='x')
plt.show()

yields = []
for T in np.arange(1,11):
	yields.append(vasicek(3.9,0.1,T,0.2,0.3))
plt.plot(X,yields,marker='x')
plt.show()

yields = []
for T in np.arange(1,11):
	yields.append(vasicek(0.1,0.4,T,0.1,0.11))
plt.plot(X,yields,marker='x')
plt.show()


for r in np.linspace(0.1,1,10):
	yields = []
	for T in range(10,511):
		yields.append(vasicek(5.9,0.2,T,r,0.3))
	plt.plot(range(10,511),yields)
plt.show()

for r in np.linspace(0.1,1,10):
	yields = []
	for T in range(10,511):
		yields.append(vasicek(3.9,0.1,T,r,0.3))
	plt.plot(range(10,511),yields)
plt.show()

for r in np.linspace(0.1,1,10):
	yields = []
	for T in range(10,511):
		yields.append(vasicek(0.1,0.4,T,r,0.11))
	plt.plot(range(10,511),yields)
plt.show()