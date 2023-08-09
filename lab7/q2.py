from matplotlib import pyplot as plt
import numpy as np
import math
from scipy.stats import norm


def bsmOption(t,s,k,sig,r,T):
	if(T==t):
		return [max(0,s-k),max(0,k-s)]
	d1 = (1/(sig*((T-t)**0.5)))*(math.log(s/k) + ((r + (sig*sig/2))*(T-t)))
	d2 = d1 - sig*((T-t)**0.5)
	c = (s*norm.cdf(d1) - (k*math.exp(-r*(T-t))*norm.cdf(d2)))
	p = (c - s + (k*math.exp(-r*(T-t))))
	return [c,p]

f1 = []
f2 = []

times = [0,0.2,0.4,0.6,0.8,1]

for t in [0,0.2,0.4,0.6,0.8,1]:
	callPrices = []
	putPrices = []
	for s in np.linspace(0.001,2,5000):
		t1 = bsmOption(t,s,1,0.6,0.05,1)
		callPrices.append(t1[0])
		putPrices.append(t1[1])
		#print(s,t)
	f1.append(callPrices)
	f2.append(putPrices)

S = np.linspace(0.001,2,5000)
plt.style.use("seaborn")

for i in range(6):
	plt.plot(S,f1[i])
plt.style.use("seaborn")	
plt.title("Variation of C(t,s) with s")
plt.xlabel("s")
plt.ylabel("C(t,s)")
plt.legend(["t = 0","t = 0.2","t = 0.4","t = 0.6","t = 0.8","t = 1"])
plt.show()

for i in range(6):
	plt.plot(S,f2[i])
plt.style.use("seaborn")
plt.title("Variation of P(t,s) with s")
plt.xlabel("s")
plt.ylabel("P(t,s)")
plt.legend(["t = 0","t = 0.2","t = 0.4","t = 0.6","t = 0.8","t = 1"])
plt.show()

ax = plt.axes(projection='3d')
for i in range(6):
	ax.plot3D([times[i] for j in range(5000)],S,f1[i])
ax.set_xlabel("t")
ax.set_ylabel("s")
ax.set_zlabel("C(t,s)")
ax.set_title("C(t,s) vs t and s")
plt.show()

ax = plt.axes(projection='3d')
for i in range(6):
	ax.plot3D([times[i] for j in range(5000)],S,f2[i])
ax.set_xlabel("t")
ax.set_ylabel("s")
ax.set_zlabel("P(t,s)")
ax.set_title("P(t,s) vs t and s")
plt.show()