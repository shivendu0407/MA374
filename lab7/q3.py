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

def plotFig():
	s1 = np.linspace(0.001,2,50)
	t1 = np.linspace(0,1,50)
	S,T = np.meshgrid(s1,t1)
	calls = np.empty(np.shape(S))
	puts = np.empty(np.shape(S))
	for i in range(np.shape(S)[0]):
		for j in range(np.shape(S)[1]):
			t = bsmOption(T[i,j],S[i,j],1,0.6,0.05,1)
			calls[i,j] = t[0]
			puts[i,j] = t[1]
	ax = plt.axes(projection="3d")
	ax.plot_surface(T,S,puts)
	ax.set_xlabel("t")
	ax.set_ylabel("s")
	ax.set_zlabel("P(t,s)")
	ax.set_title("P(t,s) vs t and s")
	plt.show()
	ax = plt.axes(projection="3d")
	ax.plot_surface(T,S,calls)
	ax.set_xlabel("t")
	ax.set_ylabel("s")
	ax.set_zlabel("C(t,s)")
	ax.set_title("C(t,s) vs t and s")
	plt.show()

plotFig()
