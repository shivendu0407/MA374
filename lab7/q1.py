from matplotlib import pyplot as plt
import numpy as np
import math
from scipy.stats import norm


def bsmOption(t,s,k,sig,r,T):
	d1 = (1/(sig*((T-t)**0.5)))*(math.log(s/k) + ((r + (sig*sig/2))*(T-t)))
	d2 = d1 - sig*((T-t)**0.5)
	c = (s*norm.cdf(d1) - (k*math.exp(-r*(T-t))*norm.cdf(d2)))
	p = (c - s + (k*math.exp(-r*(T-t))))
	return [c,p]

print(bsmOption(0,1,1,0.6,0.05,1))