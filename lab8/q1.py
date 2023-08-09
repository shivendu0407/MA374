import pandas as pd
import numpy as np
import math
from scipy.stats import norm
from matplotlib import pyplot as plt
import matplotlib

plt.style.use('seaborn')
params = {'axes.labelsize': 12,'axes.titlesize':12,  'legend.fontsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12}

matplotlib.rcParams.update(params)
R = 0.05
curr = 1

def getHistVolatility(filename,timePeriod):
	df = pd.read_csv(filename)
	companies = (df.columns)[1:]
	final = pd.DataFrame()
	dfMonthly = df.groupby(pd.DatetimeIndex(df.Date).to_period('M')).nth(0)[60-timePeriod:]
	dfMonthly.reset_index(inplace = True, drop = True)
	idx_list = df.index[df['Date'] >= dfMonthly.iloc[0]['Date']].tolist()
	df1 = df.iloc[idx_list[0]:]
	df1 = df1.loc[:,companies]
	df1.fillna(method='ffill',inplace=True)
	vols = []
	for company in companies:
		A = []
		t = np.array(df1[company])
		for i in range(len(t)-1):
			A.append(math.log(t[i+1]/t[i]))
		vols.append(np.nanstd(A)*math.sqrt(252))
	final["Stock"] = companies
	final["Historical Volatility"] = vols
	if(timePeriod==1):
		print(final)
	return vols


def bsmOption(t,s,k,sig,r,T):
	d1 = (1/(sig*((T-t)**0.5)))*(math.log(s/k) + ((r + (sig*sig/2))*(T-t)))
	d2 = d1 - sig*((T-t)**0.5)
	c = (s*norm.cdf(d1) - (k*math.exp(-r*(T-t))*norm.cdf(d2)))
	p = (c - s + (k*math.exp(-r*(T-t))))
	return [c,p]

for filename in ["bsedata1.csv","nsedata1.csv"]:
	print("\n\nFor {f}\n\n".format(f = filename))
	callPrices = []
	putPrices = []
	volatilities = []
	callOption = []
	putOption = []
	volatility = []
	for i in range(60):
		callPrices.append(0)
		putPrices.append(0)
		volatilities.append(0)
	df = pd.read_csv(filename)
	companies = (df.columns)[1:]
	for i in range(len(companies)):
		callOption.append([])
		putOption.append([])
		volatility.append([])
	for i in range(len(companies)):
		for j in range(11):
			callOption[i].append([])
			putOption[i].append([])
	for t in range(1,61):
		vols = getHistVolatility(filename,t)
		for i in range(len(companies)):
			callPrices = []
			putPrices = []
			current = 0
			volatility[i].append(vols[i])
			for a in np.arange(0.5,1.6,0.1):
				sig = vols[i]
				s = df.iloc[-1,i+1]
				k = a*s
				x = bsmOption(0,s,k,sig,R,0.5)
				if(t==1):
					callPrices.append(x[0])
					putPrices.append(x[1])
				callOption[i][current].append(x[0])
				putOption[i][current].append(x[1])
				current+=1
			if(t==1):
				print("For {r}".format(r = companies[i]))
				df1 = pd.DataFrame()
				df1["A"] = np.arange(0.5,1.6,0.1)
				df1["Call Prices"] = callPrices
				df1["Put Prices"] = putPrices
				print(df1)
	plt.plot(np.arange(1,61),volatility[0])
	plt.xlabel("Length of period")
	plt.ylabel("Volatility")
	plt.title("Volatility vs length of period considered for {x}".format(x=companies[0]))
	plt.rcParams.update({'font.size': 6})
	plt.savefig("imgs/{curr}".format(curr=curr),bbox_inches='tight')
	curr = curr+1
	plt.show()
	
	for j in range(5):
		fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
		axes = [ax1,ax2,ax3,ax4]
		for i in range(4):
			axes[i].plot(np.arange(1,61),volatility[4*j+i+1])
			axes[i].set_xlabel("Length of period")
			axes[i].set_ylabel("Volatility")
			axes[i].set_title("Volatility vs length of period considered for {x}".format(x=companies[4*j+i+1]))
		plt.rcParams.update({'font.size': 6})
		plt.savefig("imgs/{curr}".format(curr=curr),bbox_inches='tight')
		curr = curr+1
		plt.show()
	
	current = 0
	for A in [0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5]:
		for j in range(5):
			fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
			axes = [ax1,ax2,ax3,ax4]
			for i in range(4):
				axes[i].plot(np.arange(1,61),callOption[4*j+1+1][current])
				axes[i].set_xlabel("Length of period")
				axes[i].set_ylabel("Call Price")
				axes[i].set_title("Call Price vs length of period considered for {x} and A = {A}".format(A=A,x=companies[4*j+i+1]))
			plt.rcParams.update({'font.size': 6})
			plt.savefig("imgs/{curr}".format(curr=curr),bbox_inches='tight')
			curr+=1
			plt.show()
			fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
			axes = [ax1,ax2,ax3,ax4]
			for i in range(4):
				axes[i].plot(np.arange(1,61),putOption[4*j+1+1][current])
				axes[i].set_xlabel("Length of period")
				axes[i].set_ylabel("Put Price")
				axes[i].set_title("Put Price vs length of period considered for {x} and A = {A}".format(A=A,x=companies[4*j+i+1]))
			plt.rcParams.update({'font.size': 6})
			plt.savefig("imgs/{curr}".format(curr=curr),bbox_inches='tight')
			curr+=1
			plt.show()
		current += 1