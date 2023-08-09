import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import math

companies = []

font = {'size'   : 6}

matplotlib.rc('font', **font)

def generatePath(mu,var,n,start):
	time = np.linspace(1,n+1,n+1)
	plt.title("For mu = {mu},var = {var}".format(mu=mu,var = var))
	sample = [start]
	for j in range(n):
		start = start*math.exp(((mu-var/2)*(time[j+1]-time[j]))+(np.random.normal()*((time[j+1]-time[j])**0.5)*(var**0.5)))
		sample.append(start)
	return np.array(sample)

def util(arr):
	temp = []
	for i in range(len(arr)-1):
		temp.append(math.log(arr[i+1]/arr[i]))
	temp = [0 if x != x else x for x in temp]
	n = len(temp)
	sig2 = (n)*(np.var(temp))/(n-1)
	mu = (sig2/2) + np.mean(temp)
	return [mu,sig2]

def getParams(df):
	params = []
	for i in range(21):
		params.append(util(np.array(df.iloc[:,i+1])))
	return params;
	
		

def predictStockPrice(df,interval):
	n = df.shape[0]
	sz = 0
	s = df.loc[ df['Date'] <= '2021-12-31']
	sz = n - s.shape[0]
	params = getParams(s)
	x = np.arange(0,n)
	y = df.iloc[:,1]
	plt.plot(x,y,alpha=0.8)
	int
	if(df.columns[1]=="^NSEI"):
		plt.title("{t} PLot for Nifty30".format(t = interval))
	else:
		plt.title("{t} Plot for BSE".format(t = interval))
	plt.plot(np.arange(s.shape[0]-1,s.shape[0]+sz),generatePath(params[0][0],params[0][1],sz,df.iloc[s.shape[0]-1,1]),color="red",alpha=0.5)
	if(df.columns[0]=="^NSEI"):
		plt.title("Predicted {t} PLot for Nifty30".format(t = interval))
	else:
		plt.title("Predicted {t} Plot for BSE".format(t = interval))
	plt.xlabel("Time Points")
	plt.ylabel("Value")
	plt.legend(["Actual Price","Predicted Price"])
	plt.show()
	for i in range(5):
		fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
		axes = [ax1,ax2,ax3,ax4]
		for j in [2,3,4,5]:
			axes[j-2].plot(x,df.iloc[:,4*i+j],alpha=0.8)
			axes[j-2].plot(np.arange(s.shape[0]-1,s.shape[0]+sz),generatePath(params[4*i+j-1][0],params[4*i+j-1][1],sz,df.iloc[s.shape[0]-1,4*i+j]),color="red",alpha=0.5)
			axes[j-2].set_xlabel("Time Points")
			axes[j-2].set_ylabel("Value")
			axes[j-2].set_title("Predicted {r} stock prices for {d}".format(d= companies[4*i+j-1],r=interval))
			axes[j-2].legend(["Actual Price","Predicted Price"])
		plt.show()
	
for filename in ["bsedata1.csv","nsedata1.csv"]:
	for interval in ["weekly","monthly"]:
		df = pd.read_csv(filename)
		companies = df.columns[1:]
		if(interval=='monthly'):
			df = df.groupby(pd.DatetimeIndex(df['Date']).to_period('M')).nth(0)
		elif interval=='weekly':
			df = df.groupby(pd.DatetimeIndex(df['Date']).to_period('W')).nth(0)
		df.fillna(method='ffill',inplace=True)
		df.fillna(method='bfill',inplace=True)
		predictStockPrice(df,interval)