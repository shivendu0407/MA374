import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import math
from scipy.special import ndtri


def normPdf(x):
	return math.exp(-x*x/2)/math.sqrt(2*math.pi)

companies = []


font = {'size'   : 6}

matplotlib.rc('font', **font)

def plotHist(df,interval,logReturn):
	dat = np.transpose(np.array(df))
	n = "";
	if(logReturn):
		n = "log"
	if(logReturn):
		df = np.log(dat+1)
	s = np.mean(dat[0])
	v = (np.var(dat[0]))**0.5
	plt.hist([(x-s)/v for x in dat[0]],bins = 30,density=True,edgecolor='r',color='y')
	t = np.linspace(-3,3,5000)
	plt.plot(t,[normPdf(x) for x in t])
	if(companies[1]=="^NSEI"):
		plt.title("Histogram for Nifty30 using {s} {r} return".format(r = interval,s=n))
	else:
		plt.title("Histogram for Sensex using {s} {r} return".format(r = interval,s=n))
	plt.show()
	if(companies[1]=="^NSEI"):
		plt.title("Box Plot for Nifty30 using {s} {r} return".format(r = interval,s=n))
	else:
		plt.title("Box Plot for Sensex using {s} {r} return".format(r = interval,s=n))
	plt.boxplot(dat[0])
	plt.show()
	
	for i in range(5):
		fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
		axes = [ax1,ax2,ax3,ax4]
		for j in [1,2,3,4]:
			axes[j-1].plot(t,[normPdf(x) for x in t])
			s = np.mean(dat[4*i+j])
			v = np.var(dat[4*i + j])**0.5
			axes[j-1].hist((dat[4*i+j]-s)/v,bins = 30,density=True,edgecolor='r',color='y')
			axes[j-1].set_title("Histogram for {d} using {s} {r} return".format(d= companies[4*i+j+1],s=n,r=interval))
		plt.show()
		fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
		axes = [ax1,ax2,ax3,ax4]
		for j in [1,2,3,4]:
			axes[j-1].boxplot(dat[4*i+j])
			axes[j-1].set_title("BoxPlot for {d} using {s} {r} return".format(d= companies[4*i+j+1],s=n,r=interval))
		plt.show()
  
def qqplot(df,interval):
	for j in range(5):
		fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
		axes = [ax1,ax2,ax3,ax4]
		for i in range(4):
			df1 = pd.DataFrame()
			x1 = np.array(df.loc[:,companies[4*j+i+2]])
			x1 = np.sort(x1)
			y1 = (np.arange(1,1+len(x1)))/len(x1)
			m = np.mean(x1)
			s = np.var(x1)**0.5
			x1 = (x1-m)/s
			y1 = ndtri(y1)
			axes[i].scatter(y1,x1)
			axes[i].plot([-3,-2,-1,0,1,2,3],[-3,-2,-1,0,1,2,3], color='red')
			axes[i].set_title("Quantile Quantile Plot for {x} using {t} data".format(x = companies[4*j+i+2],t = interval))
			axes[i].set_xlabel("Theoretical Quantiles")
			axes[i].set_ylabel("Actual Quantiles")
		plt.show()
		
for b in [0]:
	for filename in ["bsedata1.csv","nsedata1.csv"]:
		for interval in ["Daily","Weekly","Monthly"]:
			df = pd.read_csv(filename)
			companies = df.columns
			if(interval=='Monthly'):
				df = df.groupby(pd.DatetimeIndex(df['Date']).to_period('M')).nth(0)
				df = df.set_index('Date')
			elif interval=='Weekly':
				df = df.groupby(pd.DatetimeIndex(df['Date']).to_period('W')).nth(0)
				df = df.set_index('Date')
			df = (df.loc[:,companies[1:]].pct_change()).iloc[1:,:]
			#plotHist(df,interval,b)
			qqplot(df,interval)