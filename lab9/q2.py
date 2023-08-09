from matplotlib import pyplot as plt
import numpy as np
import math
import pandas as pd
from datetime import datetime
from scipy.stats import norm
from tabulate import tabulate

def pprint_df(dframe):
    print(tabulate(dframe, headers='keys', tablefmt='psql', showindex=False))


df1 = pd.read_csv("nsedata1.csv").loc[:,["Date","^NSEI"]]
df = pd.read_csv("NIFTYoptiondata.csv")
df = pd.merge(df,df1,on="Date")
dfs = pd.read_excel("stockoptiondata.xlsx", sheet_name=None)

r = 0.05
companies = ["NSE","AsianPaint","ICICIBANK","TECHM","WIPRO","UPL"]
stocks = ["^NSEI","ASIANPAINT.NS","ICICIBANK.NS","TECHM.NS","WIPRO.NS","UPL.NS"]


def normPdf(x):
	return (1/(math.sqrt(2*math.pi)))*math.exp(-x*x/2)

def comp(s,K,tau,sig,p):
	d1 = (1/(sig*(tau**0.5)))*(math.log(s/K) + ((r+(sig*sig/2))*tau))
	d2 = d1 - (sig*math.sqrt(tau))
	num = (s*norm.cdf(d1)) - (K*math.exp(-r*tau)*norm.cdf(d2)) - p
	den = s*norm.pdf(d1)*(tau**0.5)
	return [num,den]

def newtonRaphson(s,K,tau,p):
	sig = 0.3
	for i in range(1000):
		x = comp(s,K,tau,sig,p)
		if(x[1]==0):
			return -1
		t = sig
		sig = sig - x[0]/x[1]
		#print(sig)
		if(abs(sig-t)<1e-6):
			return sig
	return sig

def getHistVolatility(df3,timePeriod,company,currentDate):
	idx_list = []
	for i in range(df3.shape[0]):
		x  = datetime.strptime(df3["Date"][i],"%d-%b-%y")
		if((currentDate-x).days>=0 and (currentDate-x).days<=timePeriod):
			idx_list.append(i)
	#print(idx_list)
	df4 = df3.iloc[idx_list,:]
	df4 = df4.loc[:,company]
	df4.fillna(method='ffill',inplace=True)
	if(df4.shape[0]<=1):
		return -1
	A = []
	t = np.array(df4)
	for i in range(len(t)-1):
		A.append(math.log(t[i+1]/t[i]))
	return (np.nanstd(A)*math.sqrt(252))


for j in range(6): 
	print("\nFor ",companies[j])
	if(j>0):
		df1 = pd.read_csv("nsedata1.csv").loc[:,["Date",stocks[j]]]
		df = dfs[companies[j]]
		df = pd.merge(df,df1,on="Date")
	callPrices = []
	putPrices = []
	dates = []
	strikes = []
	stockPrice = []
	impVhis = {}

	idxs = np.random.randint(0,df.shape[0],10000)
	
	for i in idxs:
		callPrices.append(df["Call Price"][i])
		putPrices.append(df["Put Price"][i])
		strikes.append(df["Strike Price"][i])
		stockPrice.append(df[stocks[j]][i])
		d1 = datetime.strptime(df["Date"][i],"%d-%b-%y")
		d2 = datetime.strptime(df["Maturity"][i],"%d-%b-%y")
		dates.append((d2-d1).days)
			
	ax = plt.axes(projection="3d")
	ax.scatter3D(dates,strikes,callPrices)
	ax.set_xlabel("Days to Maturity")
	ax.set_ylabel("Strike Price")
	ax.set_zlabel("Price")
	ax.set_title("Price of European Call 3d for {u}".format(u=companies[j]))
	plt.show()

	ax = plt.axes(projection="3d")
	ax.scatter3D(dates,strikes,putPrices)
	ax.set_xlabel("Days to Maturity")
	ax.set_ylabel("Strike Price")
	ax.set_zlabel("Price")
	ax.set_title("Price of European Put 3d for {u}".format(u=companies[j]))
	plt.show()

	plt.scatter(dates,callPrices)
	plt.xlabel("Days to Maturity")
	plt.ylabel("Price")
	plt.title("Price vs Days to Maturity of call option for {u}".format(u=companies[j]))
	plt.show()

	plt.scatter(strikes,callPrices)
	plt.xlabel("Strike Price")
	plt.ylabel("Price")
	plt.title("Price vs Strike of call option for {u}".format(u=companies[j]))
	plt.show()

	plt.scatter(dates,putPrices)
	plt.xlabel("Days to Maturity")
	plt.ylabel("Price")
	plt.title("Price vs Days to Maturity of put option for {u}".format(u=companies[j]))
	plt.show()

	plt.scatter(strikes,putPrices)
	plt.xlabel("Strike Price")
	plt.ylabel("Price")
	plt.title("Price vs Strike of put option for {u}".format(u=companies[j]))
	plt.show()


	callPrices = []
	putPrices = []
	dates = []
	strikes = []
	stockPrice = []
	idx = np.random.randint(0,df.shape[0],5000)
	currentDates = []

	for i in idx:
			callPrices.append(df["Call Price"][i])
			putPrices.append(df["Put Price"][i])
			strikes.append(df["Strike Price"][i])
			stockPrice.append(df[stocks[j]][i])
			d1 = datetime.strptime(df["Date"][i],"%d-%b-%y")
			d2 = datetime.strptime(df["Maturity"][i],"%d-%b-%y")
			currentDates.append(d1)
			dates.append((d2-d1).days)
			
	k1 = []
	d1 = []

	impliedVol = []
	histvol = []
	impliedVol1 = []
	times1 = []
	current = []
	for i in range(len(idx)):
		#print(i)
		if(callPrices[i]!=0 and dates[i]!=0):
			t = newtonRaphson(stockPrice[i],strikes[i],dates[i]/365,callPrices[i])
			if(math.isnan(t)==0 and t>0 and t<1.5):
				impliedVol.append(t)
				k1.append(strikes[i])
				d1.append(dates[i])
				if(np.random.uniform()<0.3 and t<1):
					t1 = getHistVolatility(df1,dates[i],stocks[j],currentDates[i])
					if(t1>-1):
						impliedVol1.append(t)
						histvol.append(t1)
						times1.append(dates[i])
						current.append(currentDates[i])

	ax = plt.axes(projection = "3d")
	ax.scatter3D(d1,k1,impliedVol)
	ax.set_xlabel("Days to maturity")
	ax.set_ylabel("Strike Price")
	ax.set_zlabel("Implied Volatility")
	ax.set_title("Implied volatility vs Maturity and Strike Price for {u}".format(u = companies[j]))
	plt.show()

	plt.scatter(d1,impliedVol,color='r')
	plt.xlabel("Days to maturity")
	plt.ylabel("Implied volatility")
	plt.title("Implied volatility vs Maturity for {u}".format(u = companies[j]))
	plt.show()

	plt.scatter(k1,impliedVol,color = 'r')
	plt.xlabel("Strike Price")
	plt.ylabel("Implied volatility")
	plt.title("Implied volatility vs strike price for {u}".format(u = companies[j]))
	plt.show()
 
	plt.scatter(histvol,impliedVol1,color = 'r')
	plt.xlabel("Historical Volatility")
	plt.ylabel("Implied Volatility")
	plt.title("Implied vs Historical Volatility for {u}".format(u = companies[j]))
	plt.axis("square")
	plt.show()
 
	ax = plt.axes(projection="3d")
	ax.scatter3D(times1,histvol,impliedVol1,color='r')
	ax.set_xlabel("Days to Maturity")
	ax.set_ylabel("Historical Volatility")
	ax.set_zlabel("Implied Volatility")
	ax.set_title("Days to Maturiy vs Implied vs Historical Volatility")
	plt.show()
 
 
	fin = pd.DataFrame()
	fin["Days to Maturity"] = times1
	fin["Implied Volatility"] = impliedVol1
	fin["Historical Volatility"] = histvol
	pprint_df(fin.iloc[np.sort(np.random.randint(0,fin.shape[0],15)),:])
	fin.to_csv("{u}.csv".format(u=companies[j]))
	