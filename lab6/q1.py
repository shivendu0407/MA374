import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def plotData(df,interval):
	companies = df.columns
	df1 = pd.DataFrame()
	if(interval=='Monthly'):
		df1 = df.groupby(pd.DatetimeIndex(df['Date']).to_period('M')).nth(0)
	elif interval=='Weekly':
		df1 = df.groupby(pd.DatetimeIndex(df['Date']).to_period('W')).nth(0)
	else:
		df1 = df
	df1.fillna(method='ffill',inplace=True)
	x = df1['Date']
	y = df1.iloc[:,1]
	plt.plot(x,y)
	if(df.columns[1]=="^NSEI"):
		plt.title("{t} PLot for Nifty30".format(t = interval))
	else:
		plt.title("{t} Plot for BSE".format(t = interval))
	plt.xticks(np.arange(0, len(x), int(len(x)/4)), df1['Date'][0:len(x):int(len(x)/4)])
	plt.show()
	for i in range(4):
		for j in range(5):
			y = df1.iloc[:,5*i+j+2]
			plt.plot(x,y)
		plt.title("{t} plot for stocks".format(t=interval))
		plt.legend(companies[5*i+2:5*i+7])
		plt.xticks(np.arange(0, len(x), int(len(x)/4)), df1['Date'][0:len(x):int(len(x)/4)])
		plt.show()
	
for filename in ["bsedata1.csv","nsedata1.csv"]:
	df = pd.read_csv(filename)
	for interval in ['Daily','Monthly','Weekly']:
		plotData(df,interval)