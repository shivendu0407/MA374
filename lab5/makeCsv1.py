import time
import datetime
import pandas as pd


tickers = ["^NSEI","UPL.NS","ICICIBANK.NS","WIPRO.NS","TECHM.NS","ASIANPAINT.NS","EICHERMOT.NS","NTPC.NS","LT.NS","COALINDIA.NS","GRASIM.NS","BHEL.NS","DABUR.NS","COLPAL.NS","MRF.NS","GAIL.NS","HAVELLS.NS","POONAWALLA.NS","BOSCHLTD.NS","UNIONBANK.NS","GODREJCP.NS"]
interval = '1d'
period1 = int(time.mktime(datetime.datetime(2018, 1, 1, 23, 59).timetuple()))
period2 = int(time.mktime(datetime.datetime(2023, 1, 1, 23, 59).timetuple()))

myFrame  = pd.DataFrame()
for ticker in tickers:
    print(ticker)
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)
    if(ticker=="^NSEI"):
        myFrame["date"] = df["Date"]
    myFrame[ticker] = df["Open"]
myFrame.to_csv("nse.csv")