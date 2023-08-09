import time
import datetime
import pandas as pd


tickers = ["AAPL","AMZN","META","GOOG","IBM","INTC","MSFT","NFLX","NKE","TSLA"]
interval = '1mo'
period1 = int(time.mktime(datetime.datetime(2018, 1, 1, 23, 59).timetuple()))
period2 = int(time.mktime(datetime.datetime(2023, 1, 1, 23, 59).timetuple()))

myFrame  = pd.DataFrame()
for ticker in tickers:
    print(ticker)
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)
    if(ticker=="AAPL"):
        myFrame["date"] = df["Date"]
    myFrame[ticker] = df["Close"]
myFrame.to_csv("stocks.csv")
