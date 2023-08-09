import time
import datetime
import pandas as pd


tickers = ["^BSESN","INFY.BO","RELIANCE.BO","HDFCBANK.BO","AXISBANK.BO","BHARTIARTL.BO","ICICIBANK.BO","VEDL.BO","CIPLA.BO","BAJAJ-AUTO.BO","NESTLEIND.BO","TITAN.BO","IDBI.BO","TVSMOTOR.BO","BOSCHLTD.BO","GAIL.BO","YESBANK.BO","VOLTAS.BO","MARICO.BO","UNIONBANK.BO","BERGEPAINT.BO"]
interval = '1d'
period1 = int(time.mktime(datetime.datetime(2018, 1, 1, 23, 59).timetuple()))
period2 = int(time.mktime(datetime.datetime(2023, 1, 1, 23, 59).timetuple()))

myFrame  = pd.DataFrame()
for ticker in tickers:
    print(ticker)
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)
    if(ticker=="^BSESN"):
        myFrame["date"] = df["Date"]
    myFrame[ticker] = df["Open"]
myFrame.to_csv("bse.csv")