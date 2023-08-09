import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

stock_list5 = ["^NSEI","UPL.NS","ICICIBANK.NS","WIPRO.NS","TECHM.NS","ASIANPAINT.NS"]

p = yf.download(stock_list5, start="2018-03-30", end="2023-02-28", interval = "1d")['Open']
sorted_columns = [STOCK for STOCK in stock_list5 if STOCK in p.columns]
p = p.loc[:,sorted_columns]
p.to_csv("nsedata1.csv")
df = pd.read_csv("nsedata1.csv")
y = []
for i in df.index:
    x = df["Date"][i]
    y.append(datetime.strptime(x,"%Y-%m-%d").strftime("%d-%b-%y"))
df["Date"] = y
df.to_csv("nsedata1.csv")