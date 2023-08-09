import yfinance as yf
import pandas as pd
import numpy as np

stock_list1 = ["^NSEI","UPL.NS","ICICIBANK.NS","WIPRO.NS","TECHM.NS","ASIANPAINT.NS","EICHERMOT.NS","NTPC.NS","LT.NS","COALINDIA.NS","GRASIM.NS","BHEL.NS","DABUR.NS","COLPAL.NS","MRF.NS","GAIL.NS","HAVELLS.NS","POONAWALLA.NS","BOSCHLTD.NS","UNIONBANK.NS","GODREJCP.NS"]
stock_list2 = ["^BSESN","INFY.BO","RELIANCE.BO","HDFCBANK.BO","AXISBANK.BO","BHARTIARTL.BO","ICICIBANK.BO","VEDL.BO","CIPLA.BO","TCS.BO","NESTLEIND.BO","TITAN.BO","IDBI.BO","TVSMOTOR.BO","BOSCHLTD.BO","GAIL.BO","YESBANK.BO","VOLTAS.BO","MARICO.BO","UNIONBANK.BO","NAUKRI.BO"]

p = yf.download(stock_list1, start="2018-01-01", end="2022-12-31", interval = "1d")['Open']
sorted_columns = [STOCK for STOCK in stock_list1 if STOCK in p.columns]
p = p.loc[:,sorted_columns]
p.to_csv("nsedata1.csv")

p = yf.download(stock_list2, start="2018-01-01", end="2022-12-31", interval = "1d")['Open']
sorted_columns = [STOCK for STOCK in stock_list2 if STOCK in p.columns]
p = p.loc[:,sorted_columns]
p.to_csv("bsedata1.csv")