import time
import datetime
import pandas as pd
import numpy as np

interval = '1d'
period1 = int(time.mktime(datetime.datetime(2018, 1, 1, 23, 59).timetuple()))
period2 = int(time.mktime(datetime.datetime(2023, 1, 1, 23, 59).timetuple()))

query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{"^BSESN"}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
df = pd.read_csv(query_string)
x = (np.arradf["Close"]-df["Open"])/df["Open"]
x.fillna(0,inplace=True)
print(df)
print(np.mean(np.array(x)))