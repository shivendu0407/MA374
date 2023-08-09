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
df.to_csv("abc.csv")
