import pandas as pd
import numpy as np
import swifter
from functools import reduce
from tools import featGen
pd.set_option('display.max_colwidth', -1)  # or 199
pd.set_option('display.max_columns', None)  # or 1000

print(pd.__version__)

li = pd.read_csv('data/algothon/overlapping_companies.txt')

overlapping_tickers = li.columns.tolist()

us_eod = pd.read_csv('data/algothon/unstack_us_eod.csv')
us_eod = us_eod[us_eod['ticker'].isin(overlapping_tickers)]

us_eod.to_pickle('data/algothon/filtered_us_eod.pkl')

print(us_eod.head(5))