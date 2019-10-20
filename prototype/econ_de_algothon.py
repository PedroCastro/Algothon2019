import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', -1)  # or 199
pd.set_option('display.max_columns', None)  # or 1000

# futures = pd.read_csv('data/algothon/DY_FUA_88508ff2d98f32716f2eae24114c73df.csv') ## futures
risk = pd.read_csv('data/algothon/DNB_SIRA_9ead3e6c6b4f0aaa2e4deb6e4a6ffd2f.csv') ## risk

print(risk.head(5))

risk = risk[['ticker','datepll', 'crating', 'orating', 'history', 'cond_ind', 'finance',
             'moved', 'sales', 'hicdtavg', 'pexp_s_n', 'pexp_30', 'pexp_60',
             'pexp_90', 'pexp_180', 'bnkrpt', 'dbt_ind',
             'uccfilng', 'cscore', 'cpct', 'fpct', 'paynorm', 'pubpvt','pex_sn1', 'bd_ind']]


feats = ['crating', 'orating', 'history', 'cond_ind', 'finance',
             'moved', 'sales', 'hicdtavg', 'pexp_s_n', 'pexp_30', 'pexp_60',
             'pexp_90', 'pexp_180', 'bnkrpt', 'dbt_ind',
             'uccfilng', 'cscore', 'cpct', 'fpct', 'paynorm', 'pubpvt','pex_sn1', 'bd_ind']

risk.index = pd.to_datetime(risk['datepll'], format='%Y%m')
li = pd.read_csv('data/algothon/overlapping_companies.txt')

overlapping_tickers = li.columns.tolist()

risk = risk[risk['ticker'].isin(overlapping_tickers)]

risk.to_csv('data/algothon/risk_data_cleaned.csv')

print(len(risk.ticker.unique().tolist()))

#pivoted_risk = risk.pivot_table(values=feats, index=['datepll'],columns=['ticker'])
#pivoted_risk.columns = pivoted_risk.columns.to_series().str.join('_')

#print(pivoted_risk)
#pivoted_risk.index = pd.to_datetime(pivoted_risk['datepll'])
# pivoted_risk= pivoted_risk.set_index(['datepll'])
# print(pivoted_risk.head(5))

# print(risk.head(5))
#china_eod = china_eod[['ticker', 'adj_close', 'date']]

#china_eod.to_pickle("data/algothon/china_eod.pkl")

#us_eod = pd.read_csv("data/algothon/EOD_20191019.csv", names=['ticker','Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividend', 'Split'
#                                                              ,'Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume'])

#us_eod = us_eod[['ticker','Date', 'Close']]

#us_eod.to_pickle("data/algothon/us_eod.pkl")


# chinese_stocks = pd.read_csv("data/algothon/DY_SPA_f011e0f8042c364efd40ed706cf4012c.csv")

#print(us_eod.columns)
'''

china_eod = pd.read_pickle('data/algothon/china_eod.pkl')

pivoted_china_eod = pd.pivot_table(china_eod, values='adj_close', index=['date'],columns=['ticker'])
pivoted_china_eod.index = pd.to_datetime(pivoted_china_eod.index)
pivoted_china_eod = pivoted_china_eod.sort_index().fillna(method='ffill')
# print(pivoted_us_eod.head(5))
pivoted_china_eod = pivoted_china_eod['2010-01-01':'2019-10-18']

pivoted_china_eod = pivoted_china_eod.dropna(axis=1)

pivoted_china_eod.to_csv("data/algothon/pivoted_cleaned_china_eod.csv")
print(pivoted_china_eod.head(5))
'''