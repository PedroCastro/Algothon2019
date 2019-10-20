import pandas as pd
import numpy as np
import swifter
from functools import reduce
from tools import featGen
pd.set_option('display.max_colwidth', -1)  # or 199
pd.set_option('display.max_columns', None)  # or 1000

print(pd.__version__)
'''
us_eod = pd.read_csv("data/algothon/EOD_20191019.csv", names=['ticker','Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividend', 'Split'
                                                              ,'Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume'])

us_eod = us_eod[['ticker','Date', 'Adj_Close']]

us_eod.to_pickle("data/algothon/us_eod.pkl")


# chinese_stocks = pd.read_csv("data/algothon/DY_SPA_f011e0f8042c364efd40ed706cf4012c.csv")

#print(us_eod.columns)


us_eod = pd.read_pickle('data/algothon/us_eod.pkl')

pivoted_us_eod = pd.pivot_table(us_eod, values='Adj_Close', index=['Date'],columns=['ticker'])
pivoted_us_eod.index = pd.to_datetime(pivoted_us_eod.index)
pivoted_us_eod = pivoted_us_eod.sort_index().fillna(method='ffill')

# print(pivoted_us_eod.head(5))
pivoted_us_eod = pivoted_us_eod['2009-01-01':'2019-10-18']
pivoted_us_eod = pivoted_us_eod.dropna(axis=1)

pivoted_us_eod.to_csv('data/algothon/pivoted_cleaned_us_eod.csv')

'''

#pivoted_us_eod = pd.read_pickle('data/algothon/pivoted_cleaned_us_eod.pkl')
pivoted_us_eod = pd.read_csv('data/algothon/pivoted_cleaned_us_eod.csv')

pivoted_us_eod.index = pd.to_datetime(pivoted_us_eod['Date'])
pivoted_us_eod = pivoted_us_eod.drop("Date", axis=1)
steps = 30

mom2_pivoted_us_eod = pivoted_us_eod.apply(featGen.momentum2, axis=0, args=(steps,)).fillna(method='ffill')
MACD_pivoted_us_eod = pivoted_us_eod.apply(featGen.MACD, axis=0).fillna(method='ffill')
vol_pivoted_us_eod = pivoted_us_eod.apply(featGen.retvol, axis=0, args=(steps,)).fillna(method='ffill')


unstack_adj_close = pivoted_us_eod.unstack().reset_index(name='adj_close')

unstack_mom2 = mom2_pivoted_us_eod.unstack().reset_index(name='mom')
unstack_MACD = MACD_pivoted_us_eod.unstack().reset_index(name='MACD')

unstack_vol = vol_pivoted_us_eod.unstack().reset_index(name='vol')



unstack_df = reduce(lambda X, x: pd.merge(X, x,  how='left', left_on=['level_0','Date'], right_on = ['level_0','Date'])
                    ,[unstack_adj_close, unstack_mom2, unstack_MACD, unstack_vol])

unstack_df.columns = ['ticker', 'Date', 'adj_close', 'mom', 'MACD', 'vol']
unstack_df.index = pd.to_datetime(unstack_df.Date)

unstack_df = unstack_df['2010-01-01':'2019-10-18']
unstack_df = unstack_df[unstack_df['ticker'].isnin() ]
# unstack_df = unstack_df.dropna(axis=1)
unstack_df.to_csv('data/algothon/unstack_us_eod.csv')
print(unstack_df.head(10))


#unstack_mom2.index = pd.to_datetime()
#unstack_mom2.rename(columns={'level_0': 'month', 'level_1': 'year'}, inplace=True)
# print(unstack_mom2.head(5)





'''
for i, ticker in enumerate(pivoted_us_eod.columns.tolist()):
    adj_close = pivoted_us_eod[ticker]
    adj_close.name = 'adj_close'
    col_list = [adj_close, mom2_pivoted_us_eod[ticker],
                MACD_pivoted_us_eod[ticker],
                vol_pivoted_us_eod[ticker]]

    temp_df = reduce(lambda X, x: pd.merge_asof(X.sort_index(), x.sort_index(),
                                            left_index=True, right_index=True, direction='forward',
                                            tolerance=pd.Timedelta('2ms')), col_list)
    temp_df['ticker'] = ticker

    unpivot_df.append(temp_df)

main_out_df = reduce(lambda X, x: X.append(x), unpivot_df)

print(main_out_df.head(5))
# pivoted_us_eod.to_pickle("data/algothon/pivoted_cleaned_us_eod.pkl")
# pivoted_us_eod.to_csv("data/algothon/pivoted_cleaned_us_eod.csv")
# print(pivoted_us_eod.head(5))
# print(pivoted_us_eod)
'''