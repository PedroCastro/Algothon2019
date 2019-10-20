import pandas as pd
import numpy as np

df = pd.read_pickle('../assets/market_info_supplychain.pkl')

def mrm_c(std,vol):
	value=np.tanh((10/vol)*50*std)
	value[value<-0.8] = -1
	value[value>0.8] = 1
	value[(value>=-0.8)&(value<=0.8)] = 0
	return value
print(df.shape)
df['ReturnClassifier'] = mrm_c(df['fwd_return'],df['vol'])

risk = pd.read_csv('../assets/risk_data_cleaned.csv')
risk.index = risk['datepll']
risk.index = pd.to_datetime(risk.index)
risk.index =risk.index.rename("Date")
df = pd.merge(df, risk, how="left", on=['Date','ticker'])

df = df.groupby("ticker").apply(lambda x: x.fillna(method="ffill"))

print(df.shape)
df.to_pickle("../assets/final_market_risk_supplychain.pkl")
