import numpy as np
import tqdm
import pandas as pd

supplychain = pd.read_csv("../assets/global_supplychain.csv")
supplychain["accounting_as_of_date"] = pd.to_datetime(supplychain.accounting_as_of_date)

# supplychain["accounting_as_of_date"] = supplychain["accounting_as_of_date"].dt.strftime('%m/%Y')
# supplychain.drop_duplicates(subset=["accounting_as_of_date", "supplier_ticker", "customer_ticker"], keep='first', inplace=True)

stock_market = pd.read_pickle("../assets/filtered_us_eod.pkl")
stock_market = stock_market.drop(["Date.1"], axis=1)
stock_market["Date"] = pd.to_datetime(stock_market.Date)

companies = list(set(stock_market["ticker"].values.tolist())) # get rid of date
customers = supplychain["customer_ticker"].values.tolist()
suppliers = supplychain["supplier_ticker"].values.tolist()


customer_suppliers = customers + suppliers
customer_suppliers = set(customer_suppliers)

# calculate overlaps

overlap = [x for x in tqdm.tqdm(companies) if x in customer_suppliers]
print(len(overlap))
set_customers = set(customers)
set_suppliers = set(suppliers)
overlap_customers = [x for x in tqdm.tqdm(companies) if x in set_customers]
print(len(overlap_customers))
overlap_suppliers = [x for x in tqdm.tqdm(companies) if x in set_suppliers]
print(len(overlap_suppliers))

# remove non overlapped on stock market
non_overlapping_companies = [x for x in companies if x not in overlap]
# print(len(non_overlapping_companies))

print("Before(Market):", stock_market.shape)
stock_market_updated = stock_market.drop(non_overlapping_companies, axis=1)
print(stock_market_updated.shape)
print("After(Market):", stock_market_updated.shape)

# remove non overlapped on supply chain

print("Before(Supply Chain):", supplychain.shape)
supplychain_updated = supplychain[supplychain.supplier_ticker.isin(overlap_suppliers)]
supplychain_updated = supplychain_updated[supplychain_updated.customer_ticker.isin(overlap_customers)]
print("After(Supply Chain):", supplychain_updated.shape)

stock_market_updated.set_index("Date")
supplychain_updated.set_index("accounting_as_of_date")
supplychain_updated = supplychain_updated[supplychain_updated["accounting_as_of_date"] >= stock_market_updated["Date"].min()]
supplychain_updated = supplychain_updated.drop("Unnamed: 0", axis=1)
supplychain_updated.drop_duplicates(subset=["accounting_as_of_date", "supplier_ticker", "customer_ticker"], keep="first", inplace=True)

grouped_supplychain = supplychain_updated.groupby(['accounting_as_of_date', 'supplier_ticker'])

revenue_sum = grouped_supplychain.agg({'revenue_dependency': 'sum'})

joined_market_info = stock_market_updated.merge(revenue_sum, left_on=["Date", "ticker"], right_on=["accounting_as_of_date", "supplier_ticker"], how="left").fillna(method="ffill")
