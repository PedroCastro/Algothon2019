import pandas as pd
import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
#from tools import visual_tools
import pickle
from scipy.stats.mstats import zscore, winsorize


from sklearn.metrics import classification_report, accuracy_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt

def mrm_c(std,vol):
        value=np.tanh((10/vol)*50*std)
        value[value<-0.8] = -1
        value[value>0.8] = 1
        value[(value>=-0.8)&(value<=0.8)] = 0
        return value

def xgboost_model(file, risk=True, momentum=True, supplychain=True):
	out = pd.read_pickle("../assets/final_market_risk_supplychain.pkl")
	out = out.replace([np.inf, -np.inf], np.nan)
	out = out.dropna()
	out["fwd_return"] = np.log(1 + out["fwd_return"])
	out["fwd_return"] = winsorize(out["fwd_return"] ,limits = [0.025,0.025])
	out["excess_return"] = out["fwd_return"] - out.groupby("Date")["fwd_return"].transform(lambda x: x.mean())
	out["ReturnClassifier"] = mrm_c(out["excess_return"], out["vol"])
	print("Loaded Data...")

	out = out.drop(["datepll", "datepll.1"], axis=1)
	if not risk:
		out = out.drop(['crating', 'orating',
       'history', 'cond_ind', 'finance', 'moved', 'sales', 'hicdtavg',
       'pexp_s_n', 'pexp_30', 'pexp_60', 'pexp_90', 'pexp_180', 'bnkrpt',
       'dbt_ind', 'uccfilng', 'cscore', 'cpct', 'fpct', 'paynorm', 'pubpvt',
       'pex_sn1', 'bd_ind'], axis=1)

	if not momentum:
		out = out.drop(["mom"], axis=1)

	if not supplychain:
		out = out.drop(['revenue_dependency', 'adj_close_dependency', 'mom_dependency',
       'vol_dependency', 'MACD_dependency'], axis=1)

	#out = out.drop(['index'], axis=1)
	out['Date'] = pd.to_datetime(out['Date'])
	#out.info()
	#exit(0)
	#out['cat_feat_1'] = out['fwd_return'].astype(str)
	
	#out = out.replace([np.inf, -np.inf], np.nan)
	#out = out.dropna()
	col_li = out.columns.tolist()
	
	train_X, test_X = out[out['Date']<='2018-12-31'].drop(["ReturnClassifier", "excess_return", "fwd_return", "Date", "ticker"], axis=1), out[out['Date']>'2018-12-31'].drop(["excess_return", "ReturnClassifier", "fwd_return", "Date", "ticker"], axis=1)
	train_y, test_y = out[out['Date']<='2018-12-31']['ReturnClassifier'], out[out['Date']>'2018-12-31']['ReturnClassifier']
	
	#out[out['Date']>'2018-12-31']["Date"].to_pickle("date.pkl")
	#np.savetxt("ticker.txt", out[out['Date']>'2018-12-31']["ticker"], fmt='%s;')
	

	categorical_features = out.columns[(out.dtypes.values != np.dtype('float64'))].tolist()
	numeric_features = out.columns[(out.dtypes.values == np.dtype('float64'))].tolist()
	
	numeric_features.remove('ReturnClassifier')
	numeric_features.remove('fwd_return')
	numeric_features.remove('excess_return')
	categorical_features.remove('Date')
	categorical_features.remove('ticker')
	
	numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scalar', StandardScaler())])
	
	categorical_transfomer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                       	('onehot_sparse', OneHotEncoder(handle_unknown='ignore'))])
	
	preprocessor = ColumnTransformer(
		transformers=[
			('num', numeric_transformer, numeric_features),
			('cat', categorical_transfomer, categorical_features)
		]
	)
	
	params ={'num_class': 3, 'objective': 'multi:softprob'}
	
	clf = Pipeline(steps=[('preprocessor', preprocessor),
                      	('classifier', XGBClassifier(params=params))])
	print("Preparing to train...")
	#print(train_X.shape)
	clf.fit(train_X, train_y)
	
	# pickle.dumps(clf, open('clf.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	
	pred_y = clf.predict(test_X)
	#np.savetxt(file, pred_y)
	
	print(accuracy_score(test_y, pred_y))
	
	onehot_columns =clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot_sparse'].get_feature_names(input_features=categorical_features)

	feature_importance = pd.Series(data=clf.named_steps['classifier'].feature_importances_, index=np.append(numeric_features, onehot_columns))

	feature_importance = feature_importance.sort_values(ascending=False)

	imp_coef = feature_importance.copy()
	imp_coef = imp_coef[imp_coef!=0][0:20]
	imp_coef.to_pickle(file)
	#print(imp_coef)
	#print(confusion_matrix(test_y,pred_y))	
	#imp_coef.plot(kind='barh')
	#fig =plt.gcf()
	#fig.set_size_inches(30,10)
	#plt.title("XGBoost Model Top 20 Features")
	#fig.subplots_adjust(left=0.45)


"""
np.set_printoptions(precision=2)
visual_tools.plot_confusion_matrix(test_y,pred_y, classes=[-1, 0, 1], title='Test Confusion Matrix')
plt.show()
plt.close()

onehot_columns =clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot_sparse'].get_feature_names(input_features=categorical_features)

feature_importance = pd.Series(data=clf.named_steps['classifier'].feature_importances_, index=np.append(numeric_features, onehot_columns))

feature_importance = feature_importance.sort_values(ascending=False)

imp_coef = feature_importance.copy()
imp_coef = imp_coef[imp_coef!=0][0:20]
imp_coef.plot(kind='barh')
fig =plt.gcf()
fig.set_size_inches(30,10)
plt.title("XGBoost Model Top 20 Features")
fig.subplots_adjust(left=0.45)

"""


if __name__ == "__main__":
	xgboost_model("all.pck")
	#xgboost_model("risk.txt", risk=False)
	#xgboost_model("momentum.txt", momentum=False)
	#xgboost_model("supplychain.txt", supplychain=False)
	#xgboost_model("none.txt", risk=False, momentum=False, supplychain=False)




