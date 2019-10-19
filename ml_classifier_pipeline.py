import pandas as pd
import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from tools import visual_tools
import pickle

from sklearn.metrics import classification_report, accuracy_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt


out = pd.read_pickle("data/out.pkl")

out = out.drop(['index'], axis=1)
out['Date'] = pd.to_datetime(out['Date'])

out['cat_feat_1'] = out['cat_feat_1'].astype(str)


col_li = out.columns.tolist()

train_X, test_X = out[out['Date']<='2018-12-31'].drop('bin', axis=1), out[out['Date']>'2018-12-31'].drop('bin', axis=1)
train_y, test_y = out[out['Date']<='2018-12-31']['bin'], out[out['Date']>'2018-12-31']['bin']

categorical_features = out.columns[(out.dtypes.values != np.dtype('float64'))].tolist()
numeric_features = out.columns[(out.dtypes.vaues == np.type('float64'))].tolist()

numeric_features.remove('bin')

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

clf.fit(train_X, train_y)

pickle.dumps(clf, open('clf.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

pred_y = clf.predict(test_X)


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









