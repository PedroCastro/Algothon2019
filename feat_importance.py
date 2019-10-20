import pandas as pd
import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tools import visual_tools

import pickle

from sklearn.metrics import classification_report, accuracy_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt


imp_coef = pd.read_pickle("data/all.pkl")


imp_coef.plot(kind='barh')
fig =plt.gcf()
fig.set_size_inches(30,10)
plt.title("XGBoost Model Top 20 Features")
fig.subplots_adjust(left=0.45)
plt.show()









