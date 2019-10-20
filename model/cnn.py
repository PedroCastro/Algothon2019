import keras
import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Dropout, Flatten, concatenate, Conv2DTranspose, UpSampling2D, Conv2D, MaxPooling2D, ZeroPadding2D, Activation, Lambda, Conv1D, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
import keras.backend as K
from keras.models import Model
import random
from scipy.stats.mstats import zscore, winsorize


def get_batch(dataset, batch_size, lookback_size=20, mode = "train"):
    while True:
        companies = dataset["ticker"].unique()
        for batches in range(len(companies) // batch_size):
            x_batch = []
            y_batch = []

            for company in companies[batches*batch_size:min(len(companies),(batches+1)*batch_size)]:
                current_company_rows = dataset.loc[dataset["ticker"] == company]
                if mode == "train":
                  current_company_rows =  current_company_rows[current_company_rows["Date"] <='2018-12-31']
                else:
                  current_company_rows =  current_company_rows[current_company_rows["Date"] > '2018-12-31']

                flag = True
                while flag:
                    if mode == "train":
                      random_end = random.randint(0, len(current_company_rows) - 1)
                      random_start = max(0, random_end - lookback_size)
                      if random_end - random_start == 20:
                          flag = False
                    else:
                      random_end = len(current_company_rows) - 1
                      random_start = random_end - lookback_size
                      break
                       

                selected_rows = current_company_rows.iloc[random_start:random_end]
                selected_rows = selected_rows.drop(["Date", "ticker", "fwd_return", "excess_return"], axis=1)
                selected_rows = selected_rows.drop(selected_rows.columns[(selected_rows.dtypes.values != np.dtype('float64'))].tolist(), axis=1)

                train = selected_rows.drop("ReturnClassifier", axis=1).to_numpy()
                label = selected_rows.iloc[-1]["ReturnClassifier"]
                x_batch.append(train)
                y_batch.append(label)

            x_batch = np.array(x_batch)
#             y_batch = np.array(y_batch)
            y_batch = to_categorical(y_batch,3)
  
            yield x_batch, y_batch


def create_model():
    input = Input(shape=(20,28)) #TODO

    
    conv1 = Conv1D(64, kernel_size=5)(input)
    drop1 = Dropout(0.5)(conv1)
    bn1 = BatchNormalization()(drop1)
    activ1 = Activation("relu")(bn1)
    conv2 = Conv1D(64, kernel_size=5)(activ1)
    drop2 = Dropout(0.5)(conv2)
    bn2 = BatchNormalization()(drop2)
    activ2 = Activation("relu")(bn2)
    conv3 = Conv1D(64, kernel_size=5)(activ2)
    drop3 = Dropout(0.5)(conv3)
    bn3 = BatchNormalization()(drop3)
    activ3 = Activation("relu")(bn3)

    flat = Flatten()(activ3)
    dense1 = Dense(32)(flat)
    drop4 = Dropout(0.5)(dense1)
    bn4 = BatchNormalization()(drop4)
    activ4 = Activation("relu")(bn4)
    
    output = Dense(3, activation='softmax')(activ4)
    
    model = Model(input=input, output=output)

    return model

def mrm_c(std,vol):
        value=np.tanh((10/vol)*50*std)
        value[value<-0.8] = -1
        value[value>0.8] = 1
        value[(value>=-0.8)&(value<=0.8)] = 0
        return value

dataset = pd.read_pickle("../assets/final_market_risk_supplychain.pkl")
dataset = dataset.replace([np.inf, -np.inf, np.nan], 0)
dataset["fwd_return"] = np.log(1 + dataset["fwd_return"])
dataset["fwd_return"] = winsorize(dataset["fwd_return"] ,limits = [0.025,0.025])
dataset["excess_return"] = dataset["fwd_return"] - dataset.groupby("Date")["fwd_return"].transform(lambda x: x.mean())
dataset["ReturnClassifier"] = mrm_c(dataset["excess_return"], dataset["vol"])


companies = dataset["ticker"].unique()
batch_size = 32

model = create_model()
model.summary()
model.compile(optimizer=SGD(lr=0.001, momentum =0.9), loss="categorical_crossentropy", metrics=["accuracy"])

model.fit_generator(get_batch(dataset, batch_size), epochs=5, verbose=True, steps_per_epoch=(len(companies) // batch_size), class_weight={0:1.0, 1:1.00, 2:1.00})

### testing

import tqdm
accuracies = []
dates = []
tickers = []
preds = []
for company in tqdm.tqdm(companies):
  
  current_company_rows = dataset.loc[dataset["ticker"] == company]
  current_company_rows = current_company_rows.reset_index()
  
  indexes = None
  indexes = current_company_rows.Date[current_company_rows.Date > '2018-12-31'].index.tolist()
  for idx in indexes:
    start = idx - 20
    
    selected_rows = current_company_rows.iloc[start:idx]
    date = selected_rows.iloc[-1]["Date"]
    ticker = selected_rows.iloc[-1]["ticker"]
    selected_rows = selected_rows.drop(["Date", "ticker", "fwd_return", "excess_return"], axis=1)
    selected_rows = selected_rows.drop(selected_rows.columns[(selected_rows.dtypes.values != np.dtype('float64'))].tolist(), axis=1)
    
    x = selected_rows.drop("ReturnClassifier", axis=1).to_numpy()
    y = selected_rows.iloc[-1]["ReturnClassifier"]
    y = to_categorical(y, 3)

    y_pred = model.predict(np.expand_dims(x, 0))
    if np.argmax(y_pred) == np.argmax(y):
      accuracies += [1]
    else:
      accuracies += [0]
      
    dates += [date]
    tickers += [company]
    class_pred = np.argmax(y_pred)
    if class_pred == 2:
      class_pred = -1
    
    preds += [class_pred]
    
    
    
      
print("Accuracy:", np.mean(accuracies))