# import keras
import numpy as np
import pandas as pd
# from keras.models import Model
# from keras.layers import Input, Dense, Dropout, Flatten, concatenate, Conv2DTranspose, UpSampling2D, Conv2D, MaxPooling2D, ZeroPadding2D, Activation, Lambda
# from keras.optimizers import SGD, Adam
# from keras.utils import to_categorical
# import keras
# import keras.backend as K

def get_batch(dataset, batch_size, lookback_size=20):
    while True:
        companies = dataset["ticker"].unique()
        for batches in range(len(companies) // batch_size):
            x_batch = []
            y_batch = []

            for company in companies[batches*batch_size:min(len(companies),(batches+1)*batch_size)]:
                current_company_rows = dataset.loc(dataset["ticker"] == company)
                current_company_rows =  current_company_rows[current_company_rows["Date"] <='2018-12-31']

                flag = True
                while flag:
                    random_end = random.randint(0, len(current_company_rows) - 1)
                    random_start = max(0, random_end - lookback_size)
                    if random_end - random_start != 20:
                        flag = False

                selected_rows = current_company_rows.iloc[random_start:random_end]
                selected_rows = selected_rows.drop(["Date", "ticker", "fwd_return"])
                selected_rows = selected_rows.drop(out.columns[(out.dtypes.values != np.dtype('float64'))].tolist())

                train = selected_rows.drop("ReturnClassifier").to_numpy()
                label = selected_rows["ReturnClassifier"].to_numpy()
                x_batch.append(train)
                y_batch.append(label)

            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            y_batch = to_categorical(y_batch,3)

            yield x_batch, y_batch


def create_model():
    input = Input(shape=(20,20)) #TODO

    
    conv1 = Conv2D(64, kernel_size=5)(input)
    drop1 = Dropout(0.5)(conv1)
    bn1 = BatchNorm2D()(drop1)
    activ1 = Activation("relu")(bn1)
    conv2 = Conv2D(64, kernel_size=5)(activ1)
    drop2 = Dropout(0.5)(conv2)
    bn2 = BatchNorm2D()(drop2)
    activ2 = Activation("relu")(bn2)
    conv3 = Conv2D(64, kernel_size=5)(activ2)
    drop3 = Dropout(0.5)(conv3)
    bn3 = BatchNorm2D()(drop3)
    activ3 = Activation("relu")(bn3)

    flat = Flatten()(activ3)
    dense1 = Dense(32)(flat)
    drop4 = Dropout(0.5)(dense1)
    bn4 = BatchNorm()(drop4)
    activ4 = Activation("relu")(bn4)
    
    output = Dense(3, activation='softmax')(activ4)
    
    model = Model(input=input, output=output)

    return model


def main():
    dataset = pd.read_pickle("../assets/final_market_risk_supplychain.pkl")

    print(dataset[dataset["Date"] >'2018-12-31'])

if __name__ == "__main__":
    main()