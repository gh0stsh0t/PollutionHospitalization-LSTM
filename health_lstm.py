import numpy as np
import pandas as pd
import matplotlib as plt
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

from datetime import datetime


def parse_all(file):
    print(file)
    raw_data = pd.read_csv(file+".csv")
    raw_data.fillna(-1, inplace=True)
    return raw_data

TEMPORARY = 16
X = 1
print(type(X))
for year in range(15, TEMPORARY):
    #,15,16
    for station in [2, 7, 11, 14]:
        file = str(station).zfill(2)+'_'+str(year)
        y = parse_all(file)
        print(y[:5])
        if isinstance(X, int):
            X = y
        else:
            X = pd.merge(X, y, on="date", sort=False)
print(X[:5])
target_raw = pd.read_csv("target.csv")
target_raw = target_raw[:4]
quarters = [0, 0, 0, 0]
for date in X['date'].get_values():
    which = (int(date[2:-3]) - 1)//3
    quarters[which] = quarters[which] + 1
print(quarters)

targets = target_raw['target'].get_values()
y = []
for ind, people in enumerate(targets):
    temp = [people//quarters[ind] for i in range(quarters[ind])]
    y.extend(temp)

y = pd.Series(y)
y = y.values

y_sure = y
values = y.astype('float32')
values = values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
y = scaler.fit_transform(values)

X = X.drop("date", axis=1)
print(X[:5])
X = X.values
X_sure = X
values = X.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(values)

X = X.reshape((X.shape[0], 1, X.shape[1]))
X_sure = X_sure.reshape((X_sure.shape[0], 1, X_sure.shape[1]))
print("{} {}".format(X.shape, y.shape))
# define model
model = Sequential()
model.add(LSTM(50, input_shape=(1, 16)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit model
#for i in range(500):
history = model.fit(X, y,epochs=500, batch_size=72, verbose=2, shuffle=False)
pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
# evaluate model on new data
yhat = model.predict(X_sure)
for i in range(len(X)):
    print('Expected', y_sure[i], 'Predicted', yhat[i])
