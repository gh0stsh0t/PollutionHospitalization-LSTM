import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Masking, Dropout, Activation
from keras.callbacks import TensorBoard
from math import sqrt
from time import time
from datetime import datetime


def parse_all(file):
    print(file)
    raw_data = pd.read_csv(file+".csv")
    raw_data.fillna(-1, inplace=True)
    return raw_data


def loss_plt(histo):
    x=1
    # for i in range(500):
    # pyplot.plot(histo.history['loss'], label='train')
    # pyplot.plot(histo.history['val_loss'], label='test')
    # pyplot.legend()
    # pyplot.show()

TEMPORARY = 17
holder = {}
X = pd.DataFrame()
for year in range(15, TEMPORARY):
    # ,15,16
    for station in [2, 7, 11, 14]:
        file = str(station).zfill(2)+'_'+str(year)
        y = parse_all(file)
        print(y[:5])
        if year in holder:
            holder[year] = pd.merge(holder[year], y, on="date", sort=False)
        else:
            holder[year] = y

for key in holder:
    X = X.append(other=holder[key], ignore_index=True)

X.to_csv("bogo.csv")
target_raw = pd.read_csv("target.csv")
target_raw = target_raw[:(TEMPORARY - 15) * 4]
quarters = [0 for i in range((TEMPORARY - 15) * 4)]
for date in X['date'].get_values():
    which = ((int(date[2:-3]) - 1)//3) + (int(date[-2:]) - 15) * 4
    quarters[which] = quarters[which] + 1
print(quarters)

targets = target_raw['target'].get_values()
y = []
for ind, people in enumerate(targets):
    temp = [people//quarters[ind] for i in range(quarters[ind])]
    y.extend(temp)

y = pd.Series(y)
y = y.values

y_sure = y.astype('float32')
values = y.astype('float32')
values = values.reshape(-1, 1)
scaler1 = MinMaxScaler(feature_range=(0, 1))
y = scaler1.fit_transform(values)

X = X.drop("date", axis=1)
print(X[:5])
X = X.values
X_sure = X.astype('float32')
values = X.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(values)

X = X.reshape((X.shape[0], 1, X.shape[1]))
X_sure = X_sure.reshape((X_sure.shape[0], 1, X_sure.shape[1]))
print("{} {}".format(X.shape, y.shape))

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
loss = []

def model_fit(layers*, train_X, train_y, epochs=500, optim='rmsprop', batch=10)
    model = Sequential(layers)
    model.compile(loss='mean_squared_error', optimizer=optim)
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch, verbose=1, callbacks=[tensorboard], shuffle=False)
    loss_plt(history)
    yhat = model.predict(X_sure)
    inv_yhat = scaler1.inverse_transform(yhat)
    inv_y = scaler1.inverse_transform(y)
    pyplot.plot(inv_yhat, label="third")
    loss.append((sqrt(mean_squared_error(inv_y, inv_yhat)), mean_absolute_error(inv_y, inv_yhat)))

# define model
model = Sequential()
model.add(Masking(mask_value=-1, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(50, input_shape=(1, 16)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit model
history = model.fit(X, y, epochs=500, batch_size=50, verbose=1, callbacks=[tensorboard], shuffle=False)
loss_plt(history)
# evaluate model on new data
yhat = model.predict(X_sure)
inv_yhat = scaler1.inverse_transform(yhat)
inv_y = scaler1.inverse_transform(y)

pyplot.plot(inv_y, label="real")
pyplot.plot(inv_yhat, label="first")

loss.append((sqrt(mean_squared_error(inv_y, inv_yhat)), mean_absolute_error(inv_y, inv_yhat)))

model_fit([Masking(mask_value=-1, input_shape=(X.shape[1], X.shape[2])),
           LSTM(50, input_shape(input_shape=(X.shape[1], X.shape[2]))),
           Dense(1)],
           X, y, optim='adam', batch=50)

# define model 1
model = Sequential()
model.add(Masking(mask_value=-1, input_shape=(1, 16)))
model.add(LSTM(50, input_shape=(1, 16), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit model
history = model.fit(X, y, epochs=500, batch_size=50, verbose=1, callbacks=[tensorboard], shuffle=False)
loss_plt(history)
# evaluate model on new data
yhat = model.predict(X_sure)
inv_yhat = scaler1.inverse_transform(yhat)
inv_y = scaler1.inverse_transform(y)

pyplot.plot(inv_yhat, label='second')

loss.append((sqrt(mean_squared_error(inv_y, inv_yhat)), mean_absolute_error(inv_y, inv_yhat)))

model_fit([Masking(mask_value=-1, input_shape=(X.shape[1], X.shape[2])),
           LSTM(50, input_shape(input_shape=(X.shape[1], X.shape[2]), return_sequences=True)),
           Dropout(0.2),
           Dense(1)],
           X, y, optim='adam', batch=50)

# define model 2
model = Sequential()
model.add(Masking(mask_value=-1, input_shape=(1, 16)))
model.add(LSTM(50, input_shape=(1, 16), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='rmsprop')
# fit model
history = model.fit(X, y, epochs=500, batch_size=50, verbose=1, callbacks=[tensorboard], shuffle=False)
loss_plt(history)
# evaluate model on new data
yhat = model.predict(X_sure)
inv_yhat = scaler1.inverse_transform(yhat)
inv_y = scaler1.inverse_transform(y)

pyplot.plot(inv_yhat, label="third")

loss.append((sqrt(mean_squared_error(inv_y, inv_yhat)), mean_absolute_error(inv_y, inv_yhat)))

model_fit([Masking(mask_value=-1, input_shape=(X.shape[1], X.shape[2])),
           LSTM(50, input_shape(input_shape=(X.shape[1], X.shape[2]), return_sequences=True)),
           Dropout(0.2),
           LSTM(100),
           Dropout(0.2),
           Dense(1)],
           X, y, batch=50)

# define model 3
model = Sequential()
model.add(Masking(mask_value=-1, input_shape=(1, 16)))
model.add(LSTM(50, input_shape=(1, 16), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation("linear"))
model.compile(loss='mean_squared_error', optimizer='rmsprop')
# fit model
history = model.fit(X, y, epochs=500, batch_size=50, verbose=1, callbacks=[tensorboard], shuffle=False)
loss_plt(history)
# evaluate model on new data
yhat = model.predict(X_sure)
inv_yhat = scaler1.inverse_transform(yhat)
inv_y = scaler1.inverse_transform(y)

pyplot.plot(inv_yhat, label="fourth")
pyplot.legend()

loss.append((sqrt(mean_squared_error(inv_y, inv_yhat)), mean_absolute_error(inv_y, inv_yhat)))

model_fit([Masking(mask_value=-1, input_shape=(X.shape[1], X.shape[2])),
           LSTM(50, input_shape(input_shape=(X.shape[1], X.shape[2]), return_sequences=True)),
           Dropout(0.2),
           LSTM(100),
           Dropout(0.2),
           Dense(1)
           Activation('linear')],
           X, y, batch=50)

# define model 4
model = Sequential()
model.add(Masking(mask_value=-1, input_shape=(1, 16)))
model.add(LSTM(50, input_shape=(1, 16), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation("linear"))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit model
history = model.fit(X, y, epochs=500, batch_size=50, verbose=1, callbacks=[tensorboard], shuffle=False)
loss_plt(history)
# evaluate model on new data
yhat = model.predict(X_sure)
inv_yhat = scaler1.inverse_transform(yhat)
inv_y = scaler1.inverse_transform(y)

pyplot.plot(inv_yhat, label="fifth")
pyplot.legend()

loss.append((sqrt(mean_squared_error(inv_y, inv_yhat)), mean_absolute_error(inv_y, inv_yhat)))

model_fit([Masking(mask_value=-1, input_shape=(X.shape[1], X.shape[2])),
           LSTM(50, input_shape(input_shape=(X.shape[1], X.shape[2]), return_sequences=True)),
           Dropout(0.2),
           LSTM(100),
           Dropout(0.2),
           Dense(1)
           Activation('linear')],
           X, y, optim='adam', batch=50)

# note adam is really bad

# define model 5
model = Sequential()
model.add(Masking(mask_value=-1, input_shape=(1, 16)))
model.add(LSTM(50, input_shape=(1, 16), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation("linear"))
model.compile(loss='mean_squared_error', optimizer='rmsprop')
# fit model
history = model.fit(X, y, epochs=500, batch_size=10, verbose=1, callbacks=[tensorboard], shuffle=False)
loss_plt(history)
# evaluate model on new data
yhat = model.predict(X_sure)
inv_yhat = scaler1.inverse_transform(yhat)
inv_y = scaler1.inverse_transform(y)

pyplot.plot(inv_yhat, label="sixth")
pyplot.legend()

loss.append((sqrt(mean_squared_error(inv_y, inv_yhat)), mean_absolute_error(inv_y, inv_yhat)))

model_fit([Masking(mask_value=-1, input_shape=(X.shape[1], X.shape[2])),
           LSTM(50, input_shape(input_shape=(X.shape[1], X.shape[2]), return_sequences=True)),
           Dropout(0.2),
           LSTM(100),
           Dropout(0.2),
           Dense(1)
           Activation('linear')],
           X, y)

pyplot.show()
pyplot.gcf().clear()
pyplot.plot(inv_y, label="real")
pyplot.plot(inv_yhat, label="sixth")
pyplot.legend()
pyplot.show()
for error in loss:
    print('Test RMSE: {:.2f}\nTest  MAE: {:.2f}\n'.format(error[0], error[1]))
print(model.summary())
