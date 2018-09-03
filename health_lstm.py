import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Masking, Dropout, Activation
from keras.callbacks import EarlyStopping
from summarywriter import TrainValTensorBoard
from math import sqrt
from datetime import datetime


class Pollution:

    def __init__(self):
#log_dir="logs/{}".format(datetime.now().strftime("%Y%m%d-%H%M%S")),
        tensorboard = TrainValTensorBoard(histogram_freq=1, write_graph=True)
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
        self.callbacks = [tensorboard, earlystop]
        self.loss = []
        self.scalers = [MinMaxScaler(feature_range=(0, 1)), MinMaxScaler(feature_range=(0, 1))]
        self.y = np.ndarray
        self.X = np.ndarray

    def parse_all(self, file):
        print(file)
        raw_data = pd.read_csv(file + ".csv")
        raw_data.fillna(-1, inplace=True)
        return raw_data

    def model_fit(self, layers, train_X, train_y, label, epochs=500, optim='rmsprop', batch=10):
        model = Sequential(layers)
        model.compile(loss='mean_squared_error', optimizer=optim)
        model.fit(train_X, train_y, epochs=epochs, batch_size=batch, verbose=1, callbacks=self.callbacks, validation_data=(self.X_test, self.y_test), shuffle=False)
        yhat = model.predict(self.X)
        inv_yhat = self.scalers[0].inverse_transform(yhat)
        pyplot.plot(inv_yhat, label=label)
        self.loss.append((sqrt(mean_squared_error(self.y, inv_yhat)), mean_absolute_error(self.y, inv_yhat)))
        return model

    def main(self):
        TEMPORARY = 17
        holder = {}
        X = pd.DataFrame()
        for year in range(15, TEMPORARY):
            # ,15,16
            for station in [2, 7, 11, 14]:
                file = str(station).zfill(2) + '_' + str(year)
                y = self.parse_all(file)
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
        quarters = [0 for _ in range((TEMPORARY - 15) * 4)]
        for date in X['date'].get_values():
            which = ((int(date[2:-3]) - 1) // 3) + (int(date[-2:]) - 15) * 4
            quarters[which] = quarters[which] + 1
        print(quarters)

        targets = target_raw['target'].get_values()
        y = []
        for ind, people in enumerate(targets):
            y.extend([people // quarters[ind] for _ in range(quarters[ind])])

        self.y = pd.Series(y)[1:].values.astype('float32').reshape(-1, 1)
        pyplot.plot(self.y, label="real")

        splitter = y.shape[0] // 3
        y = self.scalers[0].fit_transform(self.y)
        self.y_test = y[-splitter:, :]
        y = y[:-splitter, :]

        X = X.drop("date", axis=1)
        print(X[:5])
        X = X[:-1].values.astype('float32')
        X = self.scalers[1].fit_transform(X)
        all_X = X.reshape((X.shape[0], 1, X.shape[1]))

        self.X = all_X
        self.X_test = X[-splitter:, :]
        self.X_test = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
        X = X[:-splitter, :]
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        print("{} {}".format(X.shape, y.shape))
        # define and fit model 0
        self.model_fit([Masking(mask_value=-1, input_shape=(X.shape[1], X.shape[2])),
                        LSTM(50, input_shape=(X.shape[1], X.shape[2])),
                        Dense(1)],
                       train_X=X, train_y=y, optim='adam', batch=50, label="first")
        # define and fit model 1
        self.model_fit([Masking(mask_value=-1, input_shape=(X.shape[1], X.shape[2])),
                        LSTM(50, input_shape=(X.shape[1], X.shape[2])),
                        Dropout(0.2),
                        Dense(1)],
                       train_X=X, train_y=y, optim='adam', batch=50, label="second")
        # define and fit model 2
        self.model_fit([Masking(mask_value=-1, input_shape=(X.shape[1], X.shape[2])),
                        LSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                        Dropout(0.2),
                        LSTM(100),
                        Dropout(0.2),
                        Dense(1)],
                       train_X=X, train_y=y, batch=50, label="third")
        # define and fit model 3
        self.model_fit([Masking(mask_value=-1, input_shape=(X.shape[1], X.shape[2])),
                        LSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                        Dropout(0.2),
                        LSTM(100),
                        Dropout(0.2),
                        Dense(1),
                        Activation('linear')],
                       train_X=X, train_y=y, batch=50, label="fourth")
        # define and fit model 4
        self.model_fit([Masking(mask_value=-1, input_shape=(X.shape[1], X.shape[2])),
                        LSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                        Dropout(0.2),
                        LSTM(100),
                        Dropout(0.2),
                        Dense(1),
                        Activation('linear')],
                       train_X=X, train_y=y, optim='adam', batch=50, label="fifth")  # note adam is really bad
        # define and fit model 5
        self.model_fit([Masking(mask_value=-1, input_shape=(X.shape[1], X.shape[2])),
                        LSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                        Dropout(0.2),
                        LSTM(100),
                        Dropout(0.2),
                        Dense(1),
                        Activation('linear')],
                       train_X=X, train_y=y, label="sixth")
        # define and fit model 6
        self.model_fit([LSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                        Dropout(0.2),
                        LSTM(100),
                        Dropout(0.2),
                        Dense(1),
                        Activation('linear')],
                       train_X=X, train_y=y, label="seventh")

        pyplot.legend()
        pyplot.show()
        pyplot.gcf().clear()
        for error in self.loss:
            print('Test RMSE: {:.2f}\nTest  MAE: {:.2f}\n'.format(error[0], error[1]))

if __name__ == "__main__":
    Pollution().main()
