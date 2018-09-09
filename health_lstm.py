import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, CuDNNLSTM, Masking, Dropout, Activation
from keras.callbacks import EarlyStopping
from summarywriter import TrainValTensorBoard
from math import sqrt
from datetime import datetime
import sys
from keras import backend as K

K.set_learning_phase(1)
class Pollution:

    def __init__(self):
#log_dir="logs/{}".format(datetime.now().strftime("%Y%m%d-%H%M%S")),
        tensorboard = TrainValTensorBoard(histogram_freq=1, write_graph=True)
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
        self.callbacks = [tensorboard, earlystop]
        self.loss = []
        self.scalers = [MinMaxScaler(feature_range=(0, 1)), MinMaxScaler(feature_range=(0, 1))]
        self.y = np.ndarray
        self.X = np.ndarray
        self.times = []
        self.models = []

    def parse_all(self, file):
        print(file)
        raw_data = pd.read_csv(file + ".csv")
        raw_data.fillna(-1, inplace=True)
        return raw_data

    def model_fit(self, layers, train_X, train_y, label, epochs=500, optim='rmsprop', batch=10):
        start = datetime.now()
        model = Sequential(layers)
        model.compile(loss='mean_squared_error', optimizer=optim)
        model.fit(train_X, train_y, epochs=epochs, batch_size=batch, verbose=self.verbosity, shuffle=False)#, callbacks=self.callbacks, validation_data=(self.X_test, self.y_test))
        yhat = model.predict(self.X)
        inv_yhat = self.scalers[0].inverse_transform(yhat)
        pyplot.plot(inv_yhat, label=label)
        self.loss.append((sqrt(mean_squared_error(self.y, inv_yhat)), mean_absolute_error(self.y, inv_yhat)))
        self.times.append(datetime.now() - start)
        self.models.append(model)

    def main(self, verbosity):
        self.verbosity = verbosity
        TEMPORARY = 18
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

        splitter = (self.y.shape[0] // 3) * 2
        y = self.scalers[0].fit_transform(self.y)
        self.y_test = y[splitter:, :]
        y = y[:splitter, :]

        X = X.drop("date", axis=1)
        print(X[:5])
        X = X[:-1].values.astype('float32')
        X = self.scalers[1].fit_transform(X)
        all_X = X.reshape((X.shape[0], 1, X.shape[1]))

        self.X = all_X
        self.X_test = X[splitter:, :]
        self.X_test = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
        X = X[:splitter, :]
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        print("{} {}".format(X.shape, y.shape))
        startTime = datetime.now()

        # define and fit model 0
        self.model_fit([CuDNNLSTM(50, input_shape=(X.shape[1], X.shape[2])),
                        Dense(1)],
                       train_X=X, train_y=y, optim='adam', batch=50, label="first")
        # define and fit model 1
        self.model_fit([CuDNNLSTM(50, input_shape=(X.shape[1], X.shape[2])),
                        Dropout(0.2),
                        Dense(1)],
                       train_X=X, train_y=y, optim='adam', batch=50, label="second")
        # define and fit model 2
        self.model_fit([CuDNNLSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(100),
                        Dropout(0.2),
                        Dense(1)],
                       train_X=X, train_y=y, batch=50, label="third")
        # define and fit model 3
        self.model_fit([CuDNNLSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(100),
                        Dropout(0.2),
                        Dense(1),
                        Activation('linear')],
                       train_X=X, train_y=y, batch=50, label="fourth")
        # define and fit model 4
        self.model_fit([CuDNNLSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(100),
                        Dropout(0.2),
                        Dense(1),
                        Activation('linear')],
                       train_X=X, train_y=y, optim='adam', batch=50, label="fifth")
        # Note: adam is really bad
        # define and fit model 5
        self.model_fit([CuDNNLSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(100),
                        Dropout(0.2),
                        Dense(1),
                        Activation('linear')],
                       train_X=X, train_y=y, label="sixth")
        # define and fit model 6
        self.model_fit([CuDNNLSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(100),
                        Dropout(0.2),
                        Dense(1),
                        Activation('linear')],
                       train_X=X, train_y=y, label="seventh")
        # define and fit model 7
        self.model_fit([CuDNNLSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(100),
                        Dropout(0.2),
                        Dense(1),
                        Activation('linear')],
                       train_X=X, train_y=y, epochs=100, batch=5, label="eighth")
        # define and fit model 8
        self.model_fit([CuDNNLSTM(25, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(50),
                        Dropout(0.2),
                        Dense(1),
                        Activation('linear')],
                       train_X=X, train_y=y, epochs=250, batch=5, label="ninth")
        # define and fit model 9
        self.model_fit([CuDNNLSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(100, return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(100),
                        Dense(1),
                        Activation('linear')],
                       train_X=X, train_y=y, batch=5, label="tenth")
        # define and fit model 10
        self.model_fit([CuDNNLSTM(100, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(200, return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(100),
                        Dense(1),
                        Activation('linear')],
                       train_X=X, train_y=y, label="eleventh")
        # define and fit model 11
        self.model_fit([CuDNNLSTM(150, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(250, return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(200),
                        Dense(1),
                        Activation('linear')],
                       train_X=X, train_y=y, label="twelfth")
        # define and fit model 12
        self.model_fit([CuDNNLSTM(150, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(300, return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(200),
                        Dense(1),
                        Activation('linear')],
                       train_X=X, train_y=y, batch=5, label="thirteenth ")
        # Note: Worse performing, thus batch size must be 50 > s > 5
        # define and fit model 13
        self.model_fit([CuDNNLSTM(150, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(300, return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(200),
                        Dense(1),
                        Activation('linear')],
                       train_X=X, train_y=y, batch=20, label="fourteenth")
        # define and fit model 14
        self.model_fit([CuDNNLSTM(150, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(300, return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(300),
                        Dense(1),
                        Activation('linear')],
                       train_X=X, train_y=y, batch=20, label="fifteenth")

        for ind, x in enumerate(self.times):
            print("Time for {}: {}".format(ind, x))
        print("\nTraining time of All Models {}\n".format(datetime.now() - startTime))
        pyplot.legend()
        pyplot.show()
        pyplot.gcf().clear()
        splitter = (splitter / self.y.shape[0] ) * 100
        print("Train-Test split: {:.2f} {.2f}".format(splitter, 100-splitter))
        for index, error in enumerate(self.loss):
            print('Model #{}\nTest RMSE: {:.2f}\nTest  MAE: {:.2f}\n'.format(index+1, error[0], error[1]))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        x = int(sys.argv[1])
    else:
        x = 0
    Pollution().main(x)
