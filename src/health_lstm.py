import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
from math import sqrt
import time
from datetime import datetime
from keras import backend as K
if K.tensorflow_backend._get_available_gpus():
    from keras.layers import LSTM as CuDNNLSTM
    #from keras.layers import CuDNNLSTM
    print("Using GPU for training")
else:
    from keras.layers import LSTM as CuDNNLSTM
    print("Using CPU for training")


class Pollution:

    def __init__(self):
        self.loss = []
        self.scalers = [MinMaxScaler(feature_range=(0, 1)), MinMaxScaler(feature_range=(0, 1))]
        self.y = np.ndarray
        self.times = []
        self.predictions = []
        self.split = 0

    def parse_all(self, file):
        print(file)
        raw_data = pd.read_csv("Data/" + file + ".csv")
        return raw_data

    def model_compile_double(self, optimizer='adam', neurons1=50, neurons2=50, dropout1=0.2, dropout2=0.2):
        model = Sequential()
        model.add(CuDNNLSTM(neurons1, input_shape=(1, 16), return_sequences=True))
        model.add(Dropout(dropout1))
        model.add(CuDNNLSTM(neurons2))
        model.add(Dropout(dropout2))
        model.add(Dense(1))
        model.add(Activation('linear'))
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])
        return model

    def model_compile_triple(self, optimizer='adam', neurons1=50, neurons2=50, neurons3=50, dropout1=0.2, dropout2=0.2):
        model = Sequential()
        model.add(CuDNNLSTM(neurons1, input_shape=(1, 16), return_sequences=True))
        model.add(Dropout(dropout1))
        model.add(CuDNNLSTM(neurons2, return_sequences=True))
        model.add(Dropout(dropout2))
        model.add(CuDNNLSTM(neurons3))
        model.add(Dense(1))
        model.add(Activation('linear'))
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])
        return model

    def model_fit(self, layers, train_X, train_y, label, test_X=0, epochs=500, optim='rmsprop', batch=10, save=False):
        test_X = self.X_test if type(test_X) is int else test_X
        test_y = self.y_test
        start = datetime.now()
        model = Sequential(layers)
        model.compile(loss='mean_squared_error', optimizer=optim)
        hist = model.fit(train_X, train_y, batch_size=batch, epochs=epochs, verbose=self.verbosity, shuffle=False, validation_data=(test_X, test_y))
        with open('report.txt','a') as fh:
             model.summary(print_fn=lambda x: fh.write(x + '\n'))
        if save:
            model.save("models/lstm.h5")
            joblib.dump(self.scalers[1], "models/x_scaler.save")
            joblib.dump(self.scalers[0], "models/y_scaler.save")
        all_X = np.concatenate((train_X, test_X))
        yhat = model.predict(all_X)
        inv_yhat = self.scalers[0].inverse_transform(yhat)
# Himua na ang predictions kayma pasa nalang sa dictionry para ma chuy, pero nag groupings must be in a special way
        self.predictor_magtanggol(inv_yhat, label)
        self.create_loss(hist, label)
        self.times.append(datetime.now() - start)

    def svm_fit(self, X, y):
        X_scaled = StandardScaler().fit_transform(X)
        kfold = KFold(n_splits = 5, shuffle=False)
        svm = SVR(verbose=True)
        svm.fit(X_scaled, y)
        scores = cross_val_score(svm, X_scaled, y, cv=kfold)
        yhat = svm.predict(X_scaled)
        self.predictor_magtanggol(yhat, "svm")

    def manual(self, X, y):
        # # define and fit model 0
        # self.model_fit([CuDNNLSTM(50, input_shape=(X.shape[1], X.shape[2])),
        #                 Dense(1)],
        #                train_X=X, train_y=y, optim='adam', batch=50, label="first")
        # # define and fit model 1
        # self.model_fit([CuDNNLSTM(50, input_shape=(X.shape[1], X.shape[2])),
        #                 Dropout(0.2),
        #                 Dense(1)],
        #                train_X=X, train_y=y, optim='adam', batch=50, label="second")
        # # define and fit model 2
        # self.model_fit([CuDNNLSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
        #                 Dropout(0.2),
        #                 CuDNNLSTM(100),
        #                 Dropout(0.2),
        #                 Dense(1)],
        #                train_X=X, train_y=y, batch=50, label="third")
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
        # self.model_fit([CuDNNLSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
        #                 Dropout(0.2),
        #                 CuDNNLSTM(100),
        #                 Dropout(0.2),
        #                 Dense(1),
        #                 Activation('linear')],
        #                train_X=X, train_y=y, label="sixth")
        # define and fit model 6
        # self.model_fit([CuDNNLSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
        #                 Dropout(0.2),
        #                 CuDNNLSTM(100),
        #                 Dropout(0.2),
        #                 Dense(1),
        #                 Activation('linear')],
        #                train_X=X, train_y=y, label="seventh")
        # define and fit model 7
        # self.model_fit([CuDNNLSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
        #                 Dropout(0.2),
        #                 CuDNNLSTM(100),
        #                 Dropout(0.2),
        #                 Dense(1),
        #                 Activation('linear')],
        #                train_X=X, train_y=y, epochs=100, batch=5, label="eighth")
        # # define and fit model 8
        # self.model_fit([CuDNNLSTM(25, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
        #                 Dropout(0.2),
        #                 CuDNNLSTM(50),
        #                 Dropout(0.2),
        #                 Dense(1),
        #                 Activation('linear')],
        #                train_X=X, train_y=y, epochs=250, batch=5, label="ninth")
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
        # self.model_fit([CuDNNLSTM(100, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
        #                 Dropout(0.2),
        #                 CuDNNLSTM(200, return_sequences=True),
        #                 Dropout(0.2),
        #                 CuDNNLSTM(100),
        #                 Dense(1),
        #                 Activation('linear')],
        #                train_X=X, train_y=y, label="eleventh")
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
                       train_X=X, train_y=y, batch=5, label="thirteenth")
        # Note: Worse performing, thus batch size must be 50 > s > 5
        # define and fit model 13
        # self.model_fit([CuDNNLSTM(150, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
        #                 Dropout(0.2),
        #                 CuDNNLSTM(300, return_sequences=True),
        #                 Dropout(0.2),
        #                 CuDNNLSTM(200),
        #                 Dense(1),
        #                 Activation('linear')],
        #                train_X=X, train_y=y, batch=20, label="fourteenth")
        # # define and fit model 14
        # self.model_fit([CuDNNLSTM(150, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
        #                 Dropout(0.2),
        #                 CuDNNLSTM(300, return_sequences=True),
        #                 Dropout(0.2),
        #                 CuDNNLSTM(300),
        #                 Dense(1),
        #                 Activation('linear')],
        #                train_X=X, train_y=y, batch=20, label="fifteenth")
        # define and fit model 15
        # self.model_fit([CuDNNLSTM(150, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
        #                 Dropout(0.2),
        #                 CuDNNLSTM(400, return_sequences=True),
        #                 Dropout(0.2),
        #                 CuDNNLSTM(300),
        #                 Dense(1),
        #                 Activation('linear')],
        #                train_X=X, train_y=y, batch=4, label="sixteenth")
        # define and fit model BEST
        self.model_fit([CuDNNLSTM(150, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(300, return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(200, return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(450),
                        Dense(1),
                        Activation('linear')],
                       train_X=X, train_y=y, batch=20, label="Four layers")
        self.model_fit([CuDNNLSTM(150, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(300, return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(200, return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(450),
                        Dense(1),
                        Activation('linear')],
                       train_X=X, train_y=y, batch=5, label="Four layers v2")
        # define and fit model BEST
        self.model_fit([CuDNNLSTM(250, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(200, return_sequences=True),
                        Dropout(0.2),
                        CuDNNLSTM(300),
                        Dense(1),
                        Activation('linear')],
                       train_X=X, train_y=y, batch=5, label="Grid Searched", save=True)

    def predictor_magtanggol(self, inv_yhat, label):
        yforms = [self.y,              inv_yhat,
                  self.y[self.split:], inv_yhat[self.split:],
                  self.y[:self.split], inv_yhat[:self.split]]
        self.predictions.append((yforms[1], label))
        self.loss.append((sqrt(mean_squared_error(yforms[0], yforms[1])), mean_absolute_error(yforms[0], yforms[1]),
                          sqrt(mean_squared_error(yforms[2], yforms[3])), mean_absolute_error(yforms[2], yforms[3]),
                          sqrt(mean_squared_error(yforms[4], yforms[5])), mean_absolute_error(yforms[4], yforms[5])))

    def create_loss(self, hist, label):
        plt.subplot(2, 1, 1)
        plt.plot(hist.history['loss'], label=label)
        plt.ylabel('Train Loss')
        plt.subplot(2, 1, 2)
        plt.plot(hist.history['val_loss'], label=label)
        plt.ylabel('Test Loss')

    def main(self, verbosity, auto, graphing):
        self.verbosity = verbosity
        holder = {}
        X = pd.DataFrame()
        for year in range(15, 18):
            for station in [2, 7, 11, 14]:
                file = str(station).zfill(2) + '_' + str(year)
                y = self.parse_all(file)
                if year in holder:
                    holder[year] = pd.merge(holder[year], y, how="outer", on="date", sort=False)
                else:
                    holder[year] = y

        for key in holder:
            X = X.append(other=holder[key], ignore_index=True)

        X.to_csv("Data/Features.csv")
        X.dropna(thresh=6, inplace=True)
        X.fillna(-1, inplace=True)
        target_raw = pd.read_csv("Data/MonthlyTarget.csv")
        weeks = [0 for _ in range(len(target_raw))]
        for date in X['date'].get_values():
            which = (int(date[2:-3]) - 1) + (int(date[-2:]) - 15) * 12
            weeks[which] = weeks[which] + 1
        print(weeks)

        targets = target_raw['target'].get_values()
        y = []
        for ind, people in enumerate(targets):
            amp = [people // weeks[ind] for _ in range(weeks[ind])]
            y.extend(amp)

        self.y = pd.Series(y).values.astype('float32').reshape(-1, 1)
        self.split = (self.y.shape[0] // 3) * 2
        svm_y = y
        y = self.scalers[0].fit_transform(self.y)
        self.y_test = y[self.split:, :]
        y = y[:self.split, :]

        # X = pd.read_csv("KNN.csv")
        X = X.drop("date", axis=1)
        print(X[:5])
        X = X.values.astype('float32')
        svm_X=X
        X = self.scalers[1].fit_transform(X)
        all_X = X.reshape((X.shape[0], 1, X.shape[1]))

        self.X_test = X[self.split:]
        self.X_test = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
        X = X[:self.split]
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        print("{} {}".format(X.shape, y.shape))
        startTime = datetime.now()

        # SVM model
        self.svm_fit(svm_X, svm_y)
        # Basic feed forward neural network
        mlp_train_X = self.scalers[1].transform(svm_X[:self.split])
        mlp_test_X = self.scalers[1].transform(svm_X[self.split:])
        self.model_fit([Dense(12, input_dim=svm_X.shape[1], activation='relu'),
                        Dense(8, activation='relu'),
                        Dense(1)],
                        train_X=mlp_train_X, train_y=y, test_X=mlp_test_X, optim='adam', epochs=500, batch=2, label="FFDNN")
        if auto == 0:
            self.manual(X, y)
            for ind, x in enumerate(self.times):
                print("Time for {}: {}".format(ind, x))
            print("\nTraining time of All Models {}\n".format(datetime.now() - startTime))

            plt.legend()
            if(graphing < 1):
                plt.show()
            plt.gcf().clear()

            split = (self.split / self.y.shape[0]) * 100
            print("Train-Test split: {:.2f} {:.2f}".format(split, 100 - split))
            for index, error in enumerate(self.loss):
                print(
                    'Model #{}\nTotal RMSE: {:.2f}\nTotal MAE: {:.2f}\nTrain RMSE: {:.2f}\nTrain  MAE: {:.2f}'.format(index - 1, error[0], error[1], error[2], error[3]))
                # print('\nTest RMSE: {:.2f}\nTest  MAE: {:.2f}\n'.format(error[4], error[5]), file=open("Loss_vals.txt", "a"))
                print('\n{:.2f}\n{:.2f}\n'.format(error[4], error[5]), file=open("Data/Loss_vals.txt", "a"))
        else:
            print("Starting GridSearch in 5")
            time.sleep(5)
            param_grid = {'neurons1': list(range(100, 325, 75)),
                          'neurons2': list(range(200, 500, 100)),
                          # 'neurons3': list(range(200, 500, 150)),
                          'dropout1': [0.2],
                          'dropout2': [0.2],
                          'optimizer': ['adam', 'rmsprop'],
                          'batch_size': [5, 20, 50]}
            # model = KerasRegressor(build_fn=self.model_compile_triple, epochs=500, batch_size=10, verbose=verbosity)
            model = KerasRegressor(build_fn=self.model_compile_double, epochs=500, batch_size=10, verbose=verbosity)
            my_cv = TimeSeriesSplit(n_splits=2).split(X)
            grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=my_cv)
            grid.fit(X, y)
            print("\nTraining time of All Models {}\n".format(datetime.now() - startTime))
            print("Best: {:.2f} using {}".format(grid.best_score_, grid.best_params_))
            yhat = grid.best_estimator_.predict(all_X)
            yhat = pd.Series(yhat).values.astype('float32').reshape(-1, 1)
            inv_yhat = self.scalers[0].inverse_transform(yhat)
            self.predictor_magtanggol(inv_yhat, "Best")
            plt.legend()
            if(graphing < 1):
                plt.show()
            plt.gcf().clear()
        plt.plot(self.y, label="real")
        for vals in self.predictions:
            plt.plot(vals[0], label=vals[1])
        plt.legend()
        if(graphing < 1):
            plt.show()
        plt.gcf().clear()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        x = int(sys.argv[1])%2
        z = int(sys.argv[1])//2
        if len(sys.argv) > 2:
            y = int(sys.argv[2])
        else:
            y = 0
    else:
        x = 0
        y = 0
        z = 0
    Pollution().main(x, y, z)