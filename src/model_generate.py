import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import json
from keras.layers import CuDNNLSTM
scalers = [MinMaxScaler(feature_range=(0, 1)), MinMaxScaler(feature_range=(0, 1))]


def parse_all(file):
    print(file)
    raw_data = pd.read_csv("Data/" + file + ".csv")
    return raw_data

def main():
    holder = {}
    X = pd.DataFrame()
    for year in range(15, 18):
        for station in [2, 7, 11, 14]:
            file = str(station).zfill(2) + '_' + str(year)
            y = parse_all(file)
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

    y = pd.Series(y).values.astype('float32').reshape(-1, 1)
    split = (y.shape[0] // 3) * 2
    y = scalers[0].fit_transform(y)
    y_test = y[split:, :]
    y = y[:split, :]

    # X = pd.read_csv("KNN.csv")
    X = X.drop("date", axis=1)
    print(X[:5])
    X = X.values.astype('float32')
    svm_X = X
    X = scalers[1].fit_transform(X)
    all_X = X.reshape((X.shape[0], 1, X.shape[1]))

    X_test = X[split:]
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    X = X[:split]
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    print("{} {}".format(X.shape, y.shape))
    model_fit([CuDNNLSTM(250, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
               Dropout(0.2),
               CuDNNLSTM(200, return_sequences=True),
                    Dropout(0.2),
                    CuDNNLSTM(300),
                    Dense(1),
                    Activation('linear')],
                   train_X=X, train_y=y, test_X=X_test, test_y=y_test, batch=5, save=True)


def model_fit(layers, train_X, train_y, test_X, test_y, epochs=500, optim='rmsprop', batch=10, save=False):
    model = Sequential(layers)
    model.compile(loss='mean_squared_error', optimizer=optim)
    model.fit(train_X, train_y, batch_size=batch, epochs=epochs, verbose=1, shuffle=False,
                     validation_data=(test_X, test_y))
    if save:
        model.save("models/lstm.h5")
        with open('models/lstm.json', 'w') as file:
            json.dump(model.to_json(), file, indent=2)
        model.save_weights('models/lstm_weights.h5')
        joblib.dump(scalers[1], "models/x_scaler.save")
        joblib.dump(scalers[0], "models/y_scaler.save")

main()
