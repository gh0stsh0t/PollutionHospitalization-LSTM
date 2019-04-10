import numpy as np
import os
from sklearn.externals import joblib
from flask import Flask, render_template, request, jsonify
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras import backend as kk

# initialize our Flask application and the Keras model
app = Flask(__name__, template_folder='src/templates')


def load_models():
    global model
    global x_scaler
    global y_scaler
    global graph
    graph = tf.get_default_graph()
    if kk.tensorflow_backend._get_available_gpus():
        model = load_model('models/lstm.h5')
    else:
        import json
        from keras.models import model_from_json
        with open('models/lstm.json', 'r') as file:
            json_data = json.load(file)
            json_data = json_data.replace('CuDNNLSTM', 'LSTM')
            json_data = json_data.replace('cu_dnn', '')
        model = model_from_json(json_data)
        model.load_weights('models/lstm_weights.h5')
    x_scaler = joblib.load('models/x_scaler.save')
    y_scaler = joblib.load('models/y_scaler.save')
    model.summary()


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print(request.headers)
        if "multipart/form-data" in request.headers.get('Content-Type'):
            print("file uploaded")
            return render_template('upload.html', predict=uploadr_file(request))
        else:
            print("Tabular input")
            print(request.form)
            read_submission(request)
            return render_template('upload.html', predict=read_submission(request))
    else:
        return render_template('upload.html')


def read_submission(sent):
    X = []
    for station in ['02', '07', '11', '14']:
        for pollutant in ['pm10', 'so2', 'no2', 'o3']:
            try:
                value = float(sent.form[station+'_'+pollutant])
                value = -1 if value < 0 else value
            except ValueError:
                value = -1
            print("{} {} {}".format(X, value, sent.form[station+'_'+pollutant]))
            X.append(value)
    X = np.array(X).astype('float32').reshape(1, -1)
    return predictor(X)


def uploadr_file(sent):
    f = sent.files['file']
    X = pd.read_csv(f, index_col=0)
    print(3)
    X.fillna(-1, inplace=True)
    print(4)
    X = X.drop("date", axis=1)
    X = X.values.astype('float32')
    return predictor(X)


def predictor(data):
    print(data)
    data = x_scaler.transform(data)
    data = data.reshape((data.shape[0], 1, data.shape[1]))
    with graph.as_default():
        yhat = model.predict(data)
    inv_yhat = y_scaler.inverse_transform(yhat)
    print(inv_yhat)
    return str(inv_yhat[0])


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_models()
    port = 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
