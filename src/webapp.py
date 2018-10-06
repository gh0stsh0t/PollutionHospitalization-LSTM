# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
from sklearn.externals import joblib
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
from keras import backend as kk

# initialize our Flask application and the Keras model
app = Flask(__name__)


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
        from keras.models import model_from_json, model_to_json
        with open('models/lstm.json', 'r') as file:
            json_data = json.load(file)
            for item in json_data:
                # change below
                if item['ParameterKey'] in ["Shell","Type"]:
                    item['ParameterKey'] = "new value"
        with open('/models/lstm.json', 'w') as file:
            json.dump(json_data, file, indent=2)
        model = model_from_json('models/lstm.json')
        model.load_weights('models/lstm_weights.h5')
    x_scaler = joblib.load('models/x_scaler.save')
    y_scaler = joblib.load('models/x_scaler.save')


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    params = request.json
    if params is None:
        params = request.args

    # if parameters are found, return a prediction
    if params is not None:
        x = pd.DataFrame.from_dict(params, orient='index').transpose()
        with graph.as_default():
            data["prediction"] = str(model.predict(x)[0][0])
            data["success"] = True

    # return a response in json format
    return jsonify(data)


@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader_form', methods=['POST'])
def read_submission():
    if request.method == 'POST':
        X = []
        for station in ['02', '07', '11', '14']:
            pm10, so2, no2, o3 = request.form['pm10_'+station]
            X.append(pm10, so2, no2, o3)
        X = np.array(X).astype('float32')
        return predictor(X)

@app.route('/uploader', methods=['GET', 'POST'])
def uploadr_file():
    if request.method == 'POST':
        f = request.files['file']
        X = pd.read_csv(f)
        X.fillna(-1, inplace=True)
        X = X.drop("date", axis=1)
        X = X.values.astype('float32')
        return predictor(X)

def predictor(data):
        data = x_scaler.fit_transform(data)
        data = data.reshape((data.shape[0], 1, data.shape[1]))
        yhat = model.predict(data)
        inv_yhat = y_scaler.inverse_transform(yhat)
        return inv_yhat


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_models()
    app.run()
