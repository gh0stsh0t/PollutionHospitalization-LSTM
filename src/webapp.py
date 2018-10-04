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

# initialize our Flask application and the Keras model
app = Flask(__name__)


def load_models():
    global model
    global x_scaler
    global y_scaler
    global graph
    graph = tf.get_default_graph()
    model = load_model('models/lstm.h5')
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


@app.route('/upload')
def upload_file():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploadr_file():
    if request.method == 'POST':
        f = request.files['file']
        X = pd.read_csv(f)
        X.fillna(-1, inplace=True)
        X = X.drop("date", axis=1)
        X = X.values.astype('float32')
        X = x_scaler.fit_transform(X)
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        yhat = model.predict(X)
        inv_yhat = y_scaler.inverse_transform(yhat)
        return inv_yhat

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_models()
    app.run()
