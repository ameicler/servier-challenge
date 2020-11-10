import os
import numpy as np
import pandas as pd
import flask

from src.utils.model import load_h5_model, smile_to_pred

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

model = None

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    if flask.request.method == "POST":
        # read the molecule smile
        smile_str = flask.request.form.get("smile")

        if smile_str is not None:
            preds = smile_to_pred(smile_str)
            print("preds={}".format(preds))
            #results = imagenet_utils.decode_predictions(preds)
            data["P1_pred"] = str(preds[0][0])
            # indicate that the request was a success
            data["success"] = True

    print("Prediction results : {}".format(data))
    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print("Loading model and starting Flask server...")
	model = load_h5_model(model_path="models/cnn_1d_1110.h5")
	app.run()
