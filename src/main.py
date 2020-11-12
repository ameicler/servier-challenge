import os
import numpy as np
import pandas as pd
import flask

from utils.model import load_models, smile_to_pred

# Initialize our Flask application and the Keras model
app = flask.Flask(__name__)

model_1, model_2 = None, None

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        # Read the molecule smile and model_name from query
        smile_str = flask.request.form.get("smile")
        model_name = str(flask.request.form.get("model_name", "2"))

        if smile_str is not None:
            preds = smile_to_pred(smile_str, model_1, model_2,
                model_name=model_name)
            data["P1_pred"] = str(preds[0][0])
            data["success"] = True
        else:
            print("Failed to retrieve SMILE string from request")
            pass

    print("Prediction results : {}".format(data))
    # Return the data dictionary as a JSON response
    return flask.jsonify(data)


# If this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print("Loading models and starting Flask server...")
    model_1, model_2 = load_models()
    app.run(host='0.0.0.0')
