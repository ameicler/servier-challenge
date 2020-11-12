# Servier ML Technical Test 🧪

![](https://lh3.googleusercontent.com/proxy/GNyw5eugrrCuf0iSW8aCL6VOj5y7XcoAx0JOICy0lNclZnIU_nqeYuzPdFyz3SBeRBxQDpxD8nbDxtxUsOljOzq4CNYeqU1BfpwubvAvRvajOQNTkgOyC3-iFpfbwkinFD_EcHhNIGU5xP-LLNP2WeLT_0T6QBucHGZIdOv7D9V9nRe5l47dwLmrk10)

---

## I. Context

The prediction of a drug molecule properties plays an important role in the drug design process. The molecule properties are the cause of failure for 60% of all drugs in the clinical phases. A multi parameters optimization using machine learning methods can be used to choose an optimized molecule to be subjected to more extensive studies and to avoid any clinical phase failure.

We developed two models able to predict the property P1 of a molecule given its SMILE representation.

The proposed application can be used for **retrieving a model's prediction**, for **evaluating a given model** or for **training a new model from scratch**.

## II. Models

### A/ Model 1



### B/ Model 2



## III. Usage

### A/ Setup

In order to install and use the application, you can simply build it using Docker.  🐳

``` bash
docker build . -t servier
```

### B/ Flask server

Once you have successfully built the docker image, you can run the Flask server and mount the `data` and `models` folder as volumes while exposing the port 5000:

``` bash
docker run -v $(pwd)/data:/data -v $(pwd)/models:/models -p 5000:5000 servier
```

The Flask server will then be running and you should be able to POST a request to the `predict` route. For example:

``` bash
# Retrieving a prediction for a given smile and using model 2
curl -d "smile=Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C&model_name=2" -X POST http://localhost:5000/predict
```

You can also enter and inspect the container interactively:
``` bash
docker run -v $(pwd)/data:/data -v $(pwd)/models:/models -p 5000:5000 -it --entrypoint /bin/bash servier
```

### C/ Commands

One can also use `servier` package (for example in interactive Docker container) and leverage it for training/evaluating/retrieving predictions:

``` bash
Usage: servier [OPTIONS] COMMAND [ARGS]...

  servier - Property molecule prediction and model management

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  evaluate  Evaluate model.
  predict   Predict P1 property based on molecule fingerprint.
  train     Train model from scratch.
```

More specifically you can:

- Train model
``` bash
Usage: servier train [OPTIONS]

  Train model from scratch.

  Options:
  --data_dir   TEXT  Path to the data directory
  --model_name TEXT  Model to be trained
  --help             Show this message and exit.
```

- Evaluate model
``` bash
Usage: servier evaluate [OPTIONS]

  Evaluate model.

  Options:
    --data_dir   TEXT  Path to the data directory
    --model_path TEXT  Path of the model to be evaluated
    --model_name TEXT  Model to be evaluated
    --help             Show this message and exit.
```

- Retrieve prediction
``` bash
Usage: servier predict [OPTIONS] SMILE

  Predict P1 property based on molecule fingerprint.

  Options:
    --model_1_path TEXT  Path of the model 1 checkpoint
    --model_2_path TEXT  Path of the model 2 checkpoint
    --model_name TEXT    Model to be used for retrieving prediction
    --help               Show this message and exit.
```

## Conclusion and next steps
