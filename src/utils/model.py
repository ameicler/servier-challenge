import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Embedding, BatchNormalization, Dropout, Conv1D, Flatten

from src.utils.data import load_and_prepare_data
from src.utils.feature_extractor import fingerprint_features


def create_cnn_1d(input_length=2048):
    model = Sequential()
    model.add(Embedding(1024+1, 50, input_length=input_length))
    model.add(Conv1D(192, 10, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(192, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 1, activation='sigmoid'))
    print(model.summary())
    return model


def train_model(data_dir="data"):
    print("Training model")
    X_train, X_test, y_train, y_test = load_and_prepare_data(data_dir)
    model = create_cnn_1d(input_length = X_train.shape[1])
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs = 100, batch_size = 32,
        validation_data = (X_test, y_test))
    print("Finished model training")
    print("Saving h5")
    model_name = "cnn_1d_" + time.strftime("%Y%m%d")
    model.save(models_dir + model_name + '.h5')
    print("Model saved @ {}")
    return


def evaluate_model(data_dir="data", model_path="models/cnn_1d_1110.h5"):
    print("Evaluating model")
    model = load_h5_model(model_path)
    X_train, X_test, y_train, y_test = load_and_prepare_data(data_dir)
    y_pred_test = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    print("Model accuracy on test set: {} ({}% of positive preds)".format(acc,
        100 * np.sum(y_pred_test)/len(y_pred_test)))
    print("Finished model evaluation")
    return


def load_h5_model(model_path="models/cnn_1d_1110.h5"):
    # load the trained Keras model
    global model
    model = load_model(model_path)
    return model


def smile_to_pred(smile, model):
    smile_str = str(smile)
    print("smile passed as input: {}".format(smile_str))
    features_test = np.array(fingerprint_features(smile_str))
    features_test = features_test.reshape(1, -1)
    preds = model.predict(features_test)
    print("Predictions: {}".format(preds))
    return preds
