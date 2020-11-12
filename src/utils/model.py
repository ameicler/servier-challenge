import os
import time
import tempfile
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.models import load_model, Sequential
from keras.layers import Dense, Embedding, BatchNormalization, Dropout, Conv1D, Flatten, Input
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.applications import VGG16

from rdkit.Chem.Draw import MolToFile
from rdkit.Chem import MolFromSmiles

from utils.data import load_and_prepare_data
from utils.feature_extractor import fingerprint_features, generate_all_images


def create_model_1(input_length=2048):
    model = Sequential()
    model.add(Embedding(1024, 50, input_length=input_length))
    model.add(Conv1D(192, 10, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(192, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 1, activation='sigmoid'))
    print(model.summary())
    return model


def create_model_2(target_size=(224, 224)):
    img_width, img_height = target_size
    input_shape = (img_width, img_height, 3)
    input_tensor = Input(shape=input_shape)

    # build the VGG16 network
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
    print('Model VGG16 loaded.')

    #build a classifier model to put on top of the CNN
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    # add the model on top of the convolutional base
    model = Sequential()
    for l in base_model.layers:
        model.add(l)

    # concatenate VGG16 and top model
    model.add(top_model)

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable = False

    return model


def train_model(data_dir="../data", models_dir="../models", model_name="2",
    epochs=2, batch_size=32, target_size=(224, 224)):

    print("Loading data in order to train model")
    X_train, X_test, y_train, y_test = load_and_prepare_data(data_dir, model_name)

    assert model_name in ["1", "2"], print("Argument `model` must be either 1 or 2")

    if model_name == "1":
        print("Creating model 1")
        model = create_model_1(input_length = X_train.shape[1])
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
        print("Training model 1")
        history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size,
            validation_data = (X_test, y_test))
    elif model_name == "2":
        print("Creating model 2")
        model = create_model_2(target_size=target_size)
        img_dir = os.path.join(data_dir, "smile_images")
        train_data_dir = os.path.join(img_dir, "train")
        validation_data_dir = os.path.join(img_dir, "test")

        if not(os.path.isdir(train_data_dir)):
            generate_all_images(X_train, y_train, X_test, y_test, img_dir,
                target_size=target_size)
        else:
            print("Image directory exists - no need to generate images")

        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

        train_datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True)
        test_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='binary')
        validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='binary')

        # fine-tune the model
        print("Training model 2")
        model.fit_generator(
            train_generator,
            steps_per_epoch=len(y_train) // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=len(y_test) // batch_size)

    print("Finished model training. Saving h5")
    model_path = os.path.join(models_dir, "model_{}".format(model_name) + ".h5")
    if not(os.path.isdir(models_dir)):
        os.mkdir(models_dir)
    model.save(model_path)
    print("Model saved @ {}".format(model_path))
    return


def evaluate_model(data_dir="../data", model_path="../models/model_2.h5",
    model_name="2", target_size=(224, 224), batch_size=32):
    print("Evaluating model")
    model = load_model(model_path)
    X_train, X_test, y_train, y_test = load_and_prepare_data(data_dir)

    assert model_name in ["1", "2"], print("Argument `model` must be either 1 or 2")

    if model_name == "1":
        print("Evaluating model 1")
        y_pred_test = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred_test)
        print("Model 1 accuracy on test set: {} ({}% of positive preds)".format(acc,
            100 * np.sum(y_pred_test)/len(y_pred_test)))
    elif model_name == "2":
        print("Evaluating model 2")
        test_data_dir = os.path.join(data_dir, "smile_images/test")
        test_datagen = ImageDataGenerator()
        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='binary')
        test_loss, test_acc = model.evaluate(x=test_generator,
                                            steps=len(y_test) // batch_size,
                                            verbose =1)
        print("Model 2 accuracy on test set: {}".format(test_acc))
    print("Finished model evaluation")
    return


def load_models(model_1_path="../models/model_1.h5",
    model_2_path="../models/model_2.h5"):
    print("Loading models 1 and 2")
    # load the trained Keras model
    global model_1, model_2
    model_1 = load_model(model_1_path)
    model_2 = load_model(model_2_path)
    print("Models 1 and 2 loaded")
    return model_1, model_2


def smile_to_pred(smile, model_1, model_2, model_name="1"):
    smile_str = str(smile)
    model_name = str(model_name)
    print("Retrieving prediction for smile={} and model_name={}".format(
        smile_str, model_name))

    assert model_name in ["1", "2"], print("Argument `model` must be either 1 or 2")

    if model_name == "1":
        print("Prediction based on model 1 (fingerprint features)")
        input_arr = np.array(fingerprint_features(smile_str))
        input_arr = input_arr.reshape(1, -1)
        preds = model_1.predict(input_arr)
    elif model_name == "2":
        print("Prediction based on model 2 (Mol image)")
        fp = tempfile.NamedTemporaryFile(suffix=".png")
        molsmile = MolFromSmiles(smile_str)
        MolToFile(molsmile, fp.name, size = (224, 224))
        img = load_img(fp.name, target_size=(224, 224))
        input_arr = img_to_array(img)
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        preds = model_2.predict(input_arr)
    print("Predictions results: {}".format(preds))
    return preds
