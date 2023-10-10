#!/usr/bin/env python
"""
Run keras model for prediction using dataset provided

Usage:
-----
./predictor.py

"""

import glob

import h5py
import numpy as np
from keras import backend as K
from stringato import extract_floats
from tensorflow import keras
from tensorflow.keras import Sequential


def load_data() -> tuple:
    """
    Load and organize dataset into inputs and outputs lists

    :returns: input [ndarray], output [ndarray], shape [tuple]

    Notes:
    ------
    Final returned ndarrays are shaped similar to `outputs`

    Commented code:
    ---------------
    img = (img>0).astype(float)
    """
    files = glob.glob("../data/dataset*")
    inputs, outputs = [], []
    for file in files:
        tumble = float(extract_floats(file)[0])
        with h5py.File(file, "r") as fin:
            for key in fin.keys():
                img = fin[key][:].astype(np.float32)
                img /= img.max()
                img = img.reshape((img.shape[0], img.shape[1], 1))
                shape = img.shape
                inputs.append(img)
                outputs.append(tumble)
    order = np.arange(len(outputs)).astype(int)
    order = np.random.permutation(order)
    return np.array(inputs)[order], np.array(outputs)[order], shape


def main():
    """
    Train model with keras

    Notes:
    -----
    - Prints to stdout various things:
        - shape of data
        - shape of training data (cut to variable last)
        - results of model.evaluate()
        - prediction

    """
    x_arr, y_arr, shape = load_data()
    last = 1000
    x_train, y_train = x_arr[:-last], y_arr[:-last]
    x_val, y_val = x_arr[-last:], y_arr[-last:]
    print("Data shape is", shape)
    print("Training_data shape is", x_train.shape)
    model = keras.models.Sequential(
        [
            keras.layers.Conv2D(
                filters=5,
                kernel_size=(7, 7),
                strides=(4, 4),
                activation="relu",
                input_shape=shape,
            ),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(loss="mean_absolute_error", optimizer="SGD")
    model.fit(
        x_train,
        y_train,  # initial_epoch=3,
        epochs=20,
        verbose=True,
        batch_size=64,
        validation_data=(x_val, y_val),
    )
    results = model.evaluate(x_val, y_val, batch_size=64)
    print("Test loss, Test acc from evaluating test data:", results)
    prediction = model.predict(x_val[:10])
    print("Prediction")
    for arr, val in zip(prediction, y_val):
        print(arr[0], val)


if __name__ == "__main__":
    main()
