#!/usr/bin/env python3

import contextlib
import glob
import re

import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
from cmcrameri import cm
from keras import backend as K
from scipy.stats import pearsonr
from tensorflow import keras
from tensorflow.keras.layers import (
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    LeakyReLU,
    MaxPooling2D,
    Normalization,
    ReLU,
)
from tensorflow.keras.models import Sequential


def extract_floats(string):
    return re.findall(r"[-+]?\d*\.\d+|\d+", string)


def data_load(alphas, densities, orientation=True):
    files = []
    for alp in alphas:
        for val in densities:
            files += glob.glob(f"no_roll_data/dataset_tumble_{alp:.3f}_{val}.h5")
    # print("Loaded in:", files)
    inputs, outputs = [], []
    for f in files:
        tumble = float(extract_floats(f)[0])
        with h5py.File(f, "r") as fin:
            count = 0
            for key in fin.keys():
                img = fin[key][:]
                if not orientation:
                    img[img > 0] = 1
                else:
                    img = img / 4
                img = img.reshape((img.shape[0], img.shape[1], 1))
                shape = img.shape
                inputs.append(img)
                outputs.append(tumble)
                # AUGMENTATION
                inputs.append(np.roll(img, (64, 64), axis=(0, 1)))
                inputs.append(np.roll(img, (120, 120), axis=(0, 1)))
                outputs.append(tumble)
                outputs.append(tumble)
                count += 1

    # Scramble the dataset
    order = np.arange(len(outputs)).astype(int)
    order = np.random.permutation(order)
    return np.array(inputs)[order], np.array(outputs)[order], shape


def split_dataset(x, y, last=2000):
    print("Number of unique alpha: ", len(np.unique(y)))
    print("Shape of x: ", np.shape(x))
    print("Shape of y: ", np.shape(y))

    x_train, y_train = x[:-last], y[:-last]
    x_val, y_val = x[-last:], y[-last:]

    print("Size of training data: ", len(x_train))
    print("Size of validation data: ", len(x_val))
    return x_train, y_train, x_val, y_val


def predict_and_plot(model, x_val, y_val):
    prediction = model.predict(x_val)
    bins = np.logspace(-6, -1, 10, base=2) * 0.85
    v = prediction.T[0]
    fig, ax = plt.subplots()
    ax.scatter(y_val, v, c="k", alpha=0.25)
    ax.scatter(np.unique(y_val), np.unique(y_val), marker="_", color="r", s=200)

    ax.set_xscale("log")
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(np.unique(y_val))

    ax.set_facecolor([0.98, 0.98, 0.98, 1])

    for val in bins:
        ax.axvline(val, alpha=0.05, c="k")

    ax.set_xlabel("Input turning rate")
    ax.set_ylabel("Predicted turning rate")

    std = []
    overlap = []
    accuracy = 1e-3
    for val in np.unique(y_val):
        v_mapped = v[np.where(y_val == val)]
        std.append(np.std(v_mapped))
        overlap.append(
            (val + accuracy >= np.min(v_mapped)) & (val - accuracy <= np.max(v_mapped))
        )

    print("Overlap ratio:", np.sum(overlap) / len(overlap))
    print("(Min, Max, Avg) STD:", np.min(std), np.max(std), np.mean(std))
    print("Pearson's correlation coeff: ", pearsonr(y_val, v).statistic)

    print("Overlap ratio:", np.sum(overlap) / len(overlap))
    print("(Min, Max, Avg) STD:", np.min(std), np.max(std), np.mean(std))
    print("Pearson's correlation coeff: ", pearsonr(y_val, v).statistic)


@contextlib.contextmanager
def options(options):
    old_opts = tf.config.optimizer.get_experimental_options()
    tf.config.optimizer.set_experimental_options(options)
    try:
        yield
    finally:
        tf.config.optimizer.set_experimental_options(old_opts)


def make_net(shape):
    model = Sequential()

    model.add(Conv2D(filters=3, kernel_size=(3, 3), padding="same", input_shape=shape))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(ReLU())
    model.add(BatchNormalization())

    model.add(Conv2D(filters=4, kernel_size=(4, 4), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(ReLU())
    model.add(BatchNormalization())

    model.add(Conv2D(filters=6, kernel_size=(5, 5), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(ReLU())
    model.add(BatchNormalization())

    model.add(GlobalAveragePooling2D())

    with options({"layout_optimizer": False}):
        model.add(Dropout(0.1))

    model.add(Dense(units=128, activation="relu"))

    with options({"layout_optimizer": False}):
        model.add(Dropout(0.1))

    model.add(Dense(units=3, activation="relu"))

    model.add(Flatten())
    model.add(Dense(units=1, activation="linear"))
    return model
