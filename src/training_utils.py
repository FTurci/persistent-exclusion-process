import contextlib
import glob
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy.stats import pearsonr
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    MaxPooling2D,
    ReLU,
)
from tensorflow.keras.models import Sequential

from src.plot_utils import get_plot_configs


def extract_floats(string):
    return re.findall(r"[-+]?\d*\.\d+|\d+", string)


def data_load(
    alphas=np.logspace(-6, -1, 10, base=2),
    densities=np.arange(0, 0.55, 0.05),
    orientation=True,
):
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
    v = prediction.T[0]

    plot_configs = get_plot_configs()
    plot_configs['axes.facecolor'] = [.96,.96,.96,1]
    plot_configs['figure.facecolor'] = [.98,.98,.98,1]
    plt.rcParams.update(plot_configs)
    sns.set(rc=plot_configs)

    df = pd.DataFrame()
    df.insert(0, "predicted", np.abs(v - y_val))
    df.insert(1, "actual", y_val)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.violinplot(
        ax=ax,
        data=df,
        x="actual",
        y="predicted",
        color="w",
        alpha=0.7,
        density_norm="width",
        linewidth=1,
        inner="box",
        inner_kws={"box_width": 4, "color": "0.2"},
    )
    ax.set(xlabel=r"Tumbling rates, $\alpha$", ylabel=r"Absolute error, $|y_p - y_a|$")

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

    model.add(Conv2D(filters=4, kernel_size=(5, 5), padding="same"))
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

def get_huber_loss(y_true,y_pred,clip_delta=1.0):
    '''
    Returns huber loss to be used as metric in convolutional neural network.
    
    y_true: value of y that the model aims to predict
    y_pred: value of y that the model actually predicts
    clip_delta: threshold beyond which huber function becomes linear
    '''
    error = y_true - y_pred
    cond  = tf.keras.backend.abs(error) < clip_delta
    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
    return tf.where(cond, squared_loss, linear_loss)