#!/usr/bin/env python
"""
Run keras model for prediction on minimal test data

Usage:
-----
./predictor_minimal_test.py

"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential


def image(k: int, shape: tuple) -> np.ndarray:
    """Setup minimal test data

    :param k: TODO not clear what this is [int]
    :param shape: shape of ndarray [tuple]
    :returns: normalized lattice with random uniform values and populated centre

    Explanation:
    -----------
    - Generate lattice with dimension `shape` [tuple]
    - Fill lattice with values between [0,1] following a uniform distribution
    - At the centre of the array, add `k` to the value
    - Normalize the lattice

    """
    b_g = np.random.uniform(0, 1, size=shape)
    mid = (np.array(shape) / 2).astype(int)
    b_g[mid] += k
    normed = b_g / b_g.max()
    return normed


def main():
    """
    Run a network to predict based on test lattice and print the prediction to stdout

    Explanation:
    -----------
    - Iterate through a range of `k`
    - Repeat each `k` 100 times (set by `repeat`)
    - Train the model and output results
    """
    repeat = 100
    maxk = 20
    shape = (32, 32, 1)
    training_data = []
    result_data = []
    for k in range(1, maxk + 1):
        for _ in range(repeat):
            training_data.append(image(k, shape))
            result_data.append(k)
    training_data = np.array(training_data)
    result_data = np.array(result_data)
    filters = 1
    kernel_size = (3, 3)
    model = Sequential(
        [
            keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                activation=None,
                input_shape=shape,
                padding="same",
                data_format="channels_last",
            ),
            keras.layers.GlobalAveragePooling2D(),
        ]
    )
    model.compile(loss="mean_squared_error", optimizer="SGD")
    model.fit(training_data, result_data, epochs=20, verbose=0)
    prediction = model.predict(np.array([image(11, shape)]))
    print(prediction)
