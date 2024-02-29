import contextlib
import tensorflow as tf

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


def make_net_starter(shape):
    model = Sequential()

    model.add(
        Conv2D(
            filters=3,
            kernel_size=(3, 3),
            padding="same",
            strides=(3, 3),
            activation="relu",
            input_shape=shape,
        )
    )
    model.add(Flatten())
    model.add(Dense(units=1, activation="linear"))


def make_net_one(shape):
    model = Sequential()

    model.add(
        Conv2D(
            filters=3,
            kernel_size=(3, 3),
            padding="same",
            strides=(3, 3),
            activation="relu",
            input_shape=shape,
        )
    )
    model.add(BatchNormalization())
    model.add(Conv2D(filters=3, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(3, 3)))

    # model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=6, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=6, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Dense(units=128, activation="relu"))

    with options({"layout_optimizer": False}):
        model.add(Dropout(0.2))
    model.add(Dense(units=10, activation="softmax"))

    model.add(Flatten())
    model.add(Dense(units=1, activation="linear"))
