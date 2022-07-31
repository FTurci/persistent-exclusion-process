import numpy as np
from tensorflow.keras import Sequential
from tensorflow import keras
import h5py
import glob
from stringato import extract_floats

from keras import backend as K



def data_load():
    files = glob.glob("../data/dataset*")

    inputs,outputs = [],[]
    for f in files:
        tumble = float(extract_floats(f)[0])
        
        with h5py.File(f, "r") as fin:
            for key in fin.keys():
                img = fin[key][:].astype(np.float32)
                img /= img.max()
                # img = (img>0).astype(float)
                img = img.reshape((img.shape[0], img.shape[1],1))
                shape = img.shape
                inputs.append(img)
                outputs.append(tumble)

    order = np.arange(len(outputs)).astype(int)
    order=np.random.permutation(order)
    return np.array(inputs)[order],np.array(outputs)[order],shape


x,y,shape = data_load()
print(y)
# print (np.unique(result_data))
last = 1000
x_train, y_train = x[:-last], y[:-last]
x_val,y_val = x[-last:],y[-last:]


print("data shape is", shape)
print( "training_data shape is", x_train.shape)
filters = 1
kernel_size = (3,3)
# model = Sequential([
#     keras.layers.Conv2D(
#         filters=filters,
#         kernel_size=kernel_size,
#         input_shape = shape,
#         padding="same",
#         data_format="channels_last",
#         use_bias=True,
#         activation=None
#         ),

#     keras.layers.Flatten(),

#     # keras.layers.Dense(32, activation="relu"),
#     # keras.layers.Dense(4, activation="relu"),
#     keras.layers.Dense(1, activation="linear"),
#     # keras.layers.Dense(1)
#     # keras.layers.GlobalAveragePooling2D(),
#     ]

#     )


model = keras.models.Sequential([
    keras.layers.Conv2D(filters=5, kernel_size=(7,7), strides=(4,4), activation='relu', input_shape=shape),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    # keras.layers.Conv2D(filters=1, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    # keras.layers.BatchNormalization(),
    # keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    # keras.layers.Conv2D(filters=10, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    # keras.layers.BatchNormalization(),
    # keras.layers.Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    # keras.layers.BatchNormalization(),
    # keras.layers.Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    # keras.layers.BatchNormalization(),
    # keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    # keras.layers.Dense(4096, activation='relu'),
    # keras.layers.Dropout(0.5),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='linear')
])
# for layer in model.layers:
#     print(layer.output_shape)
# model.compile(loss="mean_squared_error", optimizer="SGD")
model.compile(loss='mean_absolute_error', optimizer='SGD')
# 
# model.fit(x_train, y_train, epochs=3, verbose=True)
# K.set_value(model.optimizer.learning_rate, 0.0008)
# 
model.fit(x_train, y_train, #initial_epoch=3,
    epochs=20, verbose=True, batch_size=64,validation_data=(x_val, y_val))


print("Evaluate on test data")
results = model.evaluate(x_val, y_val, batch_size=64)
print("test loss, test acc:", results)

# model.fit(training_data, result_data, initial_epoch=20,epochs=2000, verbose=True, batch_size=30)

# predict

# tests = glob.glob("../data/test/dataset_tumble*h5")
# for f in tests:
#     test_images = []
#     truth = extract_floats(f)[0]
#     with h5py.File(f, "r") as fin:
#         # print (fin.keys())
#         for k in fin.keys():
#             img = fin[k][:]
#             img = img.reshape((img.shape[0], img.shape[1],1))
#             test_images.append(img)


prediction = model.predict(
        x_val[:10]
        )
print("Prediction")
for p,v in zip(prediction,y_val):
    print(p[0],v)
#     print("Predicted tumble",prediction.mean(),prediction.std(), "truth ",truth)

# import matplotlib.pyplot as plt

# plt.hist(prediction.mean(axis=1))
# plt.show()