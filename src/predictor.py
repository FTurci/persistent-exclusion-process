import numpy as np
from tensorflow.keras import Sequential
from tensorflow import keras

def image(k,shape):
    bg = np.random.uniform(0,1,size=shape)
    mid = (np.array(shape)/2).astype(int)
    bg[mid]+=k
    normed = bg/bg.max()
    return normed


repeat = 100
maxk = 20
shape = (32,32,1)

training_data = []
result_data = []
for k in range(1,maxk+1):
    for r in range (repeat):
        training_data.append( image(k,shape))
        result_data.append(k)
training_data = np.array(training_data)
result_data = np.array(result_data)

filters = 1
kernel_size = (3,3)
model = Sequential([
    keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        activation=None,
        input_shape = shape,
        padding="same",

        data_format="channels_last"
        ),
    keras.layers.GlobalAveragePooling2D(),

    ]

    )

model.compile(loss="mean_squared_error", optimizer="SGD")
model.fit(training_data, result_data, epochs=20, verbose=0)

# predict

prediction = model.predict(
    np.array([image(11,shape)])
    )
print(prediction)