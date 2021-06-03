import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import multiprocessing as mp
import ray

ray.init()


@ray.remote
def nn_predictions(model, objects):
    results = []
    for obj in objects:
        res = model.predict(obj.reshape(1, -1))
        results.append(np.argmax(res))
    print(results)
    return results


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Training data shape: ", x_train.shape)
print("Test data shape", x_test.shape)

image_vector_size = 28 * 28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)

num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
print("First 5 training lables as one-hot encoded vectors:\n", y_train[:5])

image_size = 784

model = tf.keras.models.Sequential()

# The input layer requires the special input_shape parameter which should match
# the shape of our training data.
model.add(tf.keras.layers.Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))
# model.summary()

print("x_shape: ", x_train.shape)

model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=8, epochs=1, verbose=False, validation_split=.1)
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)

num_processes = 5
jobs = []

# nn_predictions(model, x_test[0:5])

for i in range(num_processes):
    process = mp.Process(name=f'background_process {i}', target=nn_predictions, args=(model, x_test[i * 5:(i + 1) * 5]))
    process.daemon = False
    jobs.append(process)
    process.start()

# futures = [nn_predictions.remote(model, x_test[i*5:(i+1)*5]) for i in range(num_processes)]
# print(ray.get(futures))
