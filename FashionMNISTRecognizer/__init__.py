import tensorflow as tf
from mlxtend.data import loadlocal_mnist
import math
import numpy as np
import matplotlib.pyplot as plt
import logging

def normalize(image):
    image = tf.cast(image, tf.float32)
    image /= 255
    return np.array(image).reshape((28, 28))


logger = tf.get_logger()
logger.setLevel(logging.ERROR)

train_x, train_labels = loadlocal_mnist(
    images_path='../resources/train-images-ubyte',
    labels_path='../resources/train-labels-ubyte')

test_x, test_labels = loadlocal_mnist(
    images_path='../resources/t10k-images-ubyte',
    labels_path='../resources/t10k-labels-ubyte')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']


# PREPROCESSING THE DATA
train_dataset = np.array([normalize(array) for array in train_x])
test_dataset = np.array([normalize(array) for array in test_x])

# PLOTTING THE DATA
# image = train_dataset[0].reshape((28, 28))
# plt.figure()
# plt.imshow(image, cmap=plt.cm.binary)
# plt.colorbar()
# plt.grid(False)
# plt.show()

# BUILDING THE DATA
layer0 = tf.keras.layers.Flatten(input_shape=(28, 28))
layer1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
layer2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
model = tf.keras.Sequential([layer0, layer1, layer2])

# COMPILING THE MODEL
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
)

# TRAINING THE MODEL
model.fit(
    x=train_dataset,
    y=train_labels,
    epochs=5)

# TESTING THE MODEL
success_counter = 0
predictions = model.predict(test_dataset)
for prediction, expected in zip(predictions, test_labels):
    result = np.argmax(prediction)
    success = (result == expected)
    if success:
        success_counter += 1
    print(result, expected, success)

print('\n\n\naccuracy = {}'.format(success_counter * 100 / len(test_labels)))
