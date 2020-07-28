from mlxtend.data import loadlocal_mnist
import tensorflow
import numpy
import matplotlib.pyplot as plt
import logging


def normalize(image):
    image = tensorflow.cast(image, tensorflow.float32)
    image /= 255
    return numpy.array(image).reshape((28, 28))


logger = tensorflow.get_logger()
logger.setLevel(logging.ERROR)

train_inputs, train_labels = loadlocal_mnist(
    images_path='../resources/train-images-ubyte',
    labels_path='../resources/train-labels-ubyte')

test_inputs, test_labels = loadlocal_mnist(
    images_path='../resources/t10k-images-ubyte',
    labels_path='../resources/t10k-labels-ubyte')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

# PREPROCESSING THE DATA
train_dataset = numpy.array([normalize(array) for array in train_inputs])
test_dataset = numpy.array([normalize(array) for array in test_inputs])

### just to reduce calculations ###
train_dataset = train_dataset[:100]
train_labels = train_labels[:100]

# PLOTTING THE DATA
# image = train_dataset[0]
# plt.figure()
# plt.imshow(image, cmap=plt.cm.binary)
# plt.colorbar()
# plt.grid(False)
# plt.show()

# BUILDING THE MODEL
layer0 = tensorflow.keras.layers.Conv2D(
    32,
    (3, 3),
    padding='same',
    activation=tensorflow.nn.relu,
    input_shape=(28, 28, 1))
layer1 = tensorflow.keras.layers.MaxPooling2D((2, 2), strides=2)
layer2 = tensorflow.keras.layers.Conv2D(
    64,
    (3, 3),
    padding='same',
    activation=tensorflow.nn.relu)
layer3 = tensorflow.keras.layers.MaxPooling2D((2, 2), strides=2)
layer4 = tensorflow.keras.layers.Flatten()
layer5 = tensorflow.keras.layers.Dense(128, activation=tensorflow.nn.relu)
layer6 = tensorflow.keras.layers.Dense(10, activation=tensorflow.nn.softmax)

model = tensorflow.keras.Sequential([layer0, layer1, layer2, layer3, layer4, layer5, layer6])

model.compile(optimizer='adam',
              loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

n_batch = len(train_dataset)
train_dataset = numpy.expand_dims(train_dataset, -1)
model.fit(x=train_dataset, y=train_labels, epochs=10, batch_size=n_batch)

# TESTING
test_dataset = numpy.expand_dims(test_dataset, -1)
success_counter = 0
predictions = model.predict(test_dataset)
for prediction, expected in zip(predictions, test_labels):
    result = numpy.argmax(prediction)
    success = (result == expected)
    if success:
        success_counter += 1
    print(result, expected, success)

print('\n\n\naccuracy = {}'.format(success_counter * 100 / len(test_labels)))