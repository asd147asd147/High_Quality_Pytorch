import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import time
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Reshape, Input
from tensorflow.keras import Model, Sequential
import os

EPOCHS = 50
BATCH_SIZE = 64

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

def gen(img_data, label_data):
    for img, label in zip(img_data, label_data):
        yield (img, label)

train_loader = tf.data.Dataset.from_generator(gen,
                                              (tf.float32, tf.int64),
                                              ((28,28),(10)),
                                              args=(x_train, y_train))

test_loader = tf.data.Dataset.from_generator(gen,
                                              (tf.float32, tf.int64),
                                              ((28,28),(10)),
                                              args=(x_test, y_test))

train_loader = train_loader.shuffle(150).batch(BATCH_SIZE)
test_loader = test_loader.shuffle(150).batch(BATCH_SIZE)

model = Sequential([
    Input((28,28,1)),
    Flatten(),
    Dense(784, activation='relu'),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax'),
])
model.build((28,28))
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
t = time.time()
model.fit_generator(
    train_loader,
    epochs = EPOCHS,
    validation_data=test_loader
)
print(time.time() - t)