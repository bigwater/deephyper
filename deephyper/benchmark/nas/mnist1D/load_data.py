import os
<<<<<<< HEAD
# from tensorflow.examples.tutorials.mnist import input_data
=======

import tensorflow as tf
>>>>>>> a5800cd7a186196aa159387cd8621a05852c1445
import numpy as np
import tensorflow as tf
import keras
from keras.datasets import mnist

from tensorflow.keras.utils import to_categorical

HERE = os.path.dirname(os.path.abspath(__file__))

np.random.seed(2018)


def load_data(prop=0.1):
    """Loads the MNIST dataset.
    Returns Tuple of Numpy arrays: `(train_X, train_y), (valid_X, valid_y)`.
    """

    dest = os.path.join('/home/hyliu/work/datasets/mnistmlp', 'mnist.npz')
    print('load data from : ', dest)

    (x_train, y_train), (x_test, y_test) = mnist.load_data(dest)

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print(f'train_X shape: {np.shape(x_train)}')
    print(f'train_y shape: {np.shape(y_train)}')
    print(f'valid_X shape: {np.shape(x_test)}')
    print(f'valid_y shape: {np.shape(y_test)}')

    return (x_train, y_train), (x_test, y_test)



if __name__ == "__main__":
    load_data()
