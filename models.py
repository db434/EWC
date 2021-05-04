import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Conv2D, Dense, Flatten
from tensorflow.keras.activations import relu, softmax, tanh
from tensorflow.python.keras.layers import Dropout, BatchNormalization

import datasets


class MLP(tf.keras.Model):
    """Simple multi-layer perceptron for MNIST."""

    def __init__(self, dataset):
        super(MLP, self).__init__()

        self.dataset = "mnist" if dataset is None else dataset
        self.model = tf.keras.Sequential([
            datasets.augmentations(self.dataset),
            Flatten(input_shape=datasets.input_shape(self.dataset)),
            Dense(100, activation=relu),
            Dense(100, activation=relu),
            Dense(datasets.num_classes(self.dataset), activation=softmax)
        ])

    def get_config(self):
        # No configuration options by default.
        return {}

    def call(self, inputs, training=None, mask=None):
        return self.model.call(inputs, training, mask)


class LeNet5(tf.keras.Model):
    """
    Simple CNN for MNIST.

    This doesn't quite match the original because it doesn't contain sparse
    connections in the second convolution layer, and other minor differences.

    https://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
    """

    def __init__(self, dataset):
        super(LeNet5, self).__init__()

        self.dataset = "mnist" if dataset is None else dataset
        self.model = tf.keras.Sequential([
            datasets.augmentations(self.dataset),
            Conv2D(6, kernel_size=5, activation=tanh,
                   padding="same",  # Inputs are smaller than originals, so pad
                   input_shape=datasets.input_shape(self.dataset)),
            AveragePooling2D(pool_size=2),
            Conv2D(16, kernel_size=5, activation=tanh),
            AveragePooling2D(pool_size=2),
            Conv2D(120, kernel_size=5, activation=tanh),
            Flatten(),
            Dense(84, activation=tanh),
            Dense(datasets.num_classes(self.dataset), activation=softmax)
        ])

    def get_config(self):
        # No configuration options by default.
        return {}

    def call(self, inputs, training=None, mask=None):
        return self.model.call(inputs, training, mask)


class CifarNet(tf.keras.Model):
    """
    Simple CNN for CIFAR-10 and CIFAR-100.
    """

    def __init__(self, dataset):
        super(CifarNet, self).__init__()

        self.dataset = "cifar10" if dataset is None else dataset
        self.num_classes = datasets.num_classes(self.dataset)
        self.model = tf.keras.Sequential([
            datasets.augmentations(self.dataset),
            Conv2D(64, kernel_size=3, padding="same", activation=relu,
                   input_shape=datasets.input_shape(self.dataset)),
            BatchNormalization(),
            Conv2D(64, kernel_size=3, padding="same", activation=relu),
            BatchNormalization(),
            Conv2D(128, kernel_size=3, padding="same", activation=relu,
                   strides=2),
            BatchNormalization(),
            Conv2D(128, kernel_size=3, padding="same", activation=relu),
            BatchNormalization(),
            Dropout(rate=0.5),
            Conv2D(128, kernel_size=3, padding="same", activation=relu),
            BatchNormalization(),
            Conv2D(192, kernel_size=3, padding="same", activation=relu,
                   strides=2),
            BatchNormalization(),
            Conv2D(192, kernel_size=3, padding="same", activation=relu),
            BatchNormalization(),
            Dropout(rate=0.5),
            Conv2D(192, kernel_size=3, padding="same", activation=relu),
            BatchNormalization(),
            AveragePooling2D(pool_size=8),
            Flatten(),
            Dense(self.num_classes, activation=softmax)
        ])

    def get_config(self):
        return {"num_classes": self.num_classes}

    def call(self, inputs, training=None, mask=None):
        return self.model.call(inputs, training, mask)


model_dict = {
    "cifarnet": CifarNet,
    "lenet": LeNet5,
    "mlp": MLP
}


def models():
    """
    :return: A list of the available model names.
    """
    return sorted(model_dict.keys())


def get_model(name, dataset=None):
    """
    :param name: Name of a machine learning model.
    :param dataset: Name of dataset to be used as input for the model.
    :return: An instance of the named model.
    """
    return model_dict[name](dataset)
