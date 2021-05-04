"""
All functions in this module directly manipulate NumPy arrays. This is not
scalable to large arrays.
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Normalization, RandomTranslation, RandomFlip

dataset_dict = {
    "mnist": tf.keras.datasets.mnist,
    "cifar10": tf.keras.datasets.cifar10,
    "cifar100": tf.keras.datasets.cifar100
}


def datasets():
    """
    :return: A list of the supported dataset names.
    """
    return sorted(dataset_dict.keys())


def input_shape(dataset):
    """
    :param dataset: Name of dataset.
    :return: Shape of input data.
    """
    sizes = {
        "mnist": (28, 28, 1),
        "cifar10": (32, 32, 3),
        "cifar100": (32, 32, 3)
    }
    return sizes[dataset]


def num_classes(dataset):
    """
    :param dataset: Name of dataset.
    :return: Number of output classes.
    """
    classes = {
        "mnist": 10,
        "cifar10": 10,
        "cifar100": 100
    }
    return classes[dataset]


def augmentations(dataset):
    """
    :param dataset: Name of dataset.
    :return: Default data augmentations to apply during training.
    """
    augments = tf.keras.Sequential([
        RandomTranslation(0.1, 0.1)
    ])
    if dataset != "mnist":
        augments.add(RandomFlip("horizontal"))
    if "cifar" in dataset:
        augments.add(Normalization(mean=[0.4914, 0.4822, 0.4465],
                                   variance=[0.04093, 0.03976, 0.04040]))

    return augments


def load_data(dataset):
    """
    :param dataset: Name of dataset.
    :return: NumPy tuples (train_data, train_labels), (test_data, test_labels)
    """
    (train_x, train_y), (test_x, test_y) = dataset_dict[dataset].load_data()

    # MNIST comes without a channels dimension, and Keras layers don't like it.
    if dataset == "mnist":
        train_x = np.expand_dims(train_x, 3)
        test_x = np.expand_dims(test_x, 3)

    # By default, MNIST's labels are 1D, but CIFAR's are 2D. Make them all 1D.
    train_y = train_y.reshape([-1])
    test_y = test_y.reshape([-1])

    # Bring all input values down to the range [0, 1].
    return (train_x / 255, train_y), (test_x / 255, test_y)


def split_data(data, fractions=None, classes=None):
    """
    Split a dataset into multiple subsets.

    :param data: Tuple of NumPy arrays (inputs, labels).
    :param fractions: Optional. List of fractions corresponding to subset sizes.
                      e.g. fractions=[0.8, 0.2]
                      Data will be shuffled to ensure a fair distribution.
    :param classes: Optional. List of lists of labels. Data whose label is not
                    listed is discarded.
                    e.g. [[0,1], [2,3], [4,5,6,7], [8,9]]
    :return: list of tuples (inputs, labels), split as requested.
    """
    if fractions is not None and classes is not None:
        raise RuntimeError("split_data can't split by fractions and classes "
                           "simultaneously")

    inputs, labels = data

    if fractions is not None:
        # Shuffle data first to ensure classes are distributed evenly.
        permutation = np.random.permutation(len(inputs))
        inputs = inputs[permutation]
        labels = labels[permutation]

        # Convert fractions of datasets to indices in the array.
        cumulative = np.cumsum(fractions)
        indices = [int(fraction * len(inputs)) for fraction in cumulative]
        inputs = np.array_split(inputs, indices)
        labels = np.array_split(labels, indices)

        # If the provided fractions sum to 1, NumPy will give us an empty
        # partition at the end. Drop this.
        data = list(zip(inputs[:len(fractions)], labels[:len(fractions)]))
    elif classes is not None:
        masks = [np.isin(labels, class_list) for class_list in classes]
        inputs = [inputs[mask] for mask in masks]
        labels = [labels[mask] for mask in masks]

        data = list(zip(inputs, labels))

    return data


def merge_data(data):
    """
    Merge multiple datasets into a single one.

    :param data: List of tuples of NumPy arrays (inputs, labels).
    :return: A single tuple of NumPy arrays (inputs, labels).
    """
    inputs = [dataset[0] for dataset in data]
    labels = [dataset[1] for dataset in data]

    return np.concatenate(inputs), np.concatenate(labels)


def permute_pixels(data, permutations, permutation=None):
    """
    Permute pixels of image data. The same permutation is applied to all images.

    :param data: Dataset to permute, a tuple of (images, labels).
    :param permutations: Number of permutations to generate.
    :param permutation: Permutations to use. Generated randomly if None.
    :return: Permuted data, and a list of used permutations.
    """
    # Use the original data as the first permutation.
    result = [data]
    images, labels = data
    shape = images.shape
    pixels = shape[1]*shape[2]

    # Assuming dimensions are (batch, height, width, [channels])
    flattened = np.reshape(images, [shape[0], pixels, -1])

    if permutation is None:
        permutation = []
        for _ in range(1, permutations):
            permutation.append(np.random.permutation(pixels))

    assert len(permutation) == (permutations - 1)

    for p in permutation:
        permuted = flattened.copy()[:, p]
        permuted = np.reshape(permuted, shape)
        result.append((permuted, labels))

    return result, permutation
