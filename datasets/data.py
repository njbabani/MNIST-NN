# -*- coding: utf-8 -*-
'''
Module for loading, normalising and flattening MNIST dataset

Functions:
    load_mnist_data(verbose=True): Loads the MNIST dataset, prints data shapes
    flatten_data(data): Flattens the images to a 2D format
    normalise_data(data): Normalises the dataset to a range of [0, 1]

Example:
    from datasets.data import load_mnist_data
    x_train, y_train, x_test, y_test = load_mnist_data(verbose=True)
'''

from tensorflow.keras.datasets import mnist


def normalise_data(data):
    '''
    Normalise dataset (i.e. for pixels: [0, 255] to [0, 1])

    Args:
        data (np.ndarray): Dataset

    Returns:
        data_norm (np.ndarray): Normalised data
    '''

    # Find the maximum value in data
    data_max = data.max()

    # Find the minimum value in data
    data_min = data.min()

    # Normalise the data
    data_norm = (data - data_min) / (data_max - data_min)

    return data_norm


def flatten_data(data):
    '''
    Flatten the dataset into (28x28, m) where m is number of examples

    Args:
        data (np.ndarray): Dataset

    Returns:
        data_flat (np.ndarray) = Flattened dataset
    '''

    # Obtain the number of training examples
    num_examples = data.shape[0]

    # MNIST image is square so width = height
    image_width = data.shape[1]

    # Compute total number of pixels in image
    total_pixels = image_width**2

    # Flatten the image
    data_flat = data.reshape(total_pixels, num_examples)

    return data_flat


def load_mnist_data(verbose=True):
    '''
    Load and preprocess the MNIST dataset

    Args:
        verbose (bool): If true, prints the shapes for features and labels

    Returns:
        x_train (np.ndarray): Features for training dataset (28x28, m)
        y_train (np.ndarray): Labels for training dataset (10, m)
        x_test (np.ndarray): Features for testing dataset (28x28, m)
        y_test (np.ndarray): Labels for testing dataset (10, m)
    '''

    # Load tuples of MNIST data from Keras
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Flatten training and testing images
    x_train = flatten_data(x_train)
    x_test = flatten_data(x_test)

    # Normalise training and testing images
    x_train = normalise_data(x_train)
    x_test = normalise_data(x_test)

    # Ensure digit labels are 2D NumPy arrays
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    # Prints the shapes
    if verbose:
        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)
        print("x_test shape:", x_test.shape)
        print("y_test shape:", y_test.shape)
        print("Number of training examples:", x_train.shape[1])
        print("Number of test examples:", x_test.shape[1])

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":

    # Loads and prints the MNIST dataset
    load_mnist_data(verbose=True)
