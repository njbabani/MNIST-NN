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

import tensorflow as tf
import numpy as np


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
    Flattens the dataset into shape (n_x, m),
    where n_x is the number of features (pixels),
    and m is the number of examples

    Args:
        data (np.ndarray): Dataset of shape (m, height, width)

    Returns:
        data_flat (np.ndarray): Flattened dataset of shape (n_x, m)
    '''
    # Number of examples
    m = data.shape[0]  # m examples

    # Image dimensions
    height = data.shape[1]
    width = data.shape[2]

    # Number of features (pixels per image)
    n_x = height * width  # n_x features

    # Flatten each image individually and transpose
    data_flat = data.reshape(m, n_x).T  # Shape: (n_x, m)

    return data_flat


def one_hot_encode(labels, num_classes=10):
    '''
    One-hot encode the labels for use in
    Categorical Cross-Entropy (CCE)

    Args:
        labels (np.ndarray): Array of labels to be encoded
        num_classes (int): Number of classes (defaults to 10)

    Returns:
        np.ndarray: One-hot encoded labels
    '''
    if np.any(labels >= num_classes):
        raise ValueError("Labels contain values outside range of num_classes.")
    return np.eye(num_classes)[labels].T


def load_mnist_data(verbose=True, encode=True):
    '''
    Load and preprocess the MNIST dataset

    Args:
        verbose (bool): If true, prints the shapes for features and labels
        encode (bool): If true, performs one-hot encoding for CCE

    Returns:
        x_train (np.ndarray): Features for training dataset (28x28, m)
        y_train (np.ndarray): Labels for training dataset (10, m)
        x_test (np.ndarray): Features for testing dataset (28x28, m)
        y_test (np.ndarray): Labels for testing dataset (10, m)
    '''

    # Load tuples of MNIST data from Keras
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Flatten training and testing images
    x_train = flatten_data(x_train)
    x_test = flatten_data(x_test)

    # Normalise training and testing images
    x_train = normalise_data(x_train)
    x_test = normalise_data(x_test)

    if encode:
        # One-hot encode the labels if encode is True
        y_train = one_hot_encode(y_train, num_classes=10)
        y_test = one_hot_encode(y_test, num_classes=10)
    else:
        # Ensure digit labels are 2D NumPy arrays
        y_train = y_train.reshape(1, y_train.shape[0])
        y_test = y_test.reshape(1, y_test.shape[0])

    # Prints the shapes
    if verbose:
        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)
        print("x_test shape:", x_test.shape)
        print("y_test shape:", y_test.shape)
        print("Number of training examples:", x_train.shape[1])
        print("Number of test examples:", x_test.shape[1])

    return x_train, y_train, x_test, y_test


def train_val_split(x, y, val_ratio=0.2, random_seed=1):
    """
    Manually splits data into training and validation sets.

    Args:
        x (np.ndarray): Input data (features, examples)
        y (np.ndarray): Labels (classes, examples)
        val_ratio (float): Fraction of data to use for validation
        random_seed (int): Seed for reproducibility

    Returns:
        x_train, y_train, x_val, y_val: Split datasets
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Determine number of examples
    num_examples = x.shape[1]

    # Generate shuffled indices
    indices = np.arange(num_examples)
    np.random.shuffle(indices)

    # Calculate the split index
    val_size = int(num_examples * val_ratio)

    # Split indices into training and validation sets
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    # Split data based on indices
    x_train, y_train = x[:, train_indices], y[:, train_indices]
    x_val, y_val = x[:, val_indices], y[:, val_indices]

    return x_train, y_train, x_val, y_val


if __name__ == "__main__":

    # Loads and prints the MNIST dataset
    load_mnist_data(verbose=True)
