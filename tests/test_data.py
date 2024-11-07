# -*- coding: utf-8 -*-

'''
Testing module for data.py

Functions:
    test_normalise_data(): Raises error if data not normalised
    test_flatten_data(): Raises error if data not flattened
    test_load_mnist_data(): Raises error if MNIST has wrong shapes
'''

import numpy as np
from datasets.data import normalise_data, flatten_data, load_mnist_data


def test_normalise_data():
    '''
    Test the normalise_data()

    Raises:
        AssertionError: If the normalised data is not within the expected range
    '''

    # Define test shape (784, 10)
    test_shape = (784, 10)

    # Create test array of random values between [0, 255] and typecast to float
    test_array = np.random.randint(0, 256, test_shape).astype(float)

    # Normalise test data
    test_norm = normalise_data(test_array)

    assert test_norm.min() >= 0, f"Incorrect min value: {test_norm.min()}"
    assert test_norm.max() <= 1, f"Incorrect max value: {test_norm.max()}"

    print("test_normalise_data passed.")


def test_flatten_data():
    '''
    Test the flatten_data()

    Raises:
        AssertionError: If the flattened data does not yield expected shapes
    '''

    # Define test shape (600, 28, 28)
    test_shape = (600, 28, 28)

    # Create test array of random values between [0, 255] and typecast to float
    test_array = np.random.randint(0, 256, test_shape).astype(float)

    # Flatten test data
    test_flat = flatten_data(test_array)

    assert test_flat.shape[0] == 784, f"Incorrect shape: {test_flat.shape}"
    assert test_flat.shape[1] == 600, f"Incorrect shape: {test_flat.shape}"

    print("test_flatten_data passed.")


def test_load_mnist_data():
    '''
    Test the load_mnist_data() by checking entire MNIST dataset

    Raises:
        AssertionError: If the loaded data does not yield expected shapes
    '''

    # Load the MNIST dataset
    x_train, y_train, x_test, y_test = load_mnist_data(verbose=False)

    # Check shapes
    assert x_train.shape[0] == (784, 60000), (
        f"Wrong x_train shape: {x_train.shape}"
    )
    assert y_train.shape[0] == (60000, 1), (
        f"Wrong y_train shape: {y_train.shape}"
    )
    assert x_test.shape[0] == (784, 10000), (
        f"Wrong x_test shape: {x_test.shape}"
    )
    assert y_test.shape[0] == (10000, 1), (
        f"Wrong y_train shape: {y_test.shape}"
    )

    print("test_load_mnist_data passed.")


if "__name__" == "__main__":
    test_normalise_data()
    test_flatten_data()
    test_load_mnist_data()
