# -*- coding: utf-8 -*-

"""
Testing module for data.py

Functions:
    test_normalise_data(): Raises error if data not normalised
    test_flatten_data(): Raises error if data not flattened
    test_load_mnist_data(): Raises error if MNIST has wrong shapes
"""

import numpy as np
import pytest
from datasets.data import normalise_data, flatten_data
from datasets.data import one_hot_encode, load_mnist_data


def test_normalise_data():
    """
    Test the normalise_data()

    Raises:
        AssertionError: If the normalised data is not within the expected range
    """

    # Define test shape (784, 10)
    test_shape = (784, 10)

    # Create test array of random values between [0, 255] and typecast to float
    test_array = np.random.randint(0, 256, test_shape).astype(float)

    # Normalise test data
    test_norm = normalise_data(test_array)

    # Check normalisation
    assert test_norm.min() >= 0, f"Incorrect min value: {test_norm.min()}"
    assert test_norm.max() <= 1, f"Incorrect max value: {test_norm.max()}"

    print("test_normalise_data passed.")


def test_flatten_data():
    """
    Test the flatten_data()

    Raises:
        AssertionError: If the flattened data does not yield expected shapes
    """

    # Define test shape (600, 28, 28)
    test_shape = (600, 28, 28)

    # Create test array of random values between [0, 255] and typecast to float
    test_array = np.random.randint(0, 256, test_shape).astype(float)

    # Flatten test data
    test_flat = flatten_data(test_array)

    # Check shapes
    assert test_flat.shape[0] == 784, f"Incorrect shape: {test_flat.shape}"
    assert test_flat.shape[1] == 600, f"Incorrect shape: {test_flat.shape}"

    print("test_flatten_data passed.")


def test_one_hot_encode():
    """
    Test the one_hot _encode()

    Raises:
        AssertionError: If one-hot array does not equal expected result
    """

    # Initialise test inputs
    test_labels = np.array([0, 1, 2, 3, 4])
    test_num_classes = 5

    # Generate one-hot encoded array
    one_hot_output = one_hot_encode(test_labels, test_num_classes)

    # Define the expected result as identiy matrix
    ideal_output = np.eye(5)

    for i in range(ideal_output.shape[0]):
        for j in range(ideal_output.shape[1]):
            assert (
                one_hot_output[i, j] == ideal_output[i, j]
            ), f"Incorrect output: {one_hot_output}"


def test_load_mnist_data():
    """
    Test the load_mnist_data() by checking entire MNIST dataset

    Raises:
        AssertionError: If the loaded data does not yield expected shapes
    """

    # Set the verbose and encode to be False
    ver = False
    enc = False

    # Load the MNIST dataset
    x_train, y_train, x_test, y_test = load_mnist_data(ver, enc)

    # Check shapes
    assert x_train.shape == (784, 60000), \
        f"Wrong x_train shape: {x_train.shape}"

    assert y_train.shape == (1, 60000), \
        f"Wrong y_train shape: {y_train.shape}"

    assert x_test.shape == (784, 10000), \
        f"Wrong x_test shape: {x_test.shape}"

    assert y_test.shape == (1, 10000), \
        f"Wrong y_test shape: {y_test.shape}"

    print("test_load_mnist_data passed.")


if "__name__" == "__main__":
    pytest.main()
