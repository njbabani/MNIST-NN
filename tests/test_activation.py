# -*- coding: utf-8 -*-

"""
Testing module for activation.py

Functions:
    test_linear_activation(): Raises error if incorrect value or grad
    test_relu_activation(): Raises error if incorrect value or grad
    test_sigmoid_activation(): Raises error if incorrect value or grad
    test_softmax_activation(): Raises error if incorrect value or grad
"""


import numpy as np
import pytest
from src.layers.activation import Linear, ReLU, Sigmoid, Softmax
from scipy.special import expit, softmax


def test_linear_activation():
    """
    Test the linear activation function
    """

    # Initialise "Linear" test object
    linear_af = Linear()

    # Initialise test input
    test_input = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Expected gradient output
    grad_out = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    # Checks if test input matches output of linear function
    np.testing.assert_array_almost_equal(
        linear_af(test_input),
        test_input
        )

    # Checks if gradient output is correct
    np.testing.assert_array_almost_equal(
        linear_af.gradient(test_input),
        grad_out
        )


def test_relu_activation():
    """
    Test ReLU activation function
    """

    # Initialise "ReLU" test object
    relu_af = ReLU()

    # Initialise test input
    test_input = np.array([[1.0, 2.0, 3.0], [-4.0, -5.0, -6.0]])

    # Expected test and gradient output
    test_out = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    grad_out = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])

    # Checks if test input matches output of ReLU function
    np.testing.assert_array_almost_equal(
        relu_af(test_input),
        test_out,
        )

    # Checks if gradient output is correct
    np.testing.assert_array_almost_equal(
        relu_af.gradient(test_input),
        grad_out
        )


def test_sigmoid_activation():
    """
    Test sigmoid activation function
    """

    # Initialise "sigmoid" test object
    sigmoid_af = Sigmoid()

    # Initialise test input
    test_input = np.random.randn(4, 4)

    # Expected test and gradient output
    test_out = expit(test_input)
    grad_out = test_out * (1 - test_out)

    # Checks if test input matches output of sigmoid function
    np.testing.assert_array_almost_equal(
        sigmoid_af(test_input),
        test_out,
        )

    # Checks if gradient output is correct
    np.testing.assert_array_almost_equal(
        sigmoid_af.gradient(test_input),
        grad_out
        )


def test_softmax_activation():
    """
    Test softmax activation function
    """

    # Fixed seed for reproducibility
    np.random.seed(1)

    # Initialise "softmax" test object
    softmax_af = Softmax()

    # Initialise test input
    test_input = np.random.randn(4, 4)

    # Expected test output
    test_out = softmax(test_input, axis=0)

    # Initialise grad_out
    grad_out = np.zeros_like(test_input)

    # Expected grad output
    for i in range(test_input.shape[0]):
        for j in range(test_input.shape[1]):
            if i == j:
                grad_out[i, j] = test_out[i, j] * (1 - test_out[i, j])
            else:
                grad_out[i, j] = -test_out[i, j] * test_out[i, j]

    # Checks if test input matches output of sigmoid function
    np.testing.assert_array_almost_equal(
        softmax_af(test_input),
        test_out,
        )

    # Checks if gradient output is correct
    np.testing.assert_array_almost_equal(
        softmax_af.gradient(test_input),
        grad_out
        )


if __name__ == "__main__":
    pytest.main()
