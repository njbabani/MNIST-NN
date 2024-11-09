# -*- coding: utf-8 -*-

"""
Testing module for optimiser.py

Functions:
    test_sgd_update_weights(): Raises error if incorrect update weights
    test_sgd_update_bias(): Raises error if incorrect update biases
    test_update_weights_invalid_grad_type(): Raises error if wrong grad type
    test_update_bias_invalid_grad_type(): Raises error if wrong grad type
"""

import numpy as np
import pytest
from src.optimisation.optimiser import SGD


class TestLayer:
    """
    Mock layer class to simulate a layer with weights and biases.

    Attributes:
        weights (np.ndarray): Weights of the layer
        bias (np.ndarray): Bias of the layer
    """

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias


def test_sgd_update_weights():
    """Test the update_weights method of SGD optimiser."""

    # Initialise test variables
    learning_rate = 0.01
    optimiser = SGD(learning_rate=learning_rate)
    layer = TestLayer(
        weights=np.array([[0.5, -0.2], [0.3, 0.8]]), bias=np.array([0.1, -0.1])
    )
    grad_weights = np.array([[0.1, -0.05], [0.05, 0.2]])

    # Expected updated weights after applying gradient descent
    expected_weights = layer.weights - learning_rate * grad_weights

    # Perform weight update
    optimiser.update_weights(layer, grad_weights)

    # Assert weights are updated as expected
    np.testing.assert_array_almost_equal(layer.weights, expected_weights)


def test_sgd_update_bias():
    """Test the update_bias method of SGD optimiser."""

    # Initialise test variables
    learning_rate = 0.01
    optimiser = SGD(learning_rate=learning_rate)
    layer = TestLayer(
        weights=np.array([[0.5, -0.2], [0.3, 0.8]]), bias=np.array([0.1, -0.1])
    )
    grad_bias = np.array([0.05, -0.02])

    # Expected updated bias after applying gradient descent
    expected_bias = layer.bias - learning_rate * grad_bias

    # Perform bias update
    optimiser.update_bias(layer, grad_bias)

    # Assert biases are updated as expected
    np.testing.assert_array_almost_equal(layer.bias, expected_bias)


def test_update_weights_invalid_grad_type():
    """Test update_weights method raises TypeError with wrong gradient type."""

    # Initialise test variables
    learning_rate = 0.01
    optimiser = SGD(learning_rate=learning_rate)
    layer = TestLayer(
        weights=np.array([[0.5, -0.2], [0.3, 0.8]]), bias=np.array([0.1, -0.1])
    )

    # Type checks
    with pytest.raises(TypeError, match="Incorrect type for grad_weights"):
        optimiser.update_weights(layer, grad_weights="invalid_type")


def test_update_bias_invalid_grad_type():
    """Test update_bias method raises TypeError with invalid gradient type."""

    # Initialise test variables
    learning_rate = 0.01
    optimiser = SGD(learning_rate=learning_rate)
    layer = TestLayer(
        weights=np.array([[0.5, -0.2], [0.3, 0.8]]), bias=np.array([0.1, -0.1])
    )

    # Type checks
    with pytest.raises(TypeError, match="Incorrect type for grad_weights"):
        optimiser.update_bias(layer, grad_bias="invalid_type")


if __name__ == "__main__":
    pytest.main()
