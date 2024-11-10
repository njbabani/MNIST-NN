# -*- coding: utf-8 -*-

"""
Testing module for layers.py

Functions:
    test_dense(): Error if forward pass or gradients are incorrect
    test_dropout_training(): Error if Dropout does not drop neurons
    test_dropout_inference(): Error if Dropout does not pass-through
"""

import numpy as np
import pytest
from src.layers.layer import Dense, Dropout
from src.optimisation.optimiser import SGD


def test_dense_layer():
    """
    Test the Dense layer forward pass and gradients

    Ensures that the forward pass of the Dense layer computes the expected
    output and that the gradients for weights and biases are correctly computed
    """

    # Set a fixed seed for reproducibility
    np.random.seed(1)

    # Initialise Dense layer
    dense = Dense(hidden_units=4)

    # Initialise test input and expected output shape
    test_input = np.random.randn(5, 3)

    # Perform a forward pass
    dense(test_input)

    # Check if the output has the correct shape
    assert dense.output.shape == (4, 3), "Output shape is incorrect."

    # Initialise a learning rate
    learning_rate = 0.01

    # Assume some gradients
    dw = np.ones_like(dense.weights)
    db = np.ones_like(dense.bias)

    # Initialise ideal weight and bias matching dense layer
    ideal_w = dense.weights
    ideal_b = dense.bias

    # Compute expected weights and biases
    ideal_w -= learning_rate * dw
    ideal_b -= learning_rate * db

    # Manually set gradients for testing update
    dense.grad_weights = np.ones_like(dense.weights)
    dense.grad_bias = np.ones_like(dense.bias)

    # Initialise optimiser
    optimiser = SGD(learning_rate)

    # Update weights and biases
    dense.update(optimiser)

    # Check if weights have been updated
    np.testing.assert_array_almost_equal(ideal_w, dense.weights)

    # Check if biases have been updated
    np.testing.assert_array_almost_equal(ideal_b, dense.bias)


def test_dropout_layer_training():
    """
    Test the Dropout layer in training mode

    Ensures Dropout layer drops neurons during forward pass when training
    """

    # Set a fixed seed for reproducibility
    np.random.seed(1)

    # Initialise Dropout layer
    dropout = Dropout(rate=0.5, training=True)

    # Initialise test input
    test_input = np.random.randn(4, 4)
    dropout_output = dropout(test_input)

    # Check if some neurons are dropped (i.e., output has zeros)
    assert np.any(dropout_output == 0), "Dropout did not drop any neurons."


def test_dropout_layer_inference():
    """
    Test the Dropout layer in inference mode

    Ensures that the Dropout layer acts as a pass-through during inference
    """
    # Initialise Dropout layer
    dropout = Dropout(rate=0.5, training=False)

    # Initialise test input
    test_input = np.random.randn(4, 4)
    dropout_output = dropout(test_input)

    # Check if the output is the same as the input in inference mode
    np.testing.assert_array_almost_equal(dropout_output, test_input)


if __name__ == "__main__":
    pytest.main()
