# -*- coding: utf-8 -*-

"""
Testing module for cost.py

Functions:
    test_MSE(): Raises error if incorrect cost value or shape
    test_BCE(): Raises error if incorrect cost value or shape
    test_CCE(): Raises error if incorrect cost value or shape
"""

import numpy as np
import pytest
from src.layers.loss import MSE, BCE, CCE


def test_MSE():
    """
    Test the Mean Squared Error (MSE) cost and gradient
    """

    # Initialise "MSE" test object
    mse = MSE()

    # Initialise true test label
    label = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Initialise a test prediction as broadcasted label
    predict = 0.1 + label

    # Expected cost output retaining dimensions
    ideal_cost = np.array([[0.01]])

    # Computed MSE cost
    mse_cost = mse(predict, label)

    # Expected gradient output
    ideal_grad = 2 / label.size * (predict - label)

    # Computed MSE gradient
    mse_grad = mse.gradient(predict, label)

    # Checks if cost has same value
    np.testing.assert_array_almost_equal(
        mse_cost, ideal_cost, err_msg="MSE cost calculation is incorrect."
    )

    # Checks if cost has same shape
    assert mse_cost.shape == ideal_cost.shape, "MSE cost shape is incorrect."

    # Checks if gradient has the expected value
    np.testing.assert_array_almost_equal(
        mse_grad, ideal_grad, err_msg="MSE gradient calculation is incorrect."
    )

    # Checks if gradient has the expected shape
    assert mse_grad.shape == ideal_grad.shape, "MSE grad shape is incorrect."


def test_BCE():
    """
    Test the Binary Cross-Entropy (BCE) cost and gradient
    """

    # Initialise "BCE" test object
    bce = BCE()

    # Initialise true binary labels
    label = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]])

    # Initialise test predictions
    predict = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.1], [0.6, 0.4, 0.3]])

    # Compute the BCE cost
    bce_cost = bce(predict, label)

    # Small value to prevent log(0)
    delta = 1e-7
    predict = np.clip(predict, delta, 1 - delta)

    # Manually calculate the expected cost
    ideal_cost = -np.mean(
        label * np.log(predict) + (1 - label) * np.log(1 - predict),
        keepdims=True
    )

    # Expected gradient using the BCE gradient formula
    ideal_grad = (
        -label / predict + (1 - label) / (1 - predict)
    )

    # Normalise gradient
    ideal_grad /= label.size

    # Compute the BCE gradient
    bce_grad = bce.gradient(predict, label)

    # Check if the computed cost is close to the expected value
    np.testing.assert_array_almost_equal(
        bce_cost, ideal_cost, err_msg="BCE cost calculation is incorrect."
    )

    # Check if the computed cost has the correct shape
    assert bce_cost.shape == ideal_cost.shape, "BCE cost shape is incorrect."

    # Check if the computed gradient is close to the expected gradient
    np.testing.assert_array_almost_equal(
        bce_grad, ideal_grad, err_msg="BCE grad calculation is incorrect."
    )

    # Check if the gradient has the correct shape
    assert bce_grad.shape == ideal_grad.shape, "BCE grad shape is incorrect."


def test_CCE():
    """
    Test the Categorical Cross-Entropy (CCE) cost and gradient
    """

    # Initialise "CCE" test object
    cce = CCE()

    # Initialise true labels in one-hot encoding format
    label = np.array([
        [1, 0, 0],  # Class 1
        [0, 1, 0],  # Class 2
        [0, 0, 1]   # Class 3
    ])

    # Initialise test predictions (probabilities output by softmax)
    predict = np.array([
        [0.7, 0.2, 0.1],  # Predicted class 1
        [0.1, 0.6, 0.3],  # Predicted class 2
        [0.2, 0.2, 0.6]   # Predicted class 3
    ])

    # Compute the CCE cost
    cce_cost = cce(predict, label)

    # Small value to prevent log(0)
    delta = 1e-7
    predict = np.clip(predict, delta, 1 - delta)

    # Manually calculate the expected cost
    ideal_cost = -np.mean(
        np.sum(label * np.log(predict), axis=0, keepdims=True), keepdims=True
    )

    # Expected gradient using the CCE gradient formula
    ideal_grad = (predict - label) / label.size

    # Compute the CCE gradient
    cce_grad = cce.gradient(predict, label)

    # Check if the computed cost is close to the expected value
    np.testing.assert_array_almost_equal(
        cce_cost, ideal_cost, err_msg="CCE cost calculation is incorrect."
    )

    # Check if the computed cost has the correct shape
    assert cce_cost.shape == ideal_cost.shape, "CCE cost shape is incorrect."

    # Check if the computed gradient is close to the expected gradient
    np.testing.assert_array_almost_equal(
        cce_grad, ideal_grad, err_msg="CCE grad calculation is incorrect."
    )

    # Check if the gradient has the correct shape
    assert cce_grad.shape == ideal_grad.shape, "CCE grad shape is incorrect."


if __name__ == "__main__":
    pytest.main()
