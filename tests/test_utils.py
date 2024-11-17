# -*- coding: utf-8 -*-

"""
Test utils.py

Functions:
    test_accuracy_metric(): Raises an error if accuracy metric is incorrect
"""

import pytest
import numpy as np
from src.common.utils import accuracy_metric


def test_accuracy_metric():
    """Test accuracy metric"""

    # Initialise test prediction probability array
    test_pred_prob = np.array([
        [0., 1., 0., 1.],
        [1., 0., 0., 0.],
        [0., 0., 1., 0.]
    ])

    # Initialise test true probability array
    test_true_prob = np.array([
        [0., 1., 0., 0.],
        [1., 0., 0., 1.],
        [0., 0., 1., 0.]
    ])

    # Upon inspection, we expect an accuracy of 75%
    expected_accuracy = 0.75

    # Assertion raises an error if accuracy is incorrect
    assert accuracy_metric(test_pred_prob, test_true_prob) == expected_accuracy


if __name__ == "__main__":
    pytest.main()
