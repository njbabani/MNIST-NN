# -*- coding: utf-8 -*-

"""
This module provides classes for optimisers to improve NN performance

Classes:
    Optimiser: Abstract base class for optimisers
    SGD: Implements the Stochastic Gradient Descent (SGD) optimisation
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class Optimiser(ABC):
    """
    Parent class for optimisers

    Attributes:
        learning_rate (float): Learning rate for the optimiser
        layer_index (int): Index of the current layer being optimised
    """
    def __init__(self, learning_rate: float):
        """
        Initialise the optimiser with a learning rate

        Args:
            learning_rate (float): The learning rate for the optimiser
        """
        self._learning_rate = learning_rate

    @property
    def learning_rate(self):
        """
        Gets the learning rate for optimiser

        Returns:
            float: Learning rate
        """
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        """
        Sets the learning rate for optimiser

        Args:
            learning_rate (int): Learning rate
        """
        self._layer_index = learning_rate

    @abstractmethod
    def update_weights(self, layer, grad_weights: np.ndarray) -> Any:
        """
        Updates the weights of the specified layer.

        This method should be implemented by subclasses to define
        how the weights are updated based on the gradients

        Args:
            layer: The layer whose weights are being updated
            grad_weights (np.ndarray): The gradient of the weights
        """
        pass

    @abstractmethod
    def update_bias(self, layer, grad_bias: np.ndarray) -> Any:
        """
        Updates the biases of the specified layer

        This method should be implemented by subclasses to define
        how the biases are updated based on the gradients

        Args:
            layer: The layer whose biases are being updated
            grad_bias (np.ndarray): The gradient of the biases
        """
        pass


class SGD(Optimiser):
    """Stochastic Gradient Descent"""
    def __init__(self, learning_rate: float):
        """
        Initialise the SGD with a learning rate

        Args:
            learning_rate (float): The learning rate for the SGD
        """
        super().__init__(learning_rate)

    def update_weights(self, layer, grad_weights: np.ndarray):
        """
        Update weights according to Gradient Descent

        Args:
            layer: The layer whose weights are being updated
            grad_weights (np.ndarray): The gradient of the weights
        """
        if not isinstance(grad_weights, np.ndarray):
            raise TypeError(
                f"Incorrect type for grad_weights: {type(grad_weights)}"
                )

        layer.weights -= self._learning_rate * grad_weights

    def update_bias(self, layer, grad_bias: np.ndarray):
        """
        Update biases according to Gradient Descent

        Args:
            layer: The layer whose biases are being updated
            grad_bias (np.ndarray): The gradient of the biases
        """
        if not isinstance(grad_bias, np.ndarray):
            raise TypeError(
                f"Incorrect type for grad_weights: {type(grad_bias)}"
                )

        layer.bias -= self._learning_rate * grad_bias
