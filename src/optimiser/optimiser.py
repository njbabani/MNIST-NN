# -*- coding: utf-8 -*-

"""
This module provides classes for optimisers to improve NN performance

Classes:
    Optimiser: Abstract base class for optimisers
    SGD: Implements the Stochastic Gradient Descent (SGD) optimisation

Typical usage example:
    optimiser = SGD(learning_rate=0.01)
    optimiser.update_weights(layer, grad_weights)
    optimiser.update_bias(layer, grad_bias)
"""

from abc import ABC, abstractmethod
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
        self.learning_rate = learning_rate
        self.layer_index = 0

        @property
        def layer_idx(self):
            """
            Gets the index of the current layer

            Returns:
                (int): The index of the layer being optimised
            """
            return self.layer_index

        @layer_idx.setter
        def layer_idx(self, layer_number: int):
            """
            Sets the index of the current layer

            Args:
                layer_number (int): The index to set for the layer
            """
            self.layer_index = layer_number

        @abstractmethod
        def update_weights(self, layer, grad_weights: np.ndarray):
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
        def update_bias(self, layer, grad_bias: np.ndarray):
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

        # Handle case if grad_weights is NOT NumPy array
        if not isinstance(grad_weights, np.ndarray):
            raise TypeError(
                f"Incorrect type for grad_weights: {type(grad_weights)}"
                )

        layer.weights -= self.learning_rate * grad_weights

    def update_bias(self, layer, grad_bias: np.ndarray):
        """
        Update biases according to Gradient Descent

        Args:
            layer: The layer whose biases are being updated
            grad_bias (np.ndarray): The gradient of the biases
        """

        # Handle case if grad_bias is NOT NumPy array
        if not isinstance(grad_bias, np.ndarray):
            raise TypeError(
                f"Incorrect type for grad_weights: {type(grad_bias)}"
                )

        layer.bias -= self.learning_rate * grad_bias
