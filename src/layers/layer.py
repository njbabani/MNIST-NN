# -*- coding: utf-8 -*-

"""
This module describes the types of layers for the NN

Classes:
    Layer: Abstract base class for layers
    Dense: Implements a fully connected (dense) layer

Typical usage example:
    dense_layer = Dense(units=64)
    output = dense_layer(data)
    dense_layer.update(optimiser)
"""

from abc import ABC, abstractmethod
import numpy as np
from src.optimiser.optimiser import Optimiser


class Layer(ABC):
    """
    Parent class for NN layers

    Attributes:
        output (np.ndarray): Output of layer after linear forward propagation
    """
    @property
    @abstractmethod
    def output(self):
        """
        np.ndarray: Output of layer after most recent forward pass
        """
        pass

    @abstractmethod
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass for the layer

        Args:
            data (np.ndarray): Input data for the layer (i.e. features)

        Returns:
            np.ndarray: Output of the layer after linear forward propagation
        """
        pass

    @abstractmethod
    def build(self, data: np.ndarray):
        """
        Builds the layer with weights and biases based on the input shape

        Args:
            data (np.ndarray): Input data for determining the shape
        """
        pass

    @abstractmethod
    def update(self, optimiser: Optimiser):
        """
        Updates the layer's weights and biases based on selected optimiser

        Args:
            optimiser (Optimiser): The optimiser used to update parameters
        """
        pass


class Dense(Layer):
    """
    Fully connected (dense) layer implementation

    Attributes:
        units (int): Number of neurons in the dense layer
        weights (np.ndarray): Weights matrix of the layer
        bias (np.ndarray): Bias vector of the layer
        grad_weights (np.ndarray): Gradient of weights after backpropagation
        grad_bias (np.ndarray): Gradient of biases after backpropagation
    """
    def __init__(self, hidden_units: int):
        """
        Initialises a Dense layer with specified number of hidden units

        Args:
            hidden_units (int): Number of neurons in the dense layer
        """
        super().__init__()
        self._hidden_units = hidden_units
        self._input_units = None
        self._weights = None
        self._bias = None
        self._output = None
        self._dw = None
        self._db = None

    @property
    def weights(self):
        """
        np.ndarray: The weights of the dense layer
        """
        return self._weights

    @weights.setter
    def weights(self, weights: np.ndarray):
        """
        np.ndarray: Sets the weights
        """
        self._weights = weights

    @property
    def bias(self):
        """
        np.ndarray: The bias of the dense layer
        """
        return self._bias

    @bias.setter
    def bias(self, bias: np.ndarray):
        """
        np.ndarray: Sets the biases
        """
        self._bias = bias

    @property
    def grad_weights(self):
        """
        np.ndarray: Gradient of the weights for backpropagation
        """
        return self._dw

    @grad_weights.setter
    def grad_weights(self, gradients: np.ndarray):
        """
        np.ndarray: Sets the gradients of the weights
        """
        self._dw = gradients

    @property
    def grad_bias(self):
        """
        np.ndarray: Gradient of the biases for backpropagation
        """
        return self._db

    @grad_bias.setter
    def grad_bias(self, gradients: np.ndarray):
        """
        np.ndarray: Sets the gradients of the biases
        """
        self._db = gradients

    @property
    def output(self):
        """
        np.ndarray: The output of the dense layer after forward pass
        """
        return self._output

    def build(self, data: np.ndarray):
        """
        Initialises weights and biases using He initialisation

        Args:
            data (np.ndarray): Input data for determining the shape.
        """
        self._input_units = data.shape[0]
        random_weights = np.random.randn(self._units, self._input_units)
        scaling_factor = np.sqrt(2.0 / self._input_units)
        self._weights = random_weights * scaling_factor
        self._bias = np.zeros((self._units, 1))

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass for the dense layer

        Args:
            data (np.ndarray): Input data to the layer

        Returns:
            np.ndarray: Output of the layer after applying weights and biases
        """
        if self._weights is None:
            self.build(data)

        self._output = np.dot(self._weights, data) + self._bias
        return self._output

    def update(self, optimiser: Optimiser):
        """
        Updates weights and biases using the specified optimiser

        Args:
            optimiser (Optimiser): The optimiser used for updating parameters
        """
        optimiser.update_weights(self, self.grad_weights)
        optimiser.update_bias(self, self.grad_bias)
