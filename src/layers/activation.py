# -*- coding: utf-8 -*-

"""
This module defines various activation functions for a NN

Classes:
    Activation: Abstract base class for activation functions
    Linear: Implements the linear activation function (identity function)
    ReLU: Implements the Rectified Linear Unit (ReLU) activation function
    Sigmoid: Implements the sigmoid activation function
    Tanh: Implements hyperbolic tangent activation function
    Softmax: Implements softmax activation function

Typical usage example:
    relu = ReLU()
    output = relu(data)
    gradient = relu.gradient(data)
"""

from abc import abstractmethod
import numpy as np
from src.common.differentiable import Differentiable


class Activation(Differentiable):
    """
    Abstract base class for activation functions

    All activation functions must implement the __call__ method
    and a gradient method to compute the derivative of the function
    """
    @abstractmethod
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the activation function

        Args:
            data (np.ndarray): data to the activation function

        Returns:
            np.ndarray: Transformed output after applying activation function
        """
        pass


class Linear(Activation):
    """
    Implements the linear (identity) activation function
    """

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Applies the linear activation function to the data

        Args:
            data (np.ndarray): Input data

        Returns:
            np.ndarray: The same data as the output.
        """
        return data

    def gradient(self, data: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the linear activation function

        Args:
            data (np.ndarray): Input data

        Returns:
            np.ndarray: An array of ones with the same shape as the input
        """
        return np.ones_like(data)


class ReLU(Activation):
    """
    Implements the ReLU activation function
    """

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Applies the ReLU activation function to the data

        Args:
            data (np.ndarray): data.

        Returns:
            np.ndarray: Equals to data when data > 0, otherwise 0
        """
        return np.maximum(data, 0)

    def gradient(self, data: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the ReLU activation function

        Args:
            data (np.ndarray): Input data

        Returns:
            np.ndarray: An array where positive values are 1, otherwise 0
        """
        _result = np.zeros_like(data)
        _result[data > 0] = 1
        return _result


class Sigmoid(Activation):
    """
    Implements the sigmoid activation function
    """

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Applies the sigmoid activation function to the data.

        Args:
            data (np.ndarray): data.

        Returns:
            np.ndarray: Transformed output is between 0 and 1
        """
        return 1.0 / (1 + np.exp(-data))

    def gradient(self, data: np.ndarray) -> np.ndarray:
        """
        The gradient is computed as sigmoid(x) * (1 - sigmoid(x))

        Args:
            data (np.ndarray): data

        Returns:
            np.ndarray: Gradient of sigmoid
        """
        sigmoid_out = self(data)
        return sigmoid_out * (1 - sigmoid_out)


class Tanh(Activation):
    """
    Implements the tanh activation function
    """

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Applies the tanh activation function to the data

        Args:
            data (np.ndarray): Input data

        Returns:
            np.ndarray: Transformed output is between -1 and 1
        """
        return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))

    def gradient(self, data: np.ndarray) -> np.ndarray:
        """
        The gradient is computed as sech^2(x) = 1 - tanh^2(x)

        Args:
            data (np.ndarray): Input data

        Returns:
            np.ndarray: Gradient of tanh
        """
        tanh_out = self(data)
        return 1 - np.square(tanh_out)


class Softmax(Activation):
    """
    Implements the softmax activation function
    """

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Applies the softmax activation function to the data

        Args:
            data (np.ndarray): Input data

        Returns:
            np.ndarray: Gives probabilities for different classes
        """
        return (np.exp(data)) / np.sum(np.exp(data), axis=1)

    def gradient(self, data: np.ndarray) -> np.ndarray:
        """
        Vectorised softmax gradient

        Args:
            data (np.ndarray): Input data

        Returns:
            np.ndarray: Gradient of softmax
        """
        softmax_out = self(data)
        softmax_out = softmax_out.reshape(-1, 1)
        return np.diagflat(softmax_out) - np.dot(softmax_out, softmax_out.T)
