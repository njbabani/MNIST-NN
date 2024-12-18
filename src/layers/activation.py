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
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any


class Activation(ABC):
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

    @abstractmethod
    def gradient(self, *args, **kwargs) -> Any:
        """
        Computes the gradient of the function with respect to its inputs

        Args:
            *args: Positional arguments for computing the gradient
            **kwargs: Keyword arguments for computing the gradient

        Returns:
            Any: The computed gradient (typically NumPy array)

        Raises:
            NotImplementedError: Subclass did not implement this method

        Example:
            For an activation function like ReLU, the gradient method would
            return 1 for positive inputs and 0 for non-positive inputs
        """
        raise NotImplementedError("Subclasses must have a gradient method.")

    @property
    @abstractmethod
    def output(self) -> np.ndarray:
        """
        np.ndarray: Output of layer after most recent forward pass
        """
        pass


class Linear(Activation):
    """
    Implements the linear (identity) activation function
    """
    def __init__(self):
        super().__init__()
        self._output = None

    @property
    def output(self):
        return self._output

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Applies the linear activation function to the data

        Args:
            data (np.ndarray): Input data

        Returns:
            np.ndarray: The same data as the output.
        """
        self._output = data
        return self._output

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
    def __init__(self):
        super().__init__()
        self._output = None

    @property
    def output(self):
        return self._output

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Applies the ReLU activation function to the data

        Args:
            data (np.ndarray): data.

        Returns:
            np.ndarray: Equals to data when data > 0, otherwise 0
        """
        self._output = np.maximum(data, 0)
        return self._output

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
    def __init__(self):
        super().__init__()
        self._output = None

    @property
    def output(self):
        return self._output

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Applies the sigmoid activation function to the data.

        Args:
            data (np.ndarray): data.

        Returns:
            np.ndarray: Transformed output is between 0 and 1
        """
        self._output = 1.0 / (1 + np.exp(-data))
        return self._output

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
    def __init__(self):
        super().__init__()
        self._output = None

    @property
    def output(self):
        return self._output

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Applies the tanh activation function to the data

        Args:
            data (np.ndarray): Input data

        Returns:
            np.ndarray: Transformed output is between -1 and 1
        """
        self._output = np.divide(
            (np.exp(data) - np.exp(-data)), (np.exp(data) + np.exp(-data))
        )
        return self._output

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
    def __init__(self):
        super().__init__()
        self._output = None

    @property
    def output(self):
        """Return softmax output"""
        return self._output

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Applies the softmax activation function to the data

        Args:
            data (np.ndarray): Input data

        Returns:
            np.ndarray: Gives probabilities for different classes
        """

        # Define small value to prevent division by zero
        delta = 1e-12

        # Prevent numerical overflow by leveraging exponential property
        exp_data = np.exp(data - np.max(data, axis=0, keepdims=True))

        # Compute sum of exponential data along columns with some small delta
        sum_exp_data = np.sum(exp_data, axis=0, keepdims=True) + delta

        # Compute softmax
        self._output = exp_data / sum_exp_data
        return self._output

    def gradient(self, data: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of softmax

        This won't be explicitly used as we will use CCE gradient

        Args:
            data (np.ndarray): Input data

        Returns:
            np.ndarray: Gradient of softmax
        """

        # Compute the softmax output
        sfmax = self(data)

        # Initialise gradient matrix
        grad = np.zeros_like(sfmax)

        # d softmax(x)_j / d x_i = softmax(x)_j * (Kron-del_ij - softmax(x)_i)
        for i in range(sfmax.shape[0]):
            for j in range(sfmax.shape[1]):
                if i == j:
                    grad[i, j] = sfmax[i, j] * (1 - sfmax[i, j])
                else:
                    grad[i, j] = -sfmax[i, j] * sfmax[i, j]

        return grad
