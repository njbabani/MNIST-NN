# -*- coding: utf-8 -*-

"""
This module defines various cost functions for a NN

Classes:
    Cost: Abstract parent class for cost functions
    Mean Squared Error (MSE): Used for regression tasks
    Binary Cross Entropy (BCE): Used for binary classification
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class Cost(ABC):
    """
    Abstract base class for cost functions

    All cost functions must implement the __call__ method
    and a gradient method to compute the derivative of the function
    """
    @abstractmethod
    def __call__(self, y_hat: np.ndarray, y: np.ndarray):
        """
        Computes the final forward pass using the cost function

        Args:
            y_hat (np.ndarray): The model's predicted outputs
            y (np.ndarray): The ground truth labels

        Returns:
            np.ndarray: The cost for a single example
        """
        pass

    @abstractmethod
    def gradient(self, *args, **kwargs) -> Any:
        """
        Computes the gradient of the function with respect to its inputs

        This step is used during back propagation

        Args:
            *args: Positional arguments for computing the gradient
            **kwargs: Keyword arguments for computing the gradient

        Returns:
            Any: The computed gradient (typically NumPy array)

        Raises:
            NotImplementedError: Subclass did not implement this method
        """
        raise NotImplementedError("Subclasses must have a gradient method.")


class MSE(Cost):
    """
    Mean Squared Error (MSE) for regression tasks
    """
    def __call__(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the cost for a single example using MSE

        Args:
            y_hat (np.ndarray): The model's predicted outputs
            y (np.ndarray): The ground truth labels

        Returns:
            cost (np.ndarray): The cost for a single example
        """

        # Compute the cost (average loss)
        cost = np.mean(np.square(y - y_hat), axis=1, keepdims=True)
        return cost

    def gradient(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the gradient for MSE (for back prop)

        Args:
            y_hat (np.ndarray): The model's predicted outputs
            y (np.ndarray): The ground truth labels

        Returns:
            grad (np.ndarray): Gradient of cost function w.r.t y_hat
        """

        # Compute gradient for MSE
        grad = 2 / y.size * (y_hat - y)
        return grad


class BCE(Cost):
    """
    Binary Cross Entropy (BCE) for binary classification
    """
    def __call__(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the cost for a single example using BCE

        Args:
            y_hat (np.ndarray): The model's predicted outputs
            y (np.ndarray): The ground truth labels

        Returns:
            cost (np.ndarray): The cost for a single example
        """

        # Define a small term to prevent log(0) being undefined
        delta = 1e-7

        # Ensure y_hat within range of [delta, 1 - delta]
        y_hat = np.clip(y_hat, delta, 1 - delta)

        # Compute the cost (average loss)
        cost = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return cost

    def gradient(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the gradient for MSE (for back prop)

        Args:
            y_hat (np.ndarray): The model's predicted outputs
            y (np.ndarray): The ground truth labels

        Returns:
            grad (np.ndarray): Gradient of cost function w.r.t y_hat
        """

        # Define small term to prevent division by zero
        delta = 1e-7

        # Ensure y_hat within range of [delta, 1 - delta]
        y_hat = np.clip(y_hat, delta, 1 - delta)

        # Compute gradient for BCE
        grad = - y / y_hat + (1 - y) / (1 - y_hat)
        return grad
