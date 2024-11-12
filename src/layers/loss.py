# -*- coding: utf-8 -*-

"""
This module defines various loss functions for a NN

Classes:
"""

from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    """
    Abstract base class for loss functions

    All loss functions must implement the __call__ method
    and a gradient method to compute the derivative of the function
    """
    @abstractmethod
    def __call__(self, y_hat: np.ndarray, y: np.ndarray):
        """
        Computes the final forward pass using the loss function

        Args:
            y_hat (np.ndarray): The model's predicted outputs
            y (np.ndarray): The ground truth labels

        Returns:
            np.ndarray: The loss for a single example
        """
        pass

    @abstractmethod
    def gradient(self, *args, **kwargs):
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


class MSE(Loss):
    """
    Mean Squared Error (MSE) for regression tasks
    """
    def __call__(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the loss for a single example using MSE

        Args:
            y_hat (np.ndarray): The model's predicted outputs
            y (np.ndarray): The ground truth labels

        Returns:
            loss (np.ndarray): The loss for a single example
        """
        loss = np.mean(np.square(y - y_hat), axis=1, keepdims=True)
        return loss

    def gradient(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the gradient for MSE (for back prop)

        Args:
            y_hat (np.ndarray): The model's predicted outputs
            y (np.ndarray): The ground truth labels

        Returns:
            grad (np.ndarray): Gradient of loss function w.r.t y_hat
        """
        grad = 2 / y.size * (y_hat - y)

        return grad


class BCE(Loss):
    """
    Binary Cross Entropy (BCE) for binary classification
    """
    def __call__(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the loss for a single example using BCE

        Args:
            y_hat (np.ndarray): The model's predicted outputs
            y (np.ndarray): The ground truth labels

        Returns:
            loss (np.ndarray): The loss for a single example
        """

        # Define a small term to prevent log(0) being undefined
        delta = 1e-7

        # Compute
        loss = -1 * (
            y * np.log(y_hat + delta) + (1 - y) * np.log(1 - y_hat + delta)
        )

        return loss
