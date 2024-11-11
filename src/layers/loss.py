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
    def __call__(self, predict: np.ndarray, labels: np.ndarray):
        """
        Computes the final forward pass using the loss function

        Args:
            predict (np.ndarray): The model's predicted outputs
            labels (np.ndarray): The ground truth labels

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
    def __call__(self, predict: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute the loss for a single example

        Args:
            predict (np.ndarray): The model's predicted outputs
            labels (np.ndarray): The ground truth labels

        Returns:
            loss (np.ndarray): The loss for a single example
        """
        loss = np.mean(np.square(labels - predict), axis=1, keepdims=True)
        return loss

    def gradient(self, predict: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute the gradient for MSE (for back prop)

        Args:
            predict (np.ndarray): The model's predicted outputs
            labels (np.ndarray): The ground truth labels

        Returns:
            grad (np.ndarray): Gradient of loss function w.r.t ŷ
        """
        grad = 2 * (predict - labels) / labels.size

        return grad
