# -*- coding: utf-8 -*-

"""
This module defines the Differentiable abstract base class, which is used to
enforce a consistent interface for components in a neural network that need to
compute gradients

Classes:
    Differentiable: Abstract base class requiring the implementation of a
                    gradient method for any class that inherits from it
"""

from abc import ABC, abstractmethod
from typing import Any


class Differentiable(ABC):
    """
    Abstract base class for components that support differentiation

    This class enforces the implementation of a gradient method, ensuring
    that any subclass can compute gradients - useful for activation functions
    and loss functions

    Methods:
        gradient(*args, **kwargs): Computes the gradient of the function.
    """

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
