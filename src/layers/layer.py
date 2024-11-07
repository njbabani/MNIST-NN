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
        """np.ndarray: Output of layer after most recent forward pass"""
        pass

    @abstractmethod
    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass for the layer

        Args:
            data (np.ndarray): Input data for the layer (i.e. features)

        Returns:
            np.ndarray: Output of the layer after linear forward propagation
        """
        pass

    @abstractmethod
    def build(self, input_tensor: np.ndarray):
        """
        Builds the layer with weights and biases based on the input shape

        Args:
            data (np.ndarray): Input data for determining the shape
        """
        pass

    @abstractmethod
    def update(self, optimizer: Optimiser):
        """
        Updates the layer's weights and biases based on selected optimiser

        Args:
            optimiser (Optimiser): The optimiser used to update parameters
        """
        pass
