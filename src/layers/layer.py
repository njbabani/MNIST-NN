# -*- coding: utf-8 -*-

"""
This module describes the types of layers for the NN

Classes:
    Layer: Abstract base class for layers
    Dense: Implements a fully connected (dense) layer
    Dropout: Implements a dropout layer for regularisation
"""

from abc import ABC, abstractmethod
import numpy as np
from src.optimisation.optimiser import Optimiser


class Layer(ABC):
    """
    Parent class for NN layers

    Attributes:
        output (np.ndarray): Output of layer after linear forward propagation
    """
    @property
    @abstractmethod
    def layer_output(self) -> np.ndarray:
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


class Dense(Layer):
    """
    Fully connected (dense) layer implementation

    Attributes:
        hidden_units (int): Number of neurons in the dense layer
        input_units (int): Number of neurons in the input layer
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
        self.hidden_units = hidden_units
        self.input_units = None
        self.weights = None
        self.bias = None
        self.output = None
        self.dw = None
        self.db = None

    @property
    def layer_weights(self):
        """
        np.ndarray: The weights of the dense layer
        """
        return self.weights

    @layer_weights.setter
    def layer_weights(self, weights: np.ndarray):
        """
        np.ndarray: Sets the weights
        """
        self.weights = weights

    @property
    def layer_bias(self):
        """
        np.ndarray: The bias of the dense layer
        """
        return self.bias

    @layer_bias.setter
    def bias(self, bias: np.ndarray):
        """
        np.ndarray: Sets the biases
        """
        self.bias = bias

    @property
    def grad_weights(self):
        """
        np.ndarray: Gradient of the weights for backpropagation
        """
        return self.dw

    @grad_weights.setter
    def grad_weights(self, gradients: np.ndarray):
        """
        np.ndarray: Sets the gradients of the weights
        """
        self.dw = gradients

    @property
    def grad_bias(self):
        """
        np.ndarray: Gradient of the biases for backpropagation
        """
        return self.db

    @grad_bias.setter
    def grad_bias(self, gradients: np.ndarray):
        """
        np.ndarray: Sets the gradients of the biases
        """
        self.db = gradients

    @property
    def layer_output(self):
        """
        np.ndarray: The output of the dense layer after forward pass
        """
        return self.output

    def build(self, data: np.ndarray):
        """
        Initialises weights and biases using He initialisation

        Args:
            data (np.ndarray): Input data for determining the shape.
        """
        self.input_units = data.shape[0]
        random_weights = np.random.randn(self.hidden_units, self.input_units)
        scaling_factor = np.sqrt(2.0 / self.input_units)
        self.weights = random_weights * scaling_factor
        self.bias = np.zeros((self.hidden_units, 1))

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass for the dense layer

        Args:
            data (np.ndarray): Input data to the layer

        Returns:
            np.ndarray: Output of the layer after applying weights and biases
        """
        if self.weights is None:
            self.build(data)

        self.output = np.dot(self.weights, data) + self.bias
        return self.output

    def update(self, optimiser: Optimiser):
        """
        Updates weights and biases using the specified optimiser

        Args:
            optimiser (Optimiser): The optimiser used for updating parameters
        """
        optimiser.update_weights(self, self.grad_weights)
        optimiser.update_bias(self, self.grad_bias)


class Dropout(Layer):
    """
    Implements a Dropout layer for regularisation

    Attributes:
        rate (float): Dropout rate for neurons
        mask (np.ndarray): A copy used to drop neurons during the forward prop
    """
    def __init__(self, rate: float = 0.5, training: bool = True):
        """
        Initialises the Dropout layer

        Args:
            rate (float): The fraction of neurons to drop (defaults to 0.5)
            training (bool): Specifies model is training (defaults to True)
        """
        if not 0.0 <= rate < 1.0:
            raise ValueError("Dropout rate must be in the range [0, 1).")
        self.rate = rate
        self.mask = None
        self.training = training
        self.output = None

    @property
    def dropout_rate(self):
        """
        Obtain the dropout rate
        """
        return self.rate

    @dropout_rate.setter
    def dropout_rate(self, rate: float):
        """
        Set the dropout rate

        Args:
            rate (float): Dropout rate between [0, 1)
        """
        self.rate = rate

    @property
    def dropout_training_mode(self):
        return self.training

    @dropout_training_mode.setter
    def dropout_training_mode(self, training: bool):
        """
        Specifies whether the model is in training

        Args:
            training (bool): If performing inference, set to False
        """
        self.training = training

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Applies dropout to the input data

        Args:
            data (np.ndarray): Input data to the layer

        Returns:
            np.ndarray: Output after applying dropout
        """
        if self.training:
            # Initialise a uniform probability mask with same shape as data
            prob_mask = np.random.rand(*data.shape)

            # Create a mask where each element is zero with probability `rate`
            self.mask = (prob_mask > self.rate) / (1.0 - self.rate)

            # Element-wise product between scaled neurons and data
            self.output = data * self.mask
        else:
            # No dropout applied during inference
            self.output = data

        # Return the layer output
        return self.output

    @property
    def layer_output(self):
        """
        np.ndarray: The output after the most recent forward pass
        """
        return self.output
