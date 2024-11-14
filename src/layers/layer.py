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
    def output(self) -> np.ndarray:
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
        random_weights = np.random.randn(self._hidden_units, self._input_units)
        scaling_factor = np.sqrt(2.0 / self._input_units)
        self._weights = random_weights * scaling_factor
        self._bias = np.zeros((self._hidden_units, 1))

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
        self._rate = rate
        self._mask = None
        self._training = training
        self._output = None

    @property
    def dropout_rate(self):
        """
        Obtain the dropout rate
        """
        return self._rate

    @dropout_rate.setter
    def dropout_rate(self, rate: float):
        """
        Set the dropout rate

        Args:
            rate (float): Dropout rate between [0, 1)
        """
        self._rate = rate

    @property
    def mask(self) -> np.ndarray:
        """
        Get the current dropout mask

        The mask should be generated during forward propagation when training

        Returns:
            np.ndarray: The dropout mask used during training
        """
        return self._mask

    @property
    def dropout_training_mode(self):
        return self._training

    @dropout_training_mode.setter
    def dropout_training_mode(self, training: bool):
        """
        Specifies whether the model is in training

        Args:
            training (bool): If performing inference, set to False
        """
        self._training = training

    @property
    def output(self):
        """
        np.ndarray: The output of the dense layer after forward pass
        """
        if self._output is None:
            raise ValueError(
                "Output has not been computed yet, run a forward pass first."
                )
        return self._output

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Applies dropout to the input data

        Args:
            data (np.ndarray): Input data to the layer

        Returns:
            np.ndarray: Output after applying dropout
        """
        if self._training:
            # Initialise a uniform probability mask with same shape as data
            prob_mask = np.random.rand(*data.shape)

            # Create a mask where each element is zero with probability `rate`
            self._mask = (prob_mask > self._rate) / (1.0 - self._rate)

            # Element-wise product between scaled neurons and data
            self._output = data * self._mask
        else:
            # No dropout applied during inference
            self._output = data

        # Return the layer output
        return self._output
