# -*- coding: utf-8 -*-
"""
Module for defining a Feedforward Neural Network model

Classes:
    Model: Abstract base class for all models
    NeuralNetwork: Feedforward Neural Network
"""

from abc import ABC, abstractmethod
from typing import List, Union, Tuple
import numpy as np
import os
import pickle
from src.layers.layer import Layer, Dense
from src.layers.activation import Activation
from src.layers.cost import Cost
from src.optimisation.optimiser import Optimiser
from src.common.callback import Callback


class Model(ABC):
    """
    Abstract base model class for all neural network models

    Methods:
        learning_rate: Getter and setter for learning rate
        __call__: Perform a forward pass
        fit: Train the model on a dataset
        predict: Make predictions on new data
        evaluate: Evaluate the model on a test dataset
        back_propagate: Perform backpropagation to calculate gradients
        update_weights: Update the weights using the optimiser
        save_model: Save the model to a file
        load_model: Load a model from a file
    """

    @property
    @abstractmethod
    def learning_rate(self):
        pass

    @learning_rate.setter
    @abstractmethod
    def learning_rate(self, value: float):
        pass

    @abstractmethod
    def __call__(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        verbose: bool,
        callbacks: List[Callback],
    ):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        pass

    @abstractmethod
    def back_propagate(self, y_hat: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def update_weights(self):
        pass


class NeuralNetwork(Model):
    """
    Keras-like NeuralNetwork model for neural networks

    Attributes:
        layers (List[Union[Layer, Activation]]): List of layers and activations
        loss (Cost): Loss function
        optimiser (Optimiser): Optimiser for weight updates
        num_examples (int): Number of training examples
    """

    def __init__(self, layers: List[Union[Layer, Activation]] = None):
        """
        Initialises the NeuralNetwork model with a list of layers

        Args:
            layers (List[Union[Layer, Activation]], optional): List of layers
        """
        self.layers = layers if layers else []
        self.loss = None
        self.optimiser = None
        self.num_examples = None

    def add(self, layer: Union[Layer, Activation]):
        """
        Add a layer or activation function to the model

        Args:
            layer (Union[Layer, Activation]): The layer or activation to add
        """
        self.layers.append(layer)

    def compile(self, loss: Cost, optimiser: Optimiser):
        """
        Compile the model with a loss function and an optimiser

        Args:
            loss (Cost): Loss function
            optimiser (Optimiser): Optimiser to use
        """
        self.loss = loss
        self.optimiser = optimiser

    @property
    def learning_rate(self) -> float:
        return self.optimiser.learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float):
        self.optimiser.learning_rate = value

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through the model

        Args:
            data (np.ndarray): Input data

        Returns:
            np.ndarray: Output of the model
        """
        output = data
        for layer in self.layers:
            output = layer(output)
        return output

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        verbose: bool = False,
        callbacks: List[Callback] = [],
        validation_data: Union[None, Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """
        Train the model on the provided dataset

        Args:
            X (np.ndarray): Training data
            y (np.ndarray): Training labels
            epochs (int): Number of training epochs
            verbose (bool): If True, prints progress
            callbacks (List[Callback]): List of callbacks for training
            validation_data (optional): Validation dataset
        """
        self.num_examples = X.shape[-1]

        for epoch in range(epochs):
            y_hat = self(X)
            loss_value = self.loss(y_hat, y)
            self.back_propagate(y_hat, y)
            self.update_weights()

            for callback in callbacks:
                callback.on_epoch_end(epoch, {"loss": float(loss_value)})

            if validation_data:
                X_val, y_val = validation_data
                val_y_hat = self(X_val)
                val_loss = self.loss(val_y_hat, y_val)
                if verbose:
                    print(
                        f"Epoch {epoch + 1}/{epochs}, "
                        f"Loss: {loss_value:.4f}, Val Loss: {val_loss:.4f}"
                    )
            elif verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss_value:.4f}")

    def back_propagate(self, y_hat: np.ndarray, y: np.ndarray):
        """
        Perform backpropagation to calculate gradients

        Args:
            y_hat (np.ndarray): Predictions from forward pass
            y (np.ndarray): Ground truth labels
        """
        da = self.loss.gradient(y_hat, y)

        for layer in reversed(self.layers):
            if isinstance(layer, Activation):
                da = da * layer.gradient(layer.output)
            elif isinstance(layer, Dense):
                prev_output = (
                    self.layers[self.layers.index(layer) - 1].output
                    if self.layers.index(layer) > 0 else y_hat
                )
                dz = da
                layer.grad_weights = np.dot(dz, prev_output.T)
                layer.grad_weights /= self.num_examples
                layer.grad_bias = np.mean(dz, axis=1, keepdims=True)
                da = np.dot(layer.weights.T, dz)

    def update_weights(self):
        """
        Update the weights and biases of each layer using the optimiser
        """
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.update(self.optimiser)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data

        Args:
            X (np.ndarray): Input data

        Returns:
            np.ndarray: Predictions
        """
        return self(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the model on the test dataset

        Args:
            X (np.ndarray): Test data
            y (np.ndarray): Test labels

        Returns:
            float: Loss value
        """
        y_hat = self(X)
        return self.loss(y_hat, y)

    def save_model(self, dir: str = "params"):
        """
        Save the model to a structured directory

        Args:
            dir (str): The directory to save the model in
        """
        if not os.path.exists(dir):
            os.makedirs(dir)

        config_path = os.path.join(dir, "config.pkl")
        with open(config_path, "wb") as config_file:
            pickle.dump({
                "layers": len(self.layers),
                "loss": self.loss.__class__.__name__,
                "optimiser": self.optimiser.__class__.__name__,
                "learning_rate": self.optimiser.learning_rate,
            }, config_file)

        for idx, layer in enumerate(self.layers):
            if isinstance(layer, Layer):
                np.save(
                    os.path.join(dir, f"layer_{idx}_weights.npy"),
                    layer.weights
                )
                np.save(
                    os.path.join(dir, f"layer_{idx}_bias.npy"),
                    layer.bias
                )

        print(f"Model saved to {dir}")

    @staticmethod
    def load_model(dir: str) -> "NeuralNetwork":
        """
        Load a model from a structured directory

        Args:
            dir (str): The directory to load the model from

        Returns:
            NeuralNetwork: The loaded model
        """
        with open(os.path.join(dir, "config.pkl"), "rb") as config_file:
            config = pickle.load(config_file)

        model = NeuralNetwork()
        loss_class = getattr(
            __import__("src.layers.cost"),
            config["loss"]
        )
        optimiser_class = getattr(
            __import__("src.optimisation.optimiser"), config["optimiser"]
        )
        model.loss = loss_class()
        model.optimiser = optimiser_class(
            learning_rate=config["learning_rate"]
        )

        return model
