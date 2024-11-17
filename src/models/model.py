# -*- coding: utf-8 -*-

"""
This module defines the neural network architecture

Classes:
    Model: Abstract base class for all Neural Networks
    FeedForwardNN: Feedforward neural network
"""

import pickle
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Tuple
from src.layers.activation import Activation, Softmax
from src.common.utils import accuracy_metric
from src.common.callback import Callback
from src.layers.layer import Layer, Dense, Dropout
from src.layers.loss import Loss, CCE
from src.optimisation.optimiser import Optimiser


class Model(ABC):
    """
    Abstract base class for neural network models

    Attributes:
        layers (List[Layer]): List of layers added to the model
        loss_function (Callable): Function to compute the loss
        optimiser (Optimiser): Optimiser for training the model
    """

    def __init__(self):
        self.layers: List[Layer] = []
        self.loss_function: Loss = None
        self.optimiser: Optimiser = None

    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the model

        Args:
            X (np.ndarray): Input data

        Returns:
            np.ndarray: Output of the model
        """
        pass

    @abstractmethod
    def compile(self, loss: Loss, optimiser: Optimiser):
        """
        Configures the model for training

        Args:
            loss (Loss): Loss function
            optimiser (Optimiser): Optimiser to use for training
        """
        pass

    @abstractmethod
    def add(self, layer: Layer):
        """
        Adds a layer to the model

        Args:
            layer (Layer): The layer to be added
        """
        pass

    @abstractmethod
    def back_propagate(self, Y: np.ndarray):
        """
        Performs backpropagation through the model

        For y_hat, we simply use model output
        Args:
            Y (np.ndarray): Actual labels
        """
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int, batch_size: int):
        """
        Trains the model

        Args:
            X (np.ndarray): Training data
            Y (np.ndarray): Labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the trained model

        Args:
            X (np.ndarray): Input data

        Returns:
            np.ndarray: Predictions
        """
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Evaluates the model on the given dataset and returns the loss

        Args:
            X (np.ndarray): Input data (shape: input_size x num_examples)
            Y (np.ndarray): Truth labels (shape: output_size x num_examples)

        Returns:
            np.ndarray: Loss of the model on the dataset
        """
        pass


class FeedForwardNN(Model):
    def __init__(
            self,
            layers: List[Layer],
            loss: Optional[Loss] = None,
            optimiser: Optional[Optimiser] = None,
            lambda_reg: float = 0.0
    ):
        """
        Initialises NN with layers, loss function, optimiser, and reg factor

        Args:
            layers (List[Layer]): List of layers to add to the model
            loss_function (Loss): Loss function
            optimiser (Optimiser): Optimiser for training
            lambda_reg (float): Regularisation factor
        """
        super().__init__()
        self._layers = layers
        self._num_layers = len(layers)
        self._loss = loss
        self._optimiser = optimiser
        self._lambda_reg = lambda_reg
        self._input = None
        self._output = None
        self._num_examples = None

    def add(self, layer: Layer):
        """
        Adds a new layer to the model

        Args:
            layer (Layer): The layer to be added
        """
        self._layers.append(layer)

    def compile(self, loss: Loss, optimiser: Optimiser):
        """
        Compiles the model for training

        You specify the loss and optimiser to use for training

        Args:
            loss (Callable): Loss function to use
            optimiser (Optimiser): Optimiser for training
        """
        self._loss = loss
        self._optimiser = optimiser

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the model

        Args:
            X (np.ndarray): Input data

        Returns:
            np.ndarray: Output of the model
        """

        if self._num_examples is None:
            # When initially calling NN, defines number of examples
            self._num_examples = X.shape[-1]

        # Store the input for backpropagation
        self._input = X

        # Input into first hidden layer is output of previous layer (X)
        A = X

        # Forward propagate for different layers and activation functions
        for layer in self._layers:
            A = layer(A)

        # Returns final output
        self._output = A
        return self._output

    def back_propagate(self, Y: np.ndarray):
        """
        Performs backpropagation to update the weights and biases of the model

        Args:
            Y (np.ndarray): Truth labels (shape: output_size x num_examples)
        """
        # Step 1: Calculate the initial gradient of the loss w.r.t. output
        dA = self._loss.gradient(self._output, Y)

        # Step 2: Loop through the layers in reverse order for backpropagation
        for index in reversed(range(self._num_layers)):
            layer = self._layers[index]

            if isinstance(layer, Activation):
                if isinstance(layer, Softmax) and isinstance(self._loss, CCE):
                    # Use the optimised gradient for softmax + CCE combination
                    dA = dA
                else:
                    # For other activations, propagate the gradient through
                    dA = dA * layer.gradient(layer.output)

            # Handle Dense types layer
            elif isinstance(layer, Dense):
                # For Dense, find the grad w.r.t weights, biases, and inputs
                if index == 0:
                    prev_layer_output = self._input
                else:
                    prev_layer = self._layers[index - 1]
                    prev_layer_output = prev_layer.output

                # Step 3: Calculate dZ if layer is Dense
                dZ = dA

                # Step 4: Calculate gradients for weights and biases
                layer.grad_weights = np.divide(
                    np.dot(dZ, prev_layer_output.T), self._num_examples
                )
                layer.grad_bias = np.mean(dZ, axis=1, keepdims=True)

                # Step 5: Apply L2 regularisation if specified
                if self._lambda_reg > 0:
                    layer.grad_weights += np.divide(
                        self._lambda_reg * layer.weights,
                        self._num_examples
                    )

                # Step 6: Calculate dA for the previous layer
                dA = np.dot(layer.weights.T, dZ)

            # Handle Dropout layer
            elif isinstance(layer, Dropout):
                # For Dropout layers, adjust dA by the dropout mask
                if layer.dropout_training_mode:
                    dA *= layer.mask

        # Step 7: After computing all gradients, update weights and biases
        for layer in self._layers:
            if isinstance(layer, Dense):
                self._optimiser.update_weights(layer, layer.grad_weights)
                self._optimiser.update_bias(layer, layer.grad_bias)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates predictions for the given input data using the trained model

        Args:
            X (np.ndarray): Input data (shape: input_size x num_examples)

        Returns:
            np.ndarray: Model predictions (shape: output_size x num_examples)
        """
        # Ensure that the model is in evaluation mode (no dropout)
        for layer in self._layers:
            if isinstance(layer, Dropout):
                layer.dropout_training_mode = False

        # Perform a forward pass through the network
        A = X
        for layer in self._layers:
            A = layer(A)

        return A

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
        """
        Evaluates the model on the given dataset and returns the loss

        Args:
            X (np.ndarray): Input data (shape: input_size x num_examples)
            Y (np.ndarray): Truth labels (shape: output_size x num_examples)

        Returns:
            np.ndarray: Loss of the model on the dataset
        """
        # Generate predictions
        y_hat = self.predict(X)

        # Calculate loss
        loss = self._loss(y_hat, Y)

        # Calculate accuracy metric
        accuracy = accuracy_metric(y_hat, Y)

        return (loss, accuracy)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int,
        batch_size: int,
        callbacks: list[Callback] = None,
    ):
        """
        Trains the model using the provided dataset

        Args:
            X (np.ndarray): Input data (shape: input_size x num_examples)
            Y (np.ndarray): True labels (shape: output_size x num_examples)
            epochs (int): Number of training epochs
            batch_size (int): Size of each mini-batch
            callbacks (list): List of callback instances
        """
        if callbacks is None:
            callbacks = []

        # Calculate the number of batches per epoch
        num_examples = X.shape[1]
        num_batches = num_examples // batch_size

        # Initialise logs for callbacks
        logs = {"loss": None, "learning_rate": self._optimiser.learning_rate}

        # Call the on_train_begin() for each callback
        for callback in callbacks:
            callback.on_train_begin(logs)

        # Training loop over epochs
        for epoch in range(epochs):
            # Call the on_epoch_begin() for each callback
            for callback in callbacks:
                callback.on_epoch_begin(epoch, logs)

            epoch_loss = 0
            correct_predictions = 0

            # Shuffle the dataset before each epoch
            indices = np.arange(num_examples)
            np.random.shuffle(indices)
            X = X[:, indices]
            Y = Y[:, indices]

            # Mini-batch training
            for batch in range(num_batches):
                start = batch * batch_size
                end = (batch + 1) * batch_size

                X_batch = X[:, start:end]
                Y_batch = Y[:, start:end]

                # Step 1: Forward pass
                predictions = self.__call__(X_batch)

                # Step 2: Calculate loss
                loss = self._loss(predictions, Y_batch)
                epoch_loss += loss

                # Step 3: Calculate accuracy for the batch
                predicted_labels = np.argmax(predictions, axis=0)
                true_labels = np.argmax(Y_batch, axis=0)

                predicted_labels = predicted_labels.flatten()
                true_labels = true_labels.flatten()

                correct_predictions += np.sum(predicted_labels == true_labels)

                # Step 3: Backward pass and weight update
                self.back_propagate(Y_batch)

            # Average loss for the epoch
            epoch_loss /= num_batches
            accuracy = correct_predictions / num_examples
            logs["loss"] = epoch_loss
            logs["accuracy"] = accuracy

            # Update optimiser's learning rate if adjusted by callback
            if (
                "learning_rate" in logs
                and self._optimiser.learning_rate != logs["learning_rate"]
            ):
                self._optimiser.learning_rate = logs["learning_rate"]

            # Early stopping or other stopping conditions
            try:
                for callback in callbacks:
                    callback.on_epoch_end(epoch, logs)
            except StopIteration:
                print(f"Training stopped early at epoch {epoch + 1}")
                break

            # Print progress
            print(
                " - Loss: {:.4f} - Accuracy: {:.2f} - LR: {:.5f}".format(
                    float(epoch_loss.item()),
                    accuracy * 100,
                    self._optimiser.learning_rate,
                )
            )

        # Call the on_train_end() for each callback
        for callback in callbacks:
            callback.on_train_end(logs)

    def save_model(self, filepath: str):
        """
        Saves the model's parameters (weights, biases) and architecture

        Args:
            filepath (str): Path where the model should be saved
        """
        # Prepare the data to be saved
        model_data = {
            "layers": self._layers,
            "loss": self._loss,
            "optimiser": self._optimiser,
            "lambda_reg": self._lambda_reg,
        }

        # Save the data to a file using pickle
        with open(filepath, 'wb') as file:
            pickle.dump(model_data, file)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Loads the model's parameters (weights, biases) and architecture

        Args:
            filepath (str): Path from where the model should be loaded
        """
        with open(filepath, 'rb') as file:
            model_data = pickle.load(file)

        self._layers = model_data["layers"]
        self._loss = model_data["loss"]
        self._optimiser = model_data["optimiser"]
        self._lambda_reg = model_data["lambda_reg"]

        print(f"Model loaded from {filepath}")
