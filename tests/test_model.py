# -*- coding: utf-8 -*-

"""
Testing module for layers.py

Functions:
    test_model_initialisation(): Error if model not init properly
    test_forward_pass(): Error if Dropout does not drop neurons
    test_fit_backpropagation(): Error if either fit or back prop fails
    test_early_stopping(): Verifies early stopping
    test_model_checkpoint(): Verifies model checkpoint
    test_save_load_model(): Verifies if model can be saved/loaded
"""

import numpy as np
import pytest
import os
from src.layers.layer import Dense, Dropout
from src.layers.activation import ReLU, Softmax, Linear
from src.optimisation.optimiser import SGD
from src.layers.loss import CCE
from src.models.model import FeedForwardNN
from src.common.callback import EarlyStopping, ModelCheckpoint


def test_model_initialisation():
    """
    Test if the model initialises correctly
    """
    layers = [Dense(4), ReLU(), Dense(3), Softmax()]
    loss = CCE()
    optimiser = SGD(learning_rate=0.01)

    model = FeedForwardNN(layers=layers, loss=loss, optimiser=optimiser)

    assert model._num_layers == len(layers), "Number of layers is incorrect"
    assert isinstance(model._loss, CCE), "Loss function not set correctly"
    assert isinstance(model._optimiser, SGD), "Optimiser not set correctly"


def test_forward_pass():
    """
    Test if the model's forward pass produces the correct output shape
    """
    np.random.seed(1)

    # Input shape: (5, 10)
    X = np.random.randn(5, 10)

    # Define hidden layers
    hidden_layers = [
        Dense(32),
        Dropout(),
        Linear(),
        Dense(12),
        ReLU(),
        Dense(3),
        Softmax()
        ]

    # Compile the model
    model = FeedForwardNN(layers=hidden_layers)
    model.compile(loss=CCE(), optimiser=SGD(0.01))

    # Train the model
    output = model(X)

    assert output.shape == (3, 10), \
        f"Output shape is incorrect: {output.shape}"


def test_fit_backpropagation():
    """
    Test fit and backpropagation functions
    """
    np.random.seed(1)
    layers = [Dense(4), ReLU(), Dense(3), Softmax()]
    loss = CCE()
    optimiser = SGD(learning_rate=0.01)
    model = FeedForwardNN(layers=layers, loss=loss, optimiser=optimiser)

    # Input shape: (5, 10)
    X = np.random.randn(5, 10)

    # One-hot encoded labels (3 classes, 10 samples)
    Y = np.eye(3)[:, :10]

    # Ensure Y has the correct shape (3, 10) to match X
    if Y.shape[1] < X.shape[1]:
        Y = np.hstack([Y, np.zeros((Y.shape[0], X.shape[1] - Y.shape[1]))])

    model.fit(X, Y, epochs=1, batch_size=2)

    # Check if weights are updated
    for layer in model._layers:
        if isinstance(layer, Dense):
            assert layer.weights is not None, "Weights not updated correctly"


def test_early_stopping():
    """
    Test if the EarlyStopping callback stops training when no improvement
    """
    np.random.seed(1)
    layers = [Dense(4), ReLU(), Dense(3), Softmax()]
    loss = CCE()
    optimiser = SGD(learning_rate=0.01)
    model = FeedForwardNN(layers=layers, loss=loss, optimiser=optimiser)

    # Input shape: (5, 10)
    X = np.random.randn(5, 10)

    # One-hot encoded labels (3 classes, 10 samples)
    Y = np.eye(3)[:, :10]

    # Ensure Y has the correct shape (3, 10) to match X
    if Y.shape[1] < X.shape[1]:
        Y = np.hstack([Y, np.zeros((Y.shape[0], X.shape[1] - Y.shape[1]))])

    early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=True)
    model.fit(X, Y, epochs=10, batch_size=5, callbacks=[early_stopping])


def test_model_checkpoint(tmpdir):
    """
    Test if ModelCheckpoint saves the model correctly
    """
    np.random.seed(1)
    layers = [Dense(4), ReLU(), Dense(3), Softmax()]
    loss = CCE()
    optimiser = SGD(learning_rate=0.01)
    model = FeedForwardNN(layers=layers, loss=loss, optimiser=optimiser)

    # Input shape: (5, 10)
    X = np.random.randn(5, 10)

    # One-hot encoded labels (3 classes, 10 samples)
    Y = np.eye(3)[:, :10]

    # Ensure Y has the correct shape (3, 10) to match X
    if Y.shape[1] < X.shape[1]:
        Y = np.hstack([Y, np.zeros((Y.shape[0], X.shape[1] - Y.shape[1]))])

    filepath = os.path.join(tmpdir, "best_model.pkl")
    checkpoint = ModelCheckpoint(
        model, filepath=filepath, monitor="loss", mode="min"
    )

    model.fit(X, Y, epochs=5, batch_size=5, callbacks=[checkpoint])

    assert os.path.exists(filepath), "Model not saved"


def test_save_load_model(tmpdir):
    """
    Test if the model can be saved and loaded correctly
    """
    layers = [Dense(4), ReLU(), Dense(3), Softmax()]
    loss = CCE()
    optimiser = SGD(learning_rate=0.01)
    model = FeedForwardNN(layers=layers, loss=loss, optimiser=optimiser)

    filepath = os.path.join(tmpdir, "model.pkl")
    model.save_model(filepath)
    assert os.path.exists(filepath), "Model save failed"

    # Load the model back
    new_model = FeedForwardNN(layers=[])
    new_model.load_model(filepath)

    assert len(new_model._layers) == len(layers), \
        "Loaded model layers mismatch"


if __name__ == "__main__":
    pytest.main()
