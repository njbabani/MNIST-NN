# -*- coding: utf-8 -*-

"""
This script demonstrates the various features
of the neural network implemented from scratch

It performs the following steps:
    - Loads and preprocesses the MNIST dataset
    - Defines the neural network architecture with multiple layers
    - Compiles the model with a specified loss function and optimiser
    - Trains the model and employs callbacks for logging and early stopping
    - Evaluates the model on the test data and prints the loss and accuracy
    - Generates predictions on the test data
    - Displays samples of correct and incorrect predictions

Functions:
    main(): The main function that executes all steps

Usage:
    Run this script directly to train and evaluate the neural network:
        python main.py
"""

from datasets.data import load_mnist_data
from src.layers.layer import Dense, Dropout
from src.layers.activation import ReLU, Softmax
from src.layers.loss import CCE
from src.optimisation.optimiser import SGD
from src.models.model import FeedForwardNN
from src.common.callback import ProgbarLogger, EarlyStopping
from src.common.utils import display_images


def main():
    """
    Executes the main workflow:
        - Loads and preprocesses data
        - Defines and compiles the neural network model
        - Trains the model with specified callbacks
        - Evaluates the model on the test dataset
        - Displays sample predictions
    """

    # Step 1: Load data
    x_train, y_train, x_test, y_test = load_mnist_data(
        verbose=True,
        encode=True
    )

    # Step 2: Define model parameters
    output_units = y_train.shape[0]
    epochs = 5
    batch_size = 32
    l2_regulariser = 0.1
    alpha = 0.01

    # Step 3: Define hidden layers
    hidden_layers = [
        # First hidden layer with more neurons
        Dense(256),
        ReLU(),

        # Second layer with dropout
        Dropout(0.2),
        Dense(128),
        ReLU(),

        # Third dense layer
        Dense(64),
        ReLU(),

        # Output layer with softmax for categorical classification
        Dense(output_units),
        Softmax()
    ]

    # Step 4: Compile the model
    ANN = FeedForwardNN(layers=hidden_layers, lambda_reg=l2_regulariser)
    ANN.compile(loss=CCE(), optimiser=SGD(learning_rate=alpha))

    # Step 5: Add callbacks
    early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=True)
    progbar_logger = ProgbarLogger(verbose=True, total_epochs=epochs)
    callbacks = [progbar_logger, early_stopping]

    # Step 6: Train the model
    ANN.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    # Step 7: Evaluate the model
    test_loss, test_accuracy = ANN.evaluate(x_test, y_test)

    # Print results
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Step 8: Generate predictions
    y_pred = ANN.predict(x_test)

    # Step 9: Display some correct predictions
    display_images(
        images=x_test,
        y_hat=y_pred,
        y=y_test,
        is_correct=True,
        num_images=5,
        title='Correct Predictions'
    )

    # Step 10: Display some incorrect predictions
    display_images(
        images=x_test,
        y_hat=y_pred,
        y=y_test,
        is_correct=False,
        num_images=5,
        title='Incorrect Predictions'
    )


if __name__ == '__main__':
    main()
