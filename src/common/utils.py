# -*- coding: utf-8 -*-

"""
Defines useful utility functions used throughout the program
"""

import numpy as np
import matplotlib.pyplot as plt


def accuracy_metric(y_hat: np.ndarray, y: np.ndarray) -> float:
    """
    Computes accuracy metric for an entire dataset

    Args:
        y_hat (np.ndarray): Predicted outputs
        y (np.ndarray): True labels

    Returns:
        accuracy (float): Accuracy of NN
    """

    # Convert predictions to class labels
    predicted_labels = np.argmax(y_hat, axis=0)
    true_labels = np.argmax(y, axis=0)

    # Compute accuracy
    correct_predictions = np.sum(predicted_labels == true_labels)
    total_predictions = y.shape[1]
    accuracy = correct_predictions / total_predictions

    return float(accuracy)


def display_images(
        images: np.ndarray,
        y_hat: np.ndarray,
        y: np.ndarray,
        is_correct=True,
        num_images=5,
        title=""
):
    """
    Creates plots to compare predicted images with the actual images

    Args:
        images (np.ndarray): Can be from training or testing
        y_hat (np.ndarray): Predicted output
        y (np.ndarray): True labels
        is_correct (bool): Whether to displace correct images (default True)
    """

    # Convert probabilities to class labels
    pred_labels = np.argmax(y_hat, axis=0)
    true_labels = np.argmax(y, axis=0)

    # Identify correct predictions
    correct_predictions = (pred_labels == true_labels)

    # Obtain the indicies for the correct/incorrect predictions
    if is_correct:
        indices = np.where(correct_predictions)[0]
    else:
        indices = np.where(~correct_predictions)[0]

    # Plot the figures
    plt.figure(figsize=(10, 2 * num_images))
    for i, idx in enumerate(indices[:num_images]):
        # Get the image and reshape it
        image = images[:, idx].reshape(28, 28)

        # Create subplots to show each image
        plt.subplot(num_images, 1, i + 1)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.title(
            f"Predicted Label: {pred_labels[idx]},"
            f" True Label: {true_labels[idx]}"
        )
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
