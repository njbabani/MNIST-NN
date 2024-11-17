# -*- coding: utf-8 -*-

"""
Testing module for callbacks.py

Functions:
    test_learning_rate_scheduler: Tests LearningRateScheduler callback
    test_model_checkpoint: Tests ModelCheckpoint callback
    test_early_stopping: Tests EarlyStopping callback
    test_progbar_logger: Tests ProgbarLogger callback
"""

import numpy as np
import pytest
from unittest.mock import Mock
from src.common.callback import (
    LearningRateScheduler,
    ModelCheckpoint,
    EarlyStopping,
    ProgbarLogger
)


def test_learning_rate_scheduler():
    """
    Test the LearningRateScheduler callback
    """
    # Define a simple learning rate schedule function
    def schedule(epoch):
        return 0.01 * (0.9 ** epoch)

    # Initialize the callback with the schedule
    lr_scheduler = LearningRateScheduler(schedule, verbose=True)
    logs = {"learning_rate": 0.01}

    # Simulate epochs and check if learning rate adjusts
    for epoch in range(5):
        lr_scheduler.on_epoch_end(epoch, logs)
        expected_lr = schedule(epoch)
        assert logs["learning_rate"] == pytest.approx(expected_lr, rel=1e-3), \
            f"Learning rate not updated correctly at epoch {epoch}"


def test_model_checkpoint():
    """
    Test the ModelCheckpoint callback
    """
    # Make a mock model with a save_model method
    mock_model = Mock()
    mock_model.save_model = Mock()

    # Initialise ModelCheckpoint with a mocked model
    checkpoint = ModelCheckpoint(
        model=mock_model,
        filepath="mock_model.pkl",
        monitor="loss",
        mode="min",
        verbose=True
    )

    # Simulate training epochs with decreasing loss
    logs = {"loss": np.array([0.5])}
    for epoch in range(5):
        # Simulate loss reduction
        logs["loss"] *= 0.9
        checkpoint.on_epoch_end(epoch, logs)

        # Check if model is saved when loss improves
        if epoch > 0:
            mock_model.save_model.assert_called_with("mock_model.pkl")

    # Ensure save_model was called at least once
    assert mock_model.save_model.call_count == 5, \
        "Model was not saved correctly"


def test_early_stopping():
    """
    Test the EarlyStopping callback
    """
    early_stopping = EarlyStopping(
        monitor="loss",
        patience=3,
        mode="min",
        verbose=True
    )
    logs = {"loss": np.array([1.0])}

    # Simulate training with no improvement
    with pytest.raises(StopIteration):
        for epoch in range(5):
            # No improvement
            logs["loss"] = np.array([[1.0]])
            early_stopping.on_epoch_end(epoch, logs)

    # Reset for another test with improvement
    early_stopping.wait = 0
    early_stopping.best = np.inf

    # Simulate training with gradual improvement
    logs = {"loss": np.array([0.5])}
    for epoch in range(5):
        # Simulate loss reduction
        logs["loss"] *= 0.9
        print(f"Epoch {epoch + 1}: Loss = {logs['loss'][0]}")
        try:
            early_stopping.on_epoch_end(epoch, logs)
        except StopIteration:
            pytest.fail("Early stopping triggered incorrectly")


def test_progbar_logger(capsys):
    """
    Test the ProgbarLogger callback
    """
    total_epochs = 5
    progbar_logger = ProgbarLogger(verbose=True, total_epochs=total_epochs)
    logs = {"loss": np.array([1.0]), "accuracy": np.array([1.0])}

    progbar_logger.on_train_begin()

    # Simulate training and capture the console output
    for epoch in range(total_epochs):
        progbar_logger.on_epoch_end(epoch, logs)
        captured = capsys.readouterr()
        assert f"Epoch {epoch + 1}" in captured.out, \
            "Progress bar output is incorrect"

    progbar_logger.on_train_end()
    captured = capsys.readouterr()
    assert "Training complete" in captured.out, \
        "Training completion message not printed"


if __name__ == "__main__":
    pytest.main()
