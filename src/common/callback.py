# -*- coding: utf-8 -*-
"""
This module defines various callback functions for a neural network

Classes:
    Callback: Abstract base class for all callbacks
    LearningRateScheduler: Adjusts learning rate based on epoch
    ModelCheckpoint: Saves model based on performance metrics
    EarlyStopping: Stops training if no improvement is observed
    ProgbarLogger: Displays a progress bar during training
"""

from abc import ABC, abstractmethod
import numpy as np
import sys


class Callback(ABC):
    """
    Abstract base class for all callbacks

    All callbacks must implement the on_epoch_end method
    """

    @abstractmethod
    def on_epoch_begin(self, epoch: int, logs: dict = None):
        pass

    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: dict = None):
        pass

    @abstractmethod
    def on_train_begin(self, logs: dict = None):
        pass

    @abstractmethod
    def on_train_end(self, logs: dict = None):
        pass


class LearningRateScheduler(Callback):
    """
    Adjusts the learning rate according to a schedule
    """

    def __init__(self, schedule, verbose: bool = False):
        """
        Initialises the learning rate scheduler

        Args:
            schedule (function): Returns a new learning rate based on epoch
            verbose (bool): Prints additional information
        """
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch: int, logs: dict = None):
        pass

    def on_train_begin(self, logs: dict = None):
        pass

    def on_train_end(self, logs: dict = None):
        pass

    def on_epoch_end(self, epoch: int, logs: dict = None):
        """
        Adjusts the learning rate at the end of each epoch

        Args:
            epoch (int): Current epoch
            logs (dict): Training metrics
        """
        if logs and "learning_rate" in logs:
            new_lr = self.schedule(epoch)
            logs["learning_rate"] = new_lr
            if self.verbose:
                print(
                    f"\nEpoch {epoch + 1}: Learning rate adjusted {new_lr:.5f}"
                )


class ModelCheckpoint(Callback):
    """
    Saves the model when a monitored metric improves
    """

    def __init__(
        self,
        model,
        filepath: str,
        monitor: str = "loss",
        mode: str = "min",
        verbose: bool = True,
    ):
        """
        Initialises the model checkpoint callback

        Args:
            model (NeuralNetwork): The model instance to save
            filepath (str): Path to save the model
            monitor (str): Metric to monitor (e.g., 'loss', 'accuracy')
            mode (str): 'min' or 'max' to decide if lower or higher is better
            verbose (bool): Whether to print messages when saving the model
        """
        self.model = model
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best = np.inf if mode == "min" else -np.inf

    def on_epoch_begin(self, epoch: int, logs: dict = None):
        pass

    def on_train_begin(self, logs: dict = None):
        pass

    def on_train_end(self, logs: dict = None):
        pass

    def on_epoch_end(self, epoch: int, logs: dict = None):
        """
        Saves the model if the monitored metric improves

        Args:
            epoch (int): The current epoch
            logs (dict): A dictionary containing training metrics
        """
        if logs:
            curr = logs.get(self.monitor)
            if curr is None:
                if self.verbose:
                    print(f"Warning: {self.monitor} not found in logs")
                return

            if (self.mode == "min" and curr < self.best) or (
                self.mode == "max" and curr > self.best
            ):
                self.best = curr
                self.model.save_model(self.filepath)
                if self.verbose:
                    print(
                        f"\nEpoch {epoch+1}: {self.monitor} "
                        f"improved to {curr:.5f}"
                    )


class EarlyStopping(Callback):
    """
    Stops training if the monitored metric does not improve
    """

    def __init__(
        self,
        monitor: str = "loss",
        patience: int = 5,
        mode: str = "min",
        verbose: bool = True,
    ):
        """
        Initialises the early stopping callback

        Args:
            monitor (str): Metric to monitor (e.g., 'loss')
            patience (int): Epochs with no improvement before stopping
            mode (str): 'min' or 'max' to decide if lower or higher is better
            verbose (bool): Whether to print messages when stopping
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.best = np.inf if mode == "min" else -np.inf
        self.wait = 0

    def on_epoch_begin(self, epoch: int, logs: dict = None):
        pass

    def on_train_begin(self, logs: dict = None):
        pass

    def on_train_end(self, logs: dict = None):
        pass

    def on_epoch_end(self, epoch: int, logs: dict = None):
        """
        Checks if training should be stopped early

        Args:
            epoch (int): The current epoch
            logs (dict): A dictionary containing training metrics
        """
        if logs:
            curr = logs.get(self.monitor)
            if curr is None:
                if self.verbose:
                    print(f"Warning: {self.monitor} not found in logs")
                return

            if (self.mode == "min" and curr < self.best) or (
                self.mode == "max" and curr > self.best
            ):
                self.best = curr
                self.wait = 0
            else:
                self.wait += 1
                if self.verbose:
                    print(
                        f"\nEpoch {epoch+1}: No improvement in {self.monitor}"
                    )
                    print(f"Patience {self.wait}/{self.patience}")

            if self.wait >= self.patience:
                if self.verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                raise StopIteration("Training stopped due to early stopping")


class ProgbarLogger(Callback):
    """
    Displays a simple progress bar during training
    """

    def __init__(self, verbose: bool = True, total_epochs: int = None):
        """
        Initialises the progress bar logger

        Args:
            verbose (bool): Whether to show the progress bar
            total_epochs (int): Total number of epochs
        """
        self.verbose = verbose
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch: int, logs: dict = None):
        pass

    def on_train_begin(self, logs: dict = None):
        if self.verbose and self.total_epochs:
            print("Training started...")

    def on_train_end(self, logs: dict = None):
        if self.verbose:
            print("\nTraining complete")

    def on_epoch_end(self, epoch: int, logs: dict = None):
        """
        Updates the progress bar at the end of each epoch

        Args:
            epoch (int): Current epoch
            logs (dict): Training metrics
        """
        if self.verbose and self.total_epochs:
            progress = (epoch + 1) / self.total_epochs
            bar_length = 40
            block = int(round(bar_length * progress))
            progress_bar = "#" * block + "-" * (bar_length - block)

            metrics = ", ".join(
                f"{key}: {value:.4f}"
                for key, value in logs.items()
                if isinstance(value, (int, float))
            )
            sys.stdout.write(
                f"\r[{progress_bar}] {int(progress * 100)}% - Epoch {epoch+1}"
                f" /{self.total_epochs} - {metrics}"
            )
            sys.stdout.flush()
