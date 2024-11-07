# -*- coding: utf-8 -*-

'''
Optimiser module to improve NN performance
'''

from abc import ABC, abstractmethod
import numpy as np


class Optimiser(ABC):
    '''Parent class for optimisers'''

    def __init__(self, learning_rate: float):
        '''
        Initialise the optimiser with a learning rate

        Args:
            learning_rate (float): Learning rate for the optimiser
        '''
        self.learning_rate = learning_rate

        @abstractmethod
        def update_weights(self, layer, grad_weights: np.ndarray):
            """Abstract method to update weights, implemented by subclasses."""
            pass

        @abstractmethod
        def update_bias(self, layer, grad_bias: np.ndarray):
            """Abstract method to update biases, implemented by subclasses."""
            pass
