""" Provides the CNN class. """

from typing import Literal

import torch


class CNN:
    """ Implements Convolutional Neural Network. """


    def __init__(self, task: Literal['classification', 'regression']):
        """ Initializes the model hyperparameters.

        Args:
        """

        # Validate the passed arguments
        assert task in ['classification', 'regression'], f'Got unrecognized value for task {task}'

        # Store the passed arguments
        self.task = task

        # Initialize the model parameters to None


    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """ Fits the model for the given training data. """


    def forward(self, X: torch.Tensor, index: int | None = None):
        """ Performs forward propagation on the input image and returns the output of the
        final layer (or an intermediate layer optionally). """
