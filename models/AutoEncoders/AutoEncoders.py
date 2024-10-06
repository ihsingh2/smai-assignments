# pylint: disable=invalid-name

""" Provides the AutoEncoders class. """

# pylint: enable=invalid-name


import sys
from typing import Self

import numpy as np
import numpy.typing as npt

#pylint: disable=wrong-import-position

PROJECT_DIR = '../..'
sys.path.insert(0, PROJECT_DIR)

from models.MLP import MLP

#pylint: enable=wrong-import-position


class AutoEncoders:
    """ Implements AutoEncoder. """


    def __init__(self):
        """ Initializes the model hyperparameters.

        Args:
            x: ...
        """

        # Validate the passed arguments

        # Store the passed arguments

        # Initialize the model parameters to None


    def fit(self, X_train: npt.NDArray, y_train: npt.NDArray, random_seed: int | None = 0) -> Self:
        """ Fits the model for the given training data. """

        # Reinitialize the random number generator
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize the model parameters with ...

        return self


    def get_latent(self, X_test: npt.NDArray) -> npt.NDArray:
        """ Returns the reduced dataset."""

        # Check if fit method called before predict

        return y_pred
