# pylint: disable=invalid-name

""" Provides the AutoEncoder class. """

# pylint: enable=invalid-name


import sys
from typing import Literal, Self

import numpy.typing as npt

#pylint: disable=wrong-import-position

PROJECT_DIR = '../..'
sys.path.insert(0, PROJECT_DIR)

from models.MLP.activation import ActivationFunction
from models.MLP.loss import LossFunction, MeanSquaredError
from models.MLP import MLP

#pylint: enable=wrong-import-position


# pylint: disable-next=too-many-instance-attributes
class AutoEncoder:
    """ Implements AutoEncoder. """


    # pylint: disable-next=too-many-arguments, too-many-positional-arguments
    def __init__(
        self, latent_dimension: int, num_hidden_layers: int, activation: ActivationFunction,
        loss: LossFunction = MeanSquaredError(), lr: float = 1e-4, num_epochs: int = 25,
        batch_size: int = 16, optimizer: Literal['sgd', 'batch', 'mini-batch'] = 'mini-batch'
    ):
        """ Initializes the model hyperparameters.

        Args:
            latent_dimension: The dimension to reduce the data to.
            num_hidden_layers: Number of hidden layers.
            activation: The activation function to apply on each neuron's output
                                                                    (except final layer neurons).
            loss: The activation function to apply on the final output.
            lr: The learning rate to use in update step.
            num_epochs: Number of iterations over the training dataset.
            batch_size: Number of samples from the training dataset to process in a batch.
            optimizer: The optimization technique to use for updating weights.
        """

        # Validate the passed arguments
        assert latent_dimension > 0, 'latent_dimension should be positive'
        assert num_hidden_layers >= 0, 'num_hidden_layers should be non-negative'
        assert lr > 0, 'lr should be positive'
        assert num_epochs > 0, 'num_epochs should be positive'
        assert batch_size > 0, 'batch_size should be positive'
        assert optimizer in ['sgd', 'batch', 'mini-batch'], \
                                    f'Received unrecognized input {optimizer} for optimizer'

        # Store the passed arguments
        self.latent_dimension = latent_dimension
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation
        self.loss = loss
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer

        # Initialize the model parameters to None
        self.mlp = None
        self.input_dimension = None


    def fit(self, X_train: npt.NDArray, X_val: npt.NDArray, wandb_log: bool = False) -> Self:
        """ Fits the model for the given training data. """

        # Store the dimensions
        assert X_train.shape[1] == X_val.shape[1], \
                                        'The training and validation sets have inconsistent shape'
        self.input_dimension = X_train.shape[1]

        # Initialize the model
        self.mlp = MLP(
            num_hidden_layers=(self.num_hidden_layers * 2 + 1),
            num_neurons_per_layer=self._num_neurons_per_layer(),
            task='regression',
            activation=self.activation,
            loss=self.loss,
            lr=self.lr,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            optimizer=self.optimizer
        )

        # Fit the model
        self.mlp.fit(X_train, X_train, X_val, X_val, wandb_log=wandb_log)

        return self


    def get_latent(self, X_test: npt.NDArray) -> npt.NDArray:
        """ Returns the reduced dataset."""

        # Check if fit method called before predict
        assert self.mlp is not None, 'fit should be called before get_latent'

        # Compute the latent representation
        latent = self.mlp.forward(X_test, index=self.num_hidden_layers + 1)
        assert latent.shape[1] == self.latent_dimension

        return latent


    def _num_neurons_per_layer(self):
        """ Computes the number of neurons per layer for gradual reduction in dimensionality. """

        num_neurons_per_layer = [ None for _ in range(self.num_hidden_layers * 2 + 1) ]
        num_neurons_per_layer[self.num_hidden_layers] = self.latent_dimension

        slope = (self.latent_dimension - self.input_dimension) / (self.num_hidden_layers + 1)
        for idx in range(self.num_hidden_layers):
            num_neurons_per_layer[idx] = num_neurons_per_layer[- idx - 1] \
                                                = round(slope * (idx + 1) + self.input_dimension)

        return num_neurons_per_layer
