# pylint: disable=invalid-name

""" Provides the MLP class. """

# pylint: enable=invalid-name


from typing import List, Literal, Self, Tuple

import numpy as np
import numpy.typing as npt

from .activation import ActivationFunction, Softmax
from .layers import Linear, Sequential
from .loss import LossFunction


# pylint: disable-next=too-many-instance-attributes
class MLP:
    """ Implements Multi Layer Perceptron. """


    # pylint: disable-next=too-many-arguments, too-many-positional-arguments
    def __init__(
        self, num_hidden_layers: int, num_neurons_per_layer: int | List[int], classify: bool,
        activation: ActivationFunction, loss: LossFunction, lr: float = 1e-4, num_epochs: int = 15,
        batch_size: int = 16, optimizer: Literal['sgd', 'batch', 'mini-batch'] = 'mini-batch'
    ):
        """ Initializes the model hyperparameters.

        Args:
            num_hidden_layers: Number of hidden layers.
            num_neurons_per_layer: Number of neurons in each hidden layer.
            activation: The activation function to apply on each neuron's output.
            lr: The learning rate to use in update step.
            num_epochs: Number of iterations over the training dataset.
            batch_size: Number of samples from the training dataset to process in a batch.
            optimizer: The optimization technique to use for updating weights.
        """

        # Validate the passed arguments
        assert num_hidden_layers >= 0, 'num_hidden_layers should be non-negative'
        if isinstance(num_neurons_per_layer, int):
            assert num_neurons_per_layer > 0, 'num_neurons_per_layer should be positive'
        else:
            assert len(num_neurons_per_layer) == num_hidden_layers, \
                        f'Expected list of {num_hidden_layers} values for num_neurons_per_layer' \
                        f', got {len(num_neurons_per_layer)}'
            for idx, num in enumerate(num_neurons_per_layer):
                assert num > 0, \
                        f'num_neurons_per_layer should be positive, got {num} at index {idx}'
        assert lr > 0, 'lr should be positive'
        assert num_epochs > 0, 'num_epochs should be positive'
        assert batch_size > 0, 'batch_size should be positive'
        assert optimizer in ['sgd', 'batch', 'mini-batch'], \
                                    f'Received unrecognized input {optimizer} for optimizer'

        # Store the passed arguments
        self.num_hidden_layers = num_hidden_layers
        if isinstance(num_neurons_per_layer, int):
            self.num_neurons_per_layer = [num_neurons_per_layer for _ in range(num_hidden_layers)]
        else:
            self.num_neurons_per_layer = num_neurons_per_layer
        self.activation = activation
        self.loss = loss
        self.classify = classify
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer

        # Initialize the model to None
        self.sequential = None
        self.outputs = None


    def backward(self, y: npt.NDArray) -> None:
        """ Computes gradient via propagation in the backward direction. """

        # Check if forward called before backward (with requires_grad)
        assert self.outputs is not None, 'forward should be called before backward'

        # Compute the gradient with respect to the output layer
        grad = self.loss.backward(y, self.outputs[-1])

        # Apply chain rule for propagation to the remaining layers
        self.sequential.backward(grad)

        # Clear the stored outputs
        self.outputs = None


    def _early_stopping(self) -> bool:
        """ Checks the suitability for early stopping of gradient descent. """


    def fit(self, X_train: npt.NDArray, y_train: npt.NDArray, random_seed: int | None = 0) -> Self:
        """ Fits the model for the given training data. """

        # Reinitialize the random number generator
        if random_seed is not None:
            np.random.seed(random_seed)

        # Ensure consistency with dataset format
        X_train, y_train = self._process_training_dataset(X_train, y_train)

        # Initialize the sequential network
        self._init_sequential(X_train.shape[1], y_train.shape[1])

        # Train the model
        for _ in range(self.num_epochs):
            self.train(X_train, y_train)
            y_pred = self.forward(X_train)
            print(self.loss.forward(y_train, y_pred))

        return self


    def forward(self, X: npt.NDArray) -> npt.NDArray:
        """ Computes the model output sequentially in forward direction,
        optionally storing the layerwise outputs for backward pass. """

        # Check if fit called before forward
        assert self.sequential is not None, 'fit should be called before forward'

        # Store the final output for computing loss derivative
        self.outputs = self.sequential.forward(X)
        if self.classify:
            self.outputs = Softmax().forward(self.outputs)

        return self.outputs


    def _init_sequential(self, input_dim: int, output_dim: int) -> None:
        """ Initializes the sequential network based on the dimensions of training dataset. """

        layers = []
        if self.num_hidden_layers == 0:
            layers.append(Linear(input_dim, output_dim))
        else:
            layers.append(Linear(input_dim, self.num_neurons_per_layer[0]))
            for idx in range(self.num_hidden_layers - 1):
                layers.append( \
                    Linear(self.num_neurons_per_layer[idx], self.num_neurons_per_layer[idx + 1]) \
                )
            layers.append(Linear(self.num_neurons_per_layer[-1], output_dim))
        activations = [ self.activation for _ in range(len(layers)) ]
        self.sequential = Sequential(layers, activations)


    def _one_hot_encoding(self, X: npt.NDArray[int]) -> npt.NDArray:
        """ Generates the one hot encoding for a list of integer labels. """

        num_classes = np.max(X) + 1
        encoding = np.zeros((X.shape[0], num_classes))
        encoding[np.arange(X.shape[0]), X] = 1
        return encoding


    def predict(self, X_test: npt.NDArray) -> npt.NDArray:
        """ Returns the prediction for an array of test samples."""

        # Check if fit method called before predict
        assert self.sequential is not None, 'fit should be called before predict'

        # Compute the predictions
        y_pred = self.forward(X_test)

        # Find the largest logit for classification
        if self.classify:
            y_pred = np.argmax(y_pred, axis=1)

        return y_pred


    def _process_training_dataset(self, X_train: npt.NDArray, y_train: npt.NDArray) \
                                                                -> Tuple[npt.NDArray, npt.NDArray]:
        """ Ensure consistency of the class with the format of the dataset. """

        # One hot encoding for labels
        if self.classify:
            y_train = self._one_hot_encoding(y_train)

        # Match number of inputs and outputs
        assert X_train.shape[0] == y_train.shape[0]

        # Each individual output should be an array
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        return X_train, y_train


    def train(self, X_train: npt.NDArray, y_train: npt.NDArray) -> None:
        """ Trains the model for one epoch over the given training dataset. """

        # Shape of the dataset
        num_points = X_train.shape[0]

        # Stochastic Gradient Descent
        if self.optimizer == 'sgd':
            for idx in range(num_points):
                self.training_loop(X_train[[idx]], y_train[[idx]])

        # Batch Gradient Descent
        elif self.optimizer == 'batch':
            self.training_loop(X_train, y_train)

        # Mini-Batch Gradient Descent
        elif self.optimizer == 'mini-batch':
            num_batches = num_points // self.batch_size
            num_batch_points = num_batches * self.batch_size
            for idx in range(0, num_batch_points, self.batch_size):
                self.training_loop(X_train[idx: idx+self.batch_size], \
                                                        y_train[idx: idx+self.batch_size])
            if num_batch_points < num_points:
                self.training_loop(X_train[num_batch_points: -1], y_train[num_batch_points: -1])


    def training_loop(self, X_train: npt.NDArray, y_train: npt.NDArray) -> None:
        """ Performs one pass over the training data, for parameter update. """

        self.forward(X_train)
        self.backward(y_train)
        self.step()


    def step(self) -> None:
        """ Updates the parameters based on computed gradients. """

        # Check if fit method called before step
        assert self.sequential is not None, 'fit should be called before step'

        self.sequential.step(self.lr)
