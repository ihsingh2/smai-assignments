# pylint: disable=invalid-name

""" Provides the MLP class. """

# pylint: enable=invalid-name

import copy
import sys
from collections import deque
from typing import List, Literal, Self, Tuple

import numpy as np
import numpy.typing as npt
import wandb

#pylint: disable=wrong-import-position

PROJECT_DIR = '../..'
sys.path.insert(0, PROJECT_DIR)

from performance_measures import ClassificationMeasures, MultiLabelClassificationMeasures

#pylint: enable=wrong-import-position

from .activation import ActivationFunction, Sigmoid, Softmax
from .layers import Linear, Sequential
from .loss import LossFunction


# pylint: disable-next=too-many-instance-attributes
class MLP:
    """ Implements Multi Layer Perceptron. """


    # pylint: disable-next=too-many-arguments, too-many-positional-arguments
    def __init__(
        self, num_hidden_layers: int, num_neurons_per_layer: int | List[int],
        task: Literal['regression', 'single-label-classification', 'multi-label-classification'],
        activation: ActivationFunction, loss: LossFunction, lr: float = 1e-4, num_epochs: int = 25,
        batch_size: int = 16, optimizer: Literal['sgd', 'batch', 'mini-batch'] = 'mini-batch'
    ):
        """ Initializes the model hyperparameters.

        Args:
            num_hidden_layers: Number of hidden layers.
            num_neurons_per_layer: Number of neurons in each hidden layer.
            classify: Indicator for classification tasks (softmax will be applied automatically).
            activation: The activation function to apply on each neuron's output
                                                                    (except final layer neurons).
            loss: The activation function to apply on the final output.
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
        assert task in \
            ['regression', 'single-label-classification', 'multi-label-classification'], \
                                                        f'Got unrecognized value for task {task}'
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
        self.task = task
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer

        # Initialize the model parameters to None
        self.sequential = None
        self.outputs = None
        self.val_losses = None


    def backward(self, y: npt.NDArray) -> None:
        """ Computes gradient via propagation in the backward direction. """

        # Check if forward called before backward (with requires_grad)
        assert self.outputs is not None, 'forward should be called before backward'

        # Compute the gradient with respect to the output layer
        grad = self.loss.backward(y, self.outputs[-1])

        # Compute the gradient of the output activation
        if self.task == 'multi-label-classification':
            grad = Sigmoid().backward(self.outputs[-1], grad)

        # Apply chain rule for propagation to the remaining layers
        self.sequential.backward(grad)

        # Clear the stored outputs
        self.outputs = None


    def _binary_encoding(self, y: npt.NDArray[int]) -> npt.NDArray:
        """ Generates the binary encoding for a list of integer labels. """

        if self.task == 'single-label-classification':
            num_classes = np.max(y) + 1
            encoding = np.zeros((y.shape[0], num_classes), dtype=int)
            encoding[np.arange(y.shape[0]), y] = 1
            return encoding

        if self.task == 'multi-label-classification':
            num_classes = max(val for sublist in y for val in sublist) + 1
            encoding = np.zeros((y.shape[0], num_classes), dtype=int)
            for idx, labels in enumerate(y):
                encoding[idx, labels] = 1
            return encoding

        raise ValueError('Binary encoding available only for classification tasks')


    def _early_stopping(self, val_loss: float) -> bool:
        """ Checks the suitability for early stopping of gradient descent. """

        self.val_losses.append(val_loss)

        # If sufficient samples for loss available for comparision
        if len(self.val_losses) == self.val_losses.maxlen:

            # Compute mean of first half and second half
            midpoint = self.val_losses.maxlen // 2
            previous_loss_pattern = np.mean(list(self.val_losses)[:midpoint])
            current_loss_pattern = np.mean(list(self.val_losses)[midpoint:])

            # If second half loss is greater, stop
            if current_loss_pattern > previous_loss_pattern:
                return True

        return False


    # pylint: disable-next=too-many-arguments, too-many-positional-arguments
    def fit(
        self, X_train: npt.NDArray, y_train: npt.NDArray, X_val: npt.NDArray, y_val: npt.NDArray,
        wandb_log: bool = False, random_seed: int | None = 0
    ) -> Self:
        """ Fits the model for the given training data. """

        # Reinitialize the random number generator
        if random_seed is not None:
            np.random.seed(random_seed)

        # Ensure consistency with dataset format
        y_train_label, y_val_label = copy.deepcopy(y_train), copy.deepcopy(y_val)
        X_train, y_train = self._process_dataset(X_train, y_train)
        X_val, y_val = self._process_dataset(X_val, y_val)

        # Initialize the sequential network
        self._init_sequential(X_train.shape[1], y_train.shape[1])

        # Queue to store recent val losses
        self.val_losses = deque(maxlen=10)

        # Iterate until termination condition
        for epoch in range(self.num_epochs):

            # Train the model
            self.train(X_train, y_train)

            # Compute the validation loss and accuracy
            val_loss = self.loss.forward(y_val, self.forward(X_val))

            # Log metrics
            if wandb_log:
                train_loss = self.loss.forward(y_train, self.forward(X_train))

                if self.task == 'single-label-classification':
                    train_acc = ClassificationMeasures(y_train_label, \
                                                        self.predict(X_train) ).accuracy_score()
                    val_acc = ClassificationMeasures(y_val_label, \
                                                        self.predict(X_val) ).accuracy_score()
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'train_acc': train_acc,
                        'val_acc': val_acc
                    })

                elif self.task == 'multi-label-classification':
                    train_acc = MultiLabelClassificationMeasures(y_train_label, \
                                                        self.predict(X_train) ).accuracy_score()
                    val_acc = MultiLabelClassificationMeasures(y_val_label, \
                                                        self.predict(X_val) ).accuracy_score()
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'train_acc': train_acc,
                        'val_acc': val_acc
                    })

                else:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss
                    })

            # Check early termination condition
            if self._early_stopping(val_loss):
                break

        return self


    def forward(self, X: npt.NDArray, index: int | None = None) -> npt.NDArray:
        """ Computes the model output sequentially in forward direction,
        optionally storing the layerwise outputs for backward pass. """

        # Check if fit called before forward
        assert self.sequential is not None, 'fit should be called before forward'

        # Store the final output for computing derivative of loss
        self.outputs = self.sequential.forward(X, index=index)
        if self.task == 'single-label-classification':
            self.outputs = Softmax().forward(self.outputs)
        elif self.task == 'multi-label-classification':
            self.outputs = Sigmoid().forward(self.outputs)

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
        activations = [ self.activation for _ in range(len(layers) - 1) ]
        self.sequential = Sequential(layers, activations, gradient_threshold=25)


    def _label_encoding(self, y: npt.NDArray[int]) -> npt.NDArray:
        """ Generates the label encoding for a list of one hot encoded labels. """

        if self.task == 'single-label-classification':
            return np.argmax(y, axis=1)

        if self.task == 'multi-label-classification':
            y = np.where(y > 0.5, 1, 0).astype(int)
            return np.array([[int(i) for i in np.where(row == 1)[0]] for row in y], dtype=object)

        raise ValueError('Label encoding available only for classification tasks')


    def predict(self, X_test: npt.NDArray) -> npt.NDArray:
        """ Returns the prediction for an array of test samples."""

        # Check if fit method called before predict
        assert self.sequential is not None, 'fit should be called before predict'

        # Compute the predictions
        y_pred = self.forward(X_test)

        # Find the labels for classification
        if self.task != 'regression':
            y_pred = self._label_encoding(y_pred)

        return y_pred


    def _process_dataset(self, X: npt.NDArray, y: npt.NDArray) \
                                                                -> Tuple[npt.NDArray, npt.NDArray]:
        """ Ensure consistency of the class with the format of the dataset. """

        # One hot encoding for labels
        if self.task != 'regression':
            y = self._binary_encoding(y)

        # Match number of inputs and outputs
        assert X.shape[0] == y.shape[0], 'Number of inputs and outputs should be equal'

        # Each individual output should be an array
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        return X, y


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
