""" Provides the KNN class. """

from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt


class KNN:
    """ Implements K-Nearest Neighbour Classifier. """


    def __init__(self, k: int, metric: Literal['manhattan', 'euclidean', 'cosine']):
        """ Initializes the model hyperparameters.

        Args:
            k: The number of neighbours to consider for voting.
            metric: The distance metric to use.
        """

        # Validate the passed arguments
        assert k > 0, 'k should be positive'
        assert metric in ['manhattan', 'euclidean', 'cosine'], \
                                f'Unrecognized value passed for metric {metric}'

        # Store the passed arguments
        self.k = k
        self.metric = metric

        # Initialize the training data to None
        self.X_train = None
        self.y_train = None


    def fit(self, X_train: npt.NDArray, y_train: npt.NDArray) -> Self:
        """ Fits the model for the given training data. """

        # Store the training data
        self.X_train = X_train
        self.y_train = y_train

        return self


    def predict(self, X_test: npt.NDArray) -> npt.NDArray:
        """ Returns the prediction for an array of test samples."""

        # Check if fit method before predict
        assert self.X_train is not None, 'fit method should be called before predict'
        assert self.y_train is not None, 'fit method should be called before predict'

        # Collect the prediction for each test sample
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)


    def _predict(self, x) -> Any:
        """ Returns the prediction for a single test sample. """

        # Compute distances between x and all examples in the training set
        if self.metric == 'manhattan':
            diff = x - self.X_train
            distances = np.sum(np.abs(diff), axis=1)
        elif self.metric == 'euclidean':
            diff = x - self.X_train
            distances = np.linalg.norm(diff, axis=1)
        elif self.metric == 'cosine':
            X_train_norm = self.X_train / np.linalg.norm(self.X_train, axis=1).reshape(-1, 1)
            x_norm = x / np.linalg.norm(x)
            distances = 1 - np.sum(x_norm * X_train_norm, axis=1)
        else:
            raise ValueError(f'Found unrecognized value for metric {self.metric}')

        # Sort by distance and return the indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Return the most common class label
        unique_labels, label_counts = np.unique(k_nearest_labels, return_counts=True)
        return unique_labels[label_counts.argmax()]


class OldKNN:
    """ Old implementation of K Nearest Neighbour Classifier, without vectorization. """

    def __init__(self, k: int, metric: Literal['manhattan', 'euclidean', 'cosine']):
        """ Initializes the model hyperparameters.

        Args:
            k: The number of neighbours to consider for voting.
            metric: The distance metric to use.
        """

        # Validate the passed arguments
        assert k > 0, 'k should be positive'
        assert metric in ['manhattan', 'euclidean', 'cosine'], \
                                f'Unrecognized value passed for metric {metric}'

        # Store the passed arguments
        self.k = k
        self.metric = metric

        # Initialize the training data to None
        self.X_train = None
        self.y_train = None


    def fit(self, X_train: npt.NDArray, y_train: npt.NDArray) -> Self:
        """ Fits the model for the given training data. """

        # Store the training data
        self.X_train = X_train
        self.y_train = y_train

        return self


    def predict(self, X_test: npt.NDArray) -> npt.NDArray:
        """ Returns the prediction for an array of test samples."""

        # Check if fit method before predict
        assert self.X_train is not None, 'fit method should be called before predict'
        assert self.y_train is not None, 'fit method should be called before predict'

        # Collect the prediction for each test sample
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)


    def _predict(self, x) -> Any:
        """ Returns the prediction for a single test sample. """

        # Compute distances between x and all examples in the training set
        if self.metric == 'manhattan':
            distances = [ np.sum(np.abs(x - x_train)) for x_train in self.X_train ]
        elif self.metric == 'euclidean':
            distances = [ np.linalg.norm(x - x_train) for x_train in self.X_train ]
        elif self.metric == 'cosine':
            distances = [ 1 - np.sum(x * x_train) / (np.linalg.norm(x) * np.linalg.norm(x_train)) \
                                                     for x_train in self.X_train ]
        else:
            raise ValueError(f'Found unrecognized value for metric {self.metric}')

        # Sort by distance and return the indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Return the most common class label
        unique_labels, label_counts = np.unique(k_nearest_labels, return_counts=True)
        return unique_labels[label_counts.argmax()]
