""" Provides the PCA class. """

from typing import Self

import numpy as np
import numpy.typing as npt


class PCA:
    """ Implements Principal Component Analysis. """


    def __init__(self, n_components: int):
        """ Initializes the model hyperparameters.

        Args:
            n_components: Number of dimensions the data should be reduced to.
        """

        # Validate the passed arguments
        assert n_components > 0, 'n_components should be positive'

        # Store the passed arguments
        self.n_components = n_components

        # Initialize the training data and model parameters to None
        self.X_train = None
        self.components = None


    def fit(self, X_train: npt.NDArray) -> Self:
        """ Obtains the principal component of the dataset. """

        # Check if the input is compatible with the model hyperparameters
        assert X_train.shape[1] > self.n_components, \
                                    'Input dimensionality should be greater than n_components'

        # Store the training data
        self.X_train = X_train

        # Compute the covariance matrix
        X_centered = self.X_train - np.mean(self.X_train, axis=0)
        covariance = X_centered.T @ X_centered / (self.X_train.shape[0] - 1)

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance)

        # Discard the imaginary part in case it arises due to numerical issues
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        # Sort the eigenvalues and eigenvectors in non-increasing order
        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[indices]

        # Extract the top eigenvectors
        self.components = eigenvectors[:self.n_components]

        return self


    def transform(self, X: npt.NDArray) -> npt.NDArray:
        """ Transforms the data using the principal components."""

        # Check if fit method called before transform
        assert self.X_train is not None, 'transform called before fit'
        assert self.components is not None, 'transform called before fit'

        # Check if the input is compatible with the training data
        assert X.shape[1] == self.components.shape[1], \
                                        'Input dimensionality differs from the training data'

        # Project the input array to the reduced eigenspace
        X_transformed = X @ self.components.T

        return X_transformed


    # pylint: disable-next=invalid-name
    def checkPCA(self) -> None:
        """ Checks whether the dimensions were reduced appropriately. """

        # Check if fit method called before checkPCA
        assert self.X_train is not None, 'checkPCA called before fit'
        assert self.components is not None, 'checkPCA called before fit'

        # Shape of the training data
        num_original_dimensions = self.X_train.shape[1]

        # Test cases -- the number of test samples in each test case
        test_cases = [50, 100, 150]

        for num_test_samples in test_cases:
            data = np.random.randn(num_test_samples, num_original_dimensions)
            data_transformed = self.transform(data)
            assert data_transformed.shape[0] == num_test_samples, \
                                        'Received lesser samples than expected upon transform'
            assert data_transformed.shape[1] == self.n_components, \
                                        'Received wrong number of components upon transform'
