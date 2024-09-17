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
        self.means = None
        self.components = None


    def fit(self, X_train: npt.NDArray) -> Self:
        """ Obtains the principal component of the dataset. """

        # Check if the input is compatible with the model hyperparameters
        assert X_train.shape[1] >= self.n_components, \
                                    'Number of input features should be atleast n_components'

        # Store the training data
        self.X_train = X_train

        # Compute the covariance matrix
        self.means = np.mean(self.X_train, axis=0)
        covariance = np.cov(self.X_train, rowvar=False)

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # Discard the imaginary part in case it arises due to numerical issues
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        # Sort the eigenvalues and eigenvectors in non-increasing order
        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]

        # Store all the components -- useful for checking functionality
        self.components = eigenvectors

        return self


    def transform(self, X: npt.NDArray) -> npt.NDArray:
        """ Transforms the data using the n_components."""

        # Check if fit method called before transform
        assert self.X_train is not None, 'transform called before fit'
        assert self.means is not None, 'transform called before fit'
        assert self.components is not None, 'transform called before fit'

        # Check if the input is compatible with the training data
        assert X.shape[1] == self.X_train.shape[1], \
                                    'Input dimensionality differs from the training data'

        return self._transform(X, self.n_components)


    def inverse_transform(self, X_transformed: npt.NDArray) -> npt.NDArray:
        """ Reconstructs the data earlier projected to the n_components."""

        # Check if fit method called before inverse_transform
        assert self.X_train is not None, 'inverse_transform called before fit'
        assert self.means is not None, 'inverse_transform called before fit'
        assert self.components is not None, 'inverse_transform called before fit'

        # Check if the input is compatible with the training data
        assert X_transformed.shape[1] == self.n_components, \
                                    'Input dimensionality differs from the number of components'

        return self._inverse_transform(X_transformed, self.n_components)


    def _transform(self, X: npt.NDArray, n_components: int) -> npt.NDArray:
        """ Transforms the data using a subset of principal components."""

        return (X - self.means) @ self.components[:, :n_components]


    def _inverse_transform(self, X_transformed: npt.NDArray, n_components: int) -> npt.NDArray:
        """ Reconstructs the data earlier projected to a subset of principal components."""

        return (X_transformed @ self.components[:, :n_components].T) + self.means


    # pylint: disable-next=invalid-name
    def checkPCA(self, eps=1e-6) -> bool:
        """ Checks whether the dimensions were reduced appropriately. """

        # Check if fit method called before checkPCA
        assert self.X_train is not None, 'checkPCA called before fit'
        assert self.means is not None, 'checkPCA called before fit'
        assert self.components is not None, 'checkPCA called before fit'

        # Shape of the training data
        num_original_dimensions = self.X_train.shape[1]

        # Variable to keep track of test pass status
        pass_status = True

        # Iterate over test cases -- different number of data points
        test_cases = [50, 100, 150]
        for num_test_samples in test_cases:

            # Transform and reconstruct random data
            data = np.random.randn(num_test_samples, num_original_dimensions)
            data_transformed = self._transform(data, num_original_dimensions)
            data_reconstructed = self._inverse_transform(data_transformed, num_original_dimensions)

            # Test dimensionality
            pass_status &= data_transformed.shape[0] == num_test_samples
            pass_status &= data_transformed.shape[1] == num_original_dimensions

            # Test reconstruction error
            error = np.mean((data - data_reconstructed) ** 2)
            pass_status &= (error <= eps)

        return pass_status
