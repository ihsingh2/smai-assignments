""" Provides the GMM class. """

from typing import Self, Tuple

import numpy as np
import numpy.typing as npt
from scipy.stats import multivariate_normal


class GMM:
    """ Implements Gaussian Mixture Model. """


    def __init__(self, k: int):
        """ Initializes the model hyperparameters.

        Args:
            k: The number of mixture components.
        """

        # Validate the passed arguments
        assert k > 0, 'k should be positive'

        # Store the passed arguments
        self.k = k

        # Initialize the training data and model parameters to None
        self.X_train = None
        self.mixing_coeffs = None
        self.means = None
        self.covariances = None


    def fit(self, X_train: npt.NDArray, eps: float = 1e-6, random_seed: int | None = 0) -> Self:
        """ Fits the model for the given training data. """

        # Reinitialize the random number generator
        if random_seed is not None:
            np.random.seed(random_seed)

        # Store the training data
        self.X_train = X_train

        # Initialize model parameters
        self._init_parameters(X_train)

        # Initialize list to store likelihoods
        likelihoods = [ -np.inf, ]
        delta_likelihood = np.inf

        # Loop until termination criteria is satisifed
        while delta_likelihood > eps:

            # E Step
            gamma = self._evaluate_responsibilities(self.X_train)

            # M Step
            self._estimate_parameters(gamma, self.X_train)

            # Check for convergence
            likelihoods.append(self.getLikelihood())
            delta_likelihood = likelihoods[-1] - likelihoods[-2]

        return self


    def _init_parameters(self, X: npt.NDArray) -> None:
        """ Initializes the model parameters based on random assignment
        of training samples to the components. """

        # Shape of the dataset
        num_samples = X.shape[0]
        num_features = X.shape[1]

        # Create arrays to store parameters
        self.mixing_coeffs = np.empty((self.k))
        self.means = np.empty((self.k, num_features))
        self.covariances = np.empty((self.k, num_features, num_features))

        # Random initialization of responsibilties
        resp = np.zeros((num_samples, self.k))
        resp[np.arange(num_samples), np.random.choice(self.k, num_samples)] = 1

        # Iterate over all components
        for component in range(self.k):
            in_component = np.isclose(resp[:, component], 1)
            in_component_count = int(np.sum(in_component))
            self.mixing_coeffs[component] = in_component_count / num_samples
            if in_component_count > 1:
                self.means[component] = np.mean(X[in_component], axis=0)
                X_centered = X[in_component] - self.means[component]
                self.covariances[component] = (X_centered.T @ X_centered) / in_component_count
            elif in_component_count == 1:
                self.means[component] = np.mean(X[in_component], axis=0)
                self.covariances[component] = np.eye(num_features)
            else:
                self.means[component] = np.zeros(num_features)
                self.covariances[component] = np.eye(num_features)
            # assert np.isclose(self.covariances[component], self.covariances[component].T).all()
            # assert np.linalg.cholesky(self.covariances[component]) is not None


    def _evaluate_responsibilities(self, X: npt.NDArray) -> npt.NDArray:
        """ Evaluates the responsibilites given current parameters. """

        # Shape of the dataset
        num_samples = X.shape[0]

        # Compute the responsibilities
        gamma = np.empty((num_samples, self.k))
        for component in range(self.k):
            gamma[:, component] = multivariate_normal.pdf(X, mean=self.means[component], \
                                        cov=self.covariances[component], allow_singular=True)
        gamma = gamma * self.mixing_coeffs
        gamma /= np.sum(gamma, axis=1, keepdims=True)

        return gamma


    def _estimate_parameters(self, resp: npt.NDArray, X: npt.NDArray) -> None:
        """ Re-estimates the parameters given current responsibilities. """

        # Shape of the dataset
        num_samples = X.shape[0]
        num_features = X.shape[1]

        # Compute the mixing coefficients and mean
        total_resp = np.sum(resp, axis=0)
        self.mixing_coeffs = total_resp / num_samples
        self.means = np.dot(resp.T, X) / (total_resp[:, np.newaxis] + 1e-6)

        # Compute the covariances
        self.covariances = np.empty((self.k, num_features, num_features))
        for component in range(self.k):
            X_centered = X - self.means[component]
            self.covariances[component] = ((resp[:, component, np.newaxis] * \
                                    X_centered).T @ X_centered) / (total_resp[component] + 1e-6)


    # pylint: disable-next=invalid-name
    def getParams(self) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """ Returns the parameters of the Gaussian components in the mixture model. """

        return (
            self.mixing_coeffs,
            self.means,
            self.covariances
        )


    # pylint: disable-next=invalid-name
    def getMembership(self, X: npt.NDArray = None) -> npt.NDArray:
        """ Returns the membership values for each sample in the dataset. """

        # Check if fit method called before getMembership
        assert self.X_train is not None, 'getMembership called before fit'
        assert self.mixing_coeffs is not None, 'getMembership called before fit'
        assert self.means is not None, 'getMembership called before fit'
        assert self.covariances is not None, 'getMembership called before fit'

        # Use the entire training dataset
        if X is None:
            X = self.X_train

        # Find the component with the highest responsibility
        resp = self._evaluate_responsibilities(X)
        membership = np.argmax(resp, axis=1)

        return membership


    # pylint: disable-next=invalid-name
    def getLikelihood(self, X: npt.NDArray = None) -> float:
        """ Returns the overall likelihood of the dataset under the model parameters. """

        # Check if fit method called before getLikelihood
        assert self.X_train is not None, 'getLikelihood called before fit'
        assert self.mixing_coeffs is not None, 'getLikelihood called before fit'
        assert self.means is not None, 'getLikelihood called before fit'
        assert self.covariances is not None, 'getLikelihood called before fit'

        # Use the entire training dataset
        if X is None:
            X = self.X_train

        # Shape of the dataset
        num_samples = X.shape[0]

        # Compute the log likelihood
        log_likelihood = np.empty((num_samples, self.k))
        for component in range(self.k):
            log_likelihood[:, component] = multivariate_normal.pdf(X, mean=self.means[component], \
                                            cov=self.covariances[component], allow_singular=True)
        log_likelihood = log_likelihood * self.mixing_coeffs
        log_likelihood = np.sum(np.log(np.sum(log_likelihood, axis=1)))

        return log_likelihood
