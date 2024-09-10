""" Provides the KMeans class. """

from typing import Self

import numpy as np
import numpy.typing as npt


class KMeans:
    """ Implements K-Means Classifier. """


    def __init__(self, k: int):
        """ Initializes the model hyperparameters.

        Args:
            k: The number of clusters to form.
        """

        # Validate the passed arguments
        assert k > 0, 'k should be positive'

        # Store the passed arguments
        self.k = k

        # Initialize the training data and cluster centroids to None
        self.X_train = None
        self.centroids = None


    def fit(self, X_train: npt.NDArray, eps: float = 1e-6, random_seed: int | None = 0) -> Self:
        """ Fits the model for the given training data. """

        # Reinitialize the random number generator
        if random_seed is not None:
            np.random.seed(random_seed)

        # Store the training data
        self.X_train = X_train

        # Randomly initialize the cluster centroids
        indices = np.random.choice(self.X_train.shape[0], size=self.k, replace=False)
        self.centroids = X_train[indices]

        # Initialize list to store costs
        costs = [ self.getCost(), ]
        delta_cost = np.inf

        # Loop until termination criteria is satisifed
        while delta_cost > eps:

            # Compute the distances from training points to cluster centres
            diff = self.X_train - self.centroids[:, np.newaxis]
            distances = np.sqrt((diff**2).sum(axis=2))

            # Assign each training point to the nearest cluster
            clusters = np.argmin(distances, axis=0)

            # Recompute the cluster centres as the mean of the assigned points
            self.centroids = np.array(
                [X_train[clusters == i].mean(axis=0) for i in range(self.k)]
            )

            # Recompute the costs and change in cost this iteration
            costs.append(self.getCost())
            delta_cost = costs[-2] - costs[-1]

        return self


    def predict(self, X_test: npt.NDArray) -> npt.NDArray:
        """ Returns the prediction for an array of test samples."""

        # Check if fit method called before predict
        assert self.X_train is not None, 'fit method should be called before predict'
        assert self.centroids is not None, 'fit method should be called before predict'

        # Compute the distances from test points to cluster centres
        diff = X_test - self.centroids[:, np.newaxis]
        distances = np.sqrt((diff**2).sum(axis=2))

        # Assign each training point to the nearest cluster
        clusters = np.argmin(distances, axis=0)

        return clusters


    # pylint: disable-next=invalid-name
    def getCost(self):
        """ Returns the Within Cluster Sum of Squares (WCSS). """

        # Check if fit method called before getCost
        assert self.X_train is not None, 'fit method should be called before getCost'
        assert self.centroids is not None, 'fit method should be called before getCost'

        # Compute the squared distances from training points to cluster centres
        diff = self.X_train - self.centroids[:, np.newaxis]
        sq_distances = (diff**2).sum(axis=2)

        # Filter out the squared distance to the nearest cluster
        cluster_sq_distances = np.min(sq_distances, axis=0)

        # Compute sum of all the cluster distances
        cost = np.sum(cluster_sq_distances)

        return cost
