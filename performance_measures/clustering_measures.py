""" Provides ClusteringMeasures. """

import math


class ClusteringMeasures:
    """ Computes common evaluation measures for clustering based tasks. """

    def __init__(self, k: int, n: int, log_likelihood: float):
        """ Initializes the class to compute on given data.

        Args:
            k: Number of clusters.
            n: Number of points.
            log_likelihood: Log likelihood of given data.
        """

        # Store the passed arguments
        self.k = k
        self.n = n
        self.log_likelihood = log_likelihood


    def aic(self) -> float:
        """ Computes the Akaike Information Criterion. """

        return (self.k * math.log(self.n)) - (2 * self.log_likelihood)


    def bic(self) -> float:
        """ Compute the Bayesian Information Criterion. """

        return (2 * self.k) - (2 * self.log_likelihood)
