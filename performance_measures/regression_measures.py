import numpy as np
from numpy.typing import NDArray


class RegressionMeasures:

    def __init__(self, y_true: NDArray, y_pred: NDArray):
        """ Initializes the class to compute on given data.

        Args:
            y_true: Array containing true values.
            y_pred: Array containing predicted values.
        """

        self.y_true = y_true
        self.y_pred = y_pred

    def mean_squared_error(self) -> float:
        """ Computes the mean squared error. """

        return np.mean((self.y_true - self.y_pred) ** 2)

    def standard_deviation(self) -> float:
        """ Computes the standard deviation of predicted values. """

        return np.std(self.y_pred)

    def variance(self) -> float:
        """ Computes the variance of predicted values. """

        return np.var(self.y_pred)
