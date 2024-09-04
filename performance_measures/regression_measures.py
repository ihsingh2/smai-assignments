""" Provides RegressionMeasures. """

import numpy as np
import numpy.typing as npt


class RegressionMeasures:
    """ Computes common evaluation measures for regression based tasks. """

    def __init__(self, y_true: npt.NDArray, y_pred: npt.NDArray):
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
