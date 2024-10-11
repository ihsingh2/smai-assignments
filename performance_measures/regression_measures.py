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


    def mean_absolute_error(self) -> float:
        """ Computes the mean absolute error. """

        return np.mean(np.abs(self.y_true - self.y_pred))


    def mean_squared_error(self) -> float:
        """ Computes the mean squared error. """

        return np.mean((self.y_true - self.y_pred) ** 2)


    def root_mean_squared_error(self) -> float:
        """ Computes the root mean squared error. """

        return np.sqrt(np.mean((self.y_true - self.y_pred) ** 2))


    def r_squared(self) -> float:
        """ Computes the coefficient of determination. """

        return 1 - (np.sum((self.y_true - self.y_pred) ** 2) / \
                                                np.sum((self.y_true - np.mean(self.y_true)) ** 2))


    def standard_deviation(self) -> float:
        """ Computes the standard deviation of predicted values. """

        return np.std(self.y_pred)


    def variance(self) -> float:
        """ Computes the variance of predicted values. """

        return np.var(self.y_pred)


    def print_all_measures(self) -> None:
        """ Evaluates and prints all the measures. """

        print('MAE:', self.mean_absolute_error())
        print('MSE:', self.mean_squared_error())
        print('RMSE:', self.root_mean_squared_error())
        print('R2:', self.r_squared())
        print('STD:', self.standard_deviation())
        print('VAR:', self.variance())
