""" Provides loss functions commonly used in neural networks. """

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class LossFunction(ABC):
    """ Abstract class providing signature for loss functions,
    tailored for use in forward and backward propagation. """

    @abstractmethod
    def forward(self, Y: npt.NDArray[np.float64], Y_pred: npt.NDArray[np.float64]) \
                                                                        -> npt.NDArray[np.float64]:
        """ Returns the function output for a given set of true input and predicted input. """

    @abstractmethod
    def backward(self, Y: npt.NDArray[np.float64], Y_pred: npt.NDArray[np.float64]) \
                                                                        -> npt.NDArray[np.float64]:
        """ Returns the function gradient for a given set of true input and predicted input. """


class CrossEntropy(LossFunction):
    """ Implements Cross Entropy function, given by f(Y, Y_pred) = sum [ Y_i log Y_pred_i ].
    This class is meant to be used in conjunction with Softmax for backward pass. """

    def forward(self, Y: npt.NDArray[np.float64], Y_pred: npt.NDArray[np.float64]) \
                                                                        -> npt.NDArray[np.float64]:
        Y_pred = np.clip(Y_pred, 1e-9, 1 - 1e-9)
        return np.mean(-np.sum(Y * np.log(Y_pred), axis=1))

    def backward(self, Y: npt.NDArray[np.float64], Y_pred: npt.NDArray[np.float64]) \
                                                                        -> npt.NDArray[np.float64]:
        Y_pred = np.clip(Y_pred, 1e-9, 1 - 1e-9)
        return (Y_pred - Y) / Y.shape[0]


class MeanSquaredError(LossFunction):
    """ Implements Mean Squared Error, given by f(Y, Y_pred) = sum ((Y_i - Y_pred_i) ** 2). """

    def forward(self, Y: npt.NDArray[np.float64], Y_pred: npt.NDArray[np.float64]) \
                                                                        -> npt.NDArray[np.float64]:
        return np.mean((Y - Y_pred) ** 2)

    def backward(self, Y: npt.NDArray[np.float64], Y_pred: npt.NDArray[np.float64]) \
                                                                        -> npt.NDArray[np.float64]:
        return -2 * (Y - Y_pred) / (Y.shape[0] * Y.shape[1])