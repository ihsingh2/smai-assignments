""" Provides activation functions commonly used in neural networks. """

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class ActivationFunction(ABC):
    """ Abstract class providing signature for activation functions,
    tailored for use in forward and backward propagation. """

    @abstractmethod
    def forward(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """ Computes the function output for an input. """

    @abstractmethod
    def backward(self, Y: npt.NDArray[np.float64], grad: npt.NDArray[np.float64]) \
                                                                        -> npt.NDArray[np.float64]:
        """ Computes the gradient with respect to the input, based on the gradient
        with respect to function output.

        The function takes as input the function output to avoid repetitive computations.
        """


class Identity(ActivationFunction):
    """ Implements Linear activation function, given by f(X) = X. """

    def forward(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.copy(X)

    def backward(self, Y: npt.NDArray[np.float64], grad: npt.NDArray[np.float64]) \
                                                                        -> npt.NDArray[np.float64]:
        return np.ones(Y.shape) * grad


class ReLU(ActivationFunction):
    """ Implements Rectified Linear Unit (ReLU), given by f(X) = max(0, X). """

    def forward(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.maximum(0, X)

    def backward(self, Y: npt.NDArray[np.float64], grad: npt.NDArray[np.float64]) \
                                                                        -> npt.NDArray[np.float64]:
        return np.where(Y > 0, 1, 0) * grad


class Sigmoid(ActivationFunction):
    """ Implements Logistic function, given by f(X) = 1 / (1 + e^(-X)). """

    def forward(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        X = np.clip(X, -100, 100)
        return 1 / (1 + np.exp(-X))

    def backward(self, Y: npt.NDArray[np.float64], grad: npt.NDArray[np.float64]) \
                                                                        -> npt.NDArray[np.float64]:
        return (Y * (1 - Y)) * grad


class Softmax(ActivationFunction):
    """ Implements Softmax function, given by f(X) = [ (e^(X_i)) / (sum_j e^(X_j)) ]. 
    This class is meant to be used in conjunction with CrossEntropy for backward pass. """

    def forward(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        X = X - np.max(X, axis=1, keepdims=True)
        return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)

    def backward(self, Y: npt.NDArray[np.float64], grad: npt.NDArray[np.float64]) \
                                                                        -> npt.NDArray[np.float64]:
        return np.ones(Y.shape) * grad


class Tanh(ActivationFunction):
    """ Implements Hyperbolic Tangent, given by f(X) = (e^(X) - e^(-X) / (e^(X) + e^(-X)). """

    def forward(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.tanh(X)

    def backward(self, Y: npt.NDArray[np.float64], grad: npt.NDArray[np.float64]) \
                                                                        -> npt.NDArray[np.float64]:
        return (1 - Y**2) * grad
