""" Provides layers used in neural networks. """

from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from .activation import ActivationFunction


class Linear:
    """ Implements Linear layer, given by f(X) = X @ W + b. """


    def __init__(self, num_inputs: int, num_outputs: int):
        """ Initializes the layer parameters.

        Args:
            num_inputs: Number of input features.
            num_outputs: Number of output features.
        """

        # Validate the passed arguments
        assert num_inputs > 0, 'num_inputs should be positive'
        assert num_outputs > 0, 'num_outputs should be positive'

        # Store the passed arguments
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # Initialize the layer parameters
        self.weight = np.sqrt(2 / num_inputs) * np.random.randn(self.num_inputs, self.num_outputs)
        self.bias = np.zeros(self.num_outputs)


    def forward(self, X: npt.NDArray[np.float64], weight: npt.NDArray[np.float64] | None = None, \
                        bias: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
        """ Computes the function output for an input. """

        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias

        return (X @ weight) + bias


    def backward(self, X: npt.NDArray[np.float64], grad: npt.NDArray[np.float64]) \
                                        -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """ Computes the gradient with respect to function parameters (weight, bias) and input,
        based on the gradient with respect to function output. """

        return X.T @ grad, np.sum(grad, axis=0), grad @ self.weight.T


    def step(self, grad: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]], lr: float):
        """ Performs one update step based on the computed gradients and learning rate. """

        self.weight -= lr * grad[0]
        self.bias -= lr * grad[1]


class Sequential:
    """ Implements a sequence of linear layers, followed by activations. """


    def __init__(self, layers: List[Linear], activations: List[ActivationFunction], \
                                                        gradient_threshold: float | None = None):
        """ Initializes the sequential network.

        Args:
            layers: A list of initialized layers.
            activations: A list of activations succedding each layer.
        """

        # Validate the passed arguments
        assert len(layers) == len(activations) + 1, \
                                    'The number of activations should be one less than the layers'
        if gradient_threshold is not None:
            assert gradient_threshold > 0, 'gradient_threshold should be positive'

        # Store the passed arguments
        self.layers = layers
        self.activations = activations
        self.gradient_threshold = gradient_threshold

        # Initialize the outputs and gradients to None (needed for backward pass)
        self.outputs = None
        self.gradients = None


    def forward(self, X: npt.NDArray[np.float64], layers: List[Linear] | None = None, \
                                            index: int | None = None) -> npt.NDArray[np.float64]:
        """ Computes the model output sequentially in forward direction,
        storing the layerwise outputs for backward pass. """

        # Use the sequential layers if layers not provided
        if layers is None:
            layers = self.layers

        if index is not None:
            assert index < len(self.layers), \
                                    'index (zero indexing) should be less than number of layers'
            num_layers = index
        else:
            num_layers = len(self.layers)

        # Store layerwise outputs
        self.outputs = [ None for _ in range(num_layers + 1) ]
        self.outputs[0] = X
        for idx in range(num_layers):
            # Don't apply activation to last layer output
            if idx != num_layers - 1:
                self.outputs[idx + 1] = self.activations[idx].forward(
                    layers[idx].forward(self.outputs[idx])
                )
            else:
                self.outputs[idx + 1] = layers[idx].forward(self.outputs[idx])

        return self.outputs[-1]


    def backward(self, grad: npt.NDArray[np.float64]) -> None:
        """ Computes gradient via propagation in the backward direction,
        based on the gradient with respect to the output. """

        # Check if forward called before backward
        assert self.outputs is not None, 'forward should be called before backward'

        # Initialize gradients
        self.gradients = [ None for _ in range(len(self.layers)) ]

        # Apply chain rule for propagation to the remaining layers
        for idx in reversed(range(len(self.layers))):
            # Don't apply activation to last layer output
            if idx != len(self.layers) - 1:
                grad = self.activations[idx].backward(self.outputs[idx + 1], grad)
            # pylint: disable-next=invalid-name
            grad_W, grad_b, grad = self.layers[idx].backward(self.outputs[idx], grad)
            self.gradients[idx] = [grad_W, grad_b]

        # Clear the stored outputs
        self.outputs = None

        # Gradient clipping to avoid exploding gradients
        # pylint: disable=consider-using-enumerate
        if self.gradient_threshold is not None:
            for idx in range(len(self.gradients)):
                for jdx in range(len(self.gradients[idx])):
                    self.gradients[idx][jdx] = np.clip(self.gradients[idx][jdx], \
                                                -self.gradient_threshold, self.gradient_threshold)
        # pylint: enable=consider-using-enumerate

        return self.gradients


    def step(self, lr: float) -> None:
        """ Updates the layer parameters based on the computed gradients. """

        # Check if backward called before step
        assert self.gradients is not None, 'backward should be called before step'

        # Update each layer recursively
        # pylint: disable-next=consider-using-enumerate
        for idx in range(len(self.layers)):
            self.layers[idx].step(self.gradients[idx], lr)

        # Clear the stored gradients
        self.gradients = None
