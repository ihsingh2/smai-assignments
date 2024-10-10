""" Verification of gradients of functions against numerical gradients. """

import copy

import numpy as np

from .activation import ActivationFunction, Identity, ReLU, Sigmoid, Softmax, Tanh
from .layers import Linear, Sequential
from .loss import LossFunction, CrossEntropy, MeanSquaredError


EPSILON = 1e-8
THRESHOLD = 1e-4

NUM_BATCHES = 5
NUM_SAMPLES = 10

NUM_INPUT_DIMENSIONS = 15
NUM_OUTPUT_DIMENSIONS = 20

NUM_HIDDEN_DIMENSIONS = 10
NUM_HIDDEN_LAYERS = 5


def _test_activation(activation: ActivationFunction):
    """ Gradient check for any activation function with input and output dimension as one. """

    for _ in range(NUM_BATCHES):

        X = np.random.randn(NUM_SAMPLES, NUM_INPUT_DIMENSIONS)

        base_grad = np.random.randn(NUM_SAMPLES, NUM_INPUT_DIMENSIONS)
        grad = activation.backward(activation.forward(X), base_grad)
        assert grad.shape == X.shape

        numerical_grad = base_grad * (activation.forward(X + EPSILON) - activation.forward(X)) \
                                                                                        / EPSILON
        assert np.isclose(grad, numerical_grad).all()


def _test_loss(loss: LossFunction, activation: ActivationFunction = Identity(), \
                                                                            one_hot: bool = False):
    """ Gradient check for any loss function, optionally coupled with a prior activation. """

    for _ in range(NUM_BATCHES):

        if one_hot:
            Y = np.zeros((NUM_SAMPLES, NUM_INPUT_DIMENSIONS), dtype=int)
            indices = np.random.choice(NUM_INPUT_DIMENSIONS, size=NUM_SAMPLES)
            Y[np.arange(NUM_SAMPLES), indices] = 1
        else:
            Y = np.random.randn(NUM_SAMPLES, NUM_INPUT_DIMENSIONS)

        Y_pred = np.random.randn(NUM_SAMPLES, NUM_INPUT_DIMENSIONS)

        grad = loss.backward(Y, activation.forward(Y_pred))
        assert grad.shape == Y_pred.shape

        for idx in range(NUM_SAMPLES):
            for jdx in range(NUM_INPUT_DIMENSIONS):

                Y_pred_aug = np.copy(Y_pred)
                Y_pred_aug[idx, jdx] += EPSILON
                numerical_grad = ( \
                    loss.forward(Y, activation.forward(Y_pred_aug)) - \
                    loss.forward(Y, activation.forward(Y_pred)) \
                ) / EPSILON
                assert np.isclose(grad[idx, jdx], numerical_grad, atol=THRESHOLD)


# --- layers


def test_linear():
    """ Gradient check for Linear layer. """

    layer = Linear(NUM_INPUT_DIMENSIONS, NUM_OUTPUT_DIMENSIONS)

    for _ in range(NUM_BATCHES):

        X = np.random.randn(NUM_SAMPLES, NUM_INPUT_DIMENSIONS)

        y = layer.forward(X)
        assert y.shape == (NUM_SAMPLES, NUM_OUTPUT_DIMENSIONS)

        base_grad = np.random.randn(NUM_SAMPLES, NUM_OUTPUT_DIMENSIONS)
        # pylint: disable-next=invalid-name
        grad_W, grad_b, grad_X = layer.backward(X, base_grad)
        assert grad_W.shape == layer.weight.shape
        assert grad_b.shape == layer.bias.shape
        assert grad_X.shape == X.shape

        weight_aug = np.copy(layer.weight)
        bias_aug = np.copy(layer.bias)
        X_aug = np.copy(X)

        for jdx in range(NUM_INPUT_DIMENSIONS):
            for kdx in range(NUM_OUTPUT_DIMENSIONS):
                weight_aug[jdx, kdx] += EPSILON
                numerical_grad = np.sum( base_grad * ( \
                    layer.forward(X, weight=weight_aug) - layer.forward(X) \
                )) / EPSILON
                weight_aug[jdx, kdx] -= EPSILON
                assert np.isclose(grad_W[jdx, kdx], np.sum(numerical_grad), atol=THRESHOLD)

        for jdx in range(NUM_OUTPUT_DIMENSIONS):
            bias_aug[jdx] += EPSILON
            numerical_grad = np.sum( base_grad * ( \
                layer.forward(X, bias=bias_aug) - layer.forward(X) \
            )) / EPSILON
            bias_aug[jdx] -= EPSILON
            assert np.isclose(grad_b[jdx], np.sum(numerical_grad), atol=THRESHOLD)

        for jdx in range(NUM_SAMPLES):
            for kdx in range(NUM_INPUT_DIMENSIONS):
                X_aug[jdx, kdx] += EPSILON
                numerical_grad = np.sum( base_grad * ( \
                    layer.forward(X_aug) - layer.forward(X) \
                )) / EPSILON
                X_aug[jdx, kdx] -= EPSILON
                assert np.isclose(grad_X[jdx, kdx], np.sum(numerical_grad), atol=THRESHOLD)


def test_sequential():
    """ Gradient check for sequence of layers. """

    layers = [ Linear(NUM_INPUT_DIMENSIONS, NUM_HIDDEN_DIMENSIONS) ]
    for _ in range(NUM_HIDDEN_LAYERS - 1):
        layers.append(Linear(NUM_HIDDEN_DIMENSIONS, NUM_HIDDEN_DIMENSIONS))
    layers.append(Linear(NUM_HIDDEN_DIMENSIONS, NUM_OUTPUT_DIMENSIONS))

    activations = []
    for _ in range(len(layers) - 1):
        u = np.random.rand()
        if u <= 0.25:
            activations.append(Identity())
        elif u <= 0.5:
            activations.append(ReLU())
        elif u <= 0.75:
            activations.append(Sigmoid())
        else:
            activations.append(Tanh())

    sequential = Sequential(layers, activations)

    for _ in range(NUM_BATCHES):

        X = np.random.randn(NUM_SAMPLES, NUM_INPUT_DIMENSIONS)

        y = sequential.forward(X)
        assert y.shape == (NUM_SAMPLES, NUM_OUTPUT_DIMENSIONS)

        base_grad = np.random.randn(NUM_SAMPLES, NUM_OUTPUT_DIMENSIONS)
        gradient = sequential.backward(base_grad)
        assert len(gradient) == len(layers)

        layers_aug = copy.deepcopy(sequential.layers)

        for idx in range(len(layers)):

            assert gradient[idx][0].shape == sequential.layers[idx].weight.shape
            assert gradient[idx][1].shape == sequential.layers[idx].bias.shape

            for jdx in range(gradient[idx][0].shape[0]):
                for kdx in range(gradient[idx][0].shape[1]):
                    layers_aug[idx].weight[jdx, kdx] += EPSILON
                    numerical_grad = np.sum( base_grad * ( \
                        sequential.forward(X, layers=layers_aug) \
                        - sequential.forward(X)) \
                    ) / EPSILON
                    layers_aug[idx].weight[jdx, kdx] -= EPSILON

                    assert np.isclose(gradient[idx][0][jdx, kdx], numerical_grad, atol=THRESHOLD)

            for jdx in range(gradient[idx][1].shape[0]):
                layers_aug[idx].bias[jdx] += EPSILON
                numerical_grad = np.sum( base_grad * ( \
                    sequential.forward(X, layers=layers_aug) \
                    - sequential.forward(X)) \
                ) / EPSILON
                layers_aug[idx].bias[jdx] -= EPSILON

                assert np.isclose(gradient[idx][1][jdx], numerical_grad, atol=THRESHOLD)


# --- activation functions


def test_identity():
    """ Gradient check for Identity activation function. """

    _test_activation(Identity())


def test_relu():
    """ Gradient check for ReLU activation function. """

    _test_activation(ReLU())


def test_sigmoid():
    """ Gradient check for Sigmoid activation function. """

    _test_activation(Sigmoid())


def test_tanh():
    """ Gradient check for Tanh activation function. """

    _test_activation(Tanh())


# --- loss function


def test_mean_squared_error():
    """ Gradient check for Mean Squared Error loss function. """

    _test_loss(MeanSquaredError())


def test_softmax_cross_entropy():
    """ Gradient check for Cross Entropy loss, coupled with Softmax activation function. """

    _test_loss(CrossEntropy(), Softmax(), True)
