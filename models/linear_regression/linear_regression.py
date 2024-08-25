import os
import sys
from typing import Literal, Self

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image

PROJECT_DIR = '../..'
sys.path.insert(0, PROJECT_DIR)

from performance_measures import RegressionMeasures


class LinearRegression:
    """ Implements Linear Regression with regularization, using Gradient Descent. """


    def __init__(self, k: int = 1, lr: float = 0.01, alpha: float = 0.1,
        regularizer: Literal['l1', 'l2'] | None = None
    ):
        """ Initializes the model hyperparameters.

        Args:
            k: The order of the model.
            lr: The learning rate to use in gradient descent.
            alpha: The coefficient to use for regularization term.
            regularizer: The regularizer to use in conjunction with MSE loss.
        """

        # Validate the passed arguments
        assert k >= 1, 'order should be atleast 1'
        assert lr > 0.0, 'lr should be positive'
        assert alpha >= 0.0, 'alpha should be non-negative'
        assert regularizer in ['l1', 'l2', None], f'Unrecognized value for regularizer {regularizer}'

        # Store the passed arguments
        self.k = k
        self.lr = lr
        self.alpha = alpha
        self.regularizer = regularizer

        # Initialize the model parameters to None
        self.coeff = None


    def fit(self, x_train: NDArray, y_train: NDArray, eps: float = 1e-3, stop_after_max_iterations: bool = False,
        max_iterations: int = 500, random_seed: int | None = 0, animation_path: str | None = None,
        iterations_per_frame: int = 15, frame_duration: int = 75
    ) -> Self:
        """ Fits the model for the given training data. """

        # Reinitialize the random number generator 
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize the model parameters with normal values
        self.coeff = np.random.randn(1, self.k + 1)

        # Generate powers of x_train
        X_train = x_train[:, np.newaxis] ** np.arange(self.k + 1)

        # Variable to count the number of iterations
        num_iterations = 0

        # If animation not required
        if animation_path is None:

            # Repeat until termination condition is satisfied
            while self._gradient_descent_one_step(X_train, y_train, eps):
                num_iterations += 1
                if stop_after_max_iterations and num_iterations > max_iterations:
                    break

        else:

            # Initialize empty list to store evaluation metrics for each iteration
            os.makedirs('.tmp/', exist_ok=True)
            iterations = []
            all_mse = []
            all_std = []
            all_var = []

            # Useful statistics of the training data
            x_train_min = np.min(x_train)
            x_train_max = np.max(x_train)
            x_train_domain = np.linspace(x_train_min, x_train_max, 400)
            y_train_std = np.std(y_train)
            y_train_var = np.var(y_train)

            while self._gradient_descent_one_step(X_train, y_train, eps) and num_iterations < max_iterations:

                if num_iterations % iterations_per_frame == 0:

                    # Compute evaluation metrics for current iteration
                    y_train_pred = self.predict(x_train)
                    y_train_domain = self.predict(x_train_domain)
                    reg_measures = RegressionMeasures(y_train, y_train_pred)
                    all_mse.append(reg_measures.mean_squared_error())
                    all_std.append(reg_measures.standard_deviation())
                    all_var.append(reg_measures.variance())
                    iterations.append(num_iterations)

                    # Create figure for subplots
                    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
                    if self.regularizer is None:
                        fig.suptitle(f'Linear Regression (k = {self.k}), across iterations')
                    else:
                        fig.suptitle(f'Linear Regression (k = {self.k}, {str.upper(self.regularizer)} Regularization), across iterations')

                    # Generate plot for line fit and residuals
                    ax[0][0].scatter(x_train, y_train, label='Training samples', s=5)
                    ax[0][0].plot(x_train_domain, y_train_domain, label='Fitted line', c='darkorange')
                    ax[0][0].vlines(x_train[0], y_train[0], y_train_pred[0], label='Residuals', color='g', linewidth=1)
                    for idx in range(1, len(x_train), 5):
                        ax[0][0].vlines(x_train[idx], y_train[idx], y_train_pred[idx], color='g', linewidth=1)
                    ax[0][0].set_title('Line Fit and Residuals')
                    ax[0][0].set_xlabel('x')
                    ax[0][0].set_ylabel('y')
                    ax[0][0].legend()

                    # Generate plot for MSE
                    ax[0][1].plot(iterations, all_mse, c='k')
                    ax[0][1].set_title('MSE')
                    ax[0][1].set_xlabel('Number of Iterations')
                    ax[0][1].set_ylabel('Mean Squared Error')
                    ax[0][1].set_xlim(0, max_iterations)

                    # Generate plot for standard deviation
                    ax[1][0].plot(iterations, all_std, label='Predicted values', c='darkorange')
                    ax[1][0].axhline(y_train_std, label='True values', linestyle='--', c='b')
                    ax[1][0].set_title('Standard Deviation')
                    ax[1][0].set_xlabel('Number of Iterations')
                    ax[1][0].set_ylabel('Standard Deviation')
                    ax[1][0].set_xlim(0, max_iterations)
                    ax[1][0].legend()

                    # Generate plot for variance
                    ax[1][1].plot(iterations, all_var, label='Predicted values', c='darkorange')
                    ax[1][1].axhline(y_train_var, label='True values', linestyle='--', c='b')
                    ax[1][1].set_title('Variance')
                    ax[1][1].set_xlabel('Number of Iterations')
                    ax[1][1].set_ylabel('Variance')
                    ax[1][1].set_xlim(0, max_iterations)
                    ax[1][1].legend()

                    # Save figure to a temporary folder
                    plt.tight_layout()
                    plt.savefig(f'.tmp/{num_iterations}.png', bbox_inches='tight')
                    plt.close()
                    plt.clf()

                num_iterations += 1

            if animation_path is not None:

                # Create GIF
                image_paths = [f'.tmp/{idx}.png' for idx in range(0, num_iterations, iterations_per_frame)]
                images = [Image.open(img_path) for img_path in image_paths]
                images[0].save(
                    animation_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=frame_duration,
                    loop=0
                )

                # Remove figures from temporary folder
                for path in image_paths:
                    try:
                        os.remove(path)
                    except OSError:
                        pass

        return self


    def _gradient_descent_one_step(self, X_train: NDArray, y_train: NDArray, eps) -> bool:
        """ Performs one step of gradient descent. """

        # Compute predictions
        y_pred = np.sum(self.coeff * X_train, axis=1)

        # Compute error for each train sample
        error = y_train - y_pred

        # Compute gradient matrix
        gradient_matrix = - 2 * X_train * error[:, np.newaxis]
        gradient_vector = np.mean(gradient_matrix, axis=0)[np.newaxis, :]

        # Add regularization term
        if self.regularizer == 'l1':
            gradient_vector += self.alpha * (self.coeff != 0).astype(float)
        elif self.regularizer == 'l2':
            gradient_vector += 2 * self.alpha * self.coeff

        # Apply the gradient update rule
        self.coeff = self.coeff - self.lr * gradient_vector

        # Termination condition
        if np.linalg.norm(gradient_vector) < eps:
            return False
        
        return True


    def load_parameters(file_name: str) -> None:
        """ Load model parameters from a file. """

        # Load coefficients from file
        self.coeff = np.load(file_name)


    def predict(self, x_test: NDArray) -> NDArray:
        """ Returns the prediction for an array of test samples."""

        # Check if fit method called before predict
        assert self.coeff is not None, 'fit method should be called before predict'

        # Generate powers of X_test
        X_test = x_test[:, np.newaxis] ** np.arange(self.k + 1)

        # Compute predictions
        y_pred = np.sum(self.coeff * X_test, axis=1)

        return y_pred


    def save_parameters(self, file_name: str) -> None:
        """ Save model parameters to a file. """

        # Check if fit method called before save
        assert self.coeff is not None, 'fit method should be called before save'

        # Save coefficients to file
        np.save(file_name, self.coeff)
