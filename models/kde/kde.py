""" Provides the KDE class. """

from typing import Literal, Self

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


class KDE:
    """ Implements Kernel Density Estimation. """


    def __init__(self, kernel: Literal['box', 'gaussian', 'triangular'], bandwidth: float):
        """ Initializes the model hyperparameters.

        Args:
            kernel: The kernel function to use.
            bandwidth: The bandwidth to equip in the kernel function.
        """

        # Validate the passed arguments
        assert kernel in ['box', 'gaussian', 'triangular'], \
                                            f'Unrecognized value passed for kernel {kernel}'
        assert bandwidth > 0, 'bandwidth should be positive'

        # Store the passed arguments
        self.kernel = kernel
        self.bandwidth = bandwidth

        # Initialize the training data to None
        self.X_train = None


    def fit(self, X_train: npt.NDArray) -> Self:
        """ Fits the model for the given training data. """

        # Store the training data
        self.X_train = X_train

        return self


    def _kernel(self, distance: npt.NDArray) -> npt.NDArray:
        """ Applies the kernel function to an array of distances. """

        if self.kernel == 'box':
            return (np.abs(distance) <= self.bandwidth).astype(float)

        if self.kernel == 'gaussian':
            return (1 / (np.sqrt(2 * np.pi) * self.bandwidth)) * \
                                                    np.exp(-0.5 * (distance / self.bandwidth) ** 2)

        return (1 - np.abs(distance / self.bandwidth)) * \
                                                (np.abs(distance) <= self.bandwidth).astype(float)


    def predict(self, X_test: npt.NDArray) -> npt.NDArray:
        """ Returns the density for an array of test samples."""

        # Check if fit method before predict
        assert self.X_train is not None, 'fit method should be called before predict'

        # Compute the densities
        density = np.zeros(X_test.shape[0])
        for i, point in enumerate(X_test):
            distances = np.linalg.norm(self.X_train - point, axis=1)
            kernels = self._kernel(distances)
            density[i] = np.sum(kernels) / \
                                (self.X_train.shape[0] * self.bandwidth ** self.X_train.shape[1])

        return density


    def visualize(self, output_path: str = None) -> None:
        """ Visualizes the density for two dimensional training data. """

        if self.X_train.shape[1] != 2:
            raise ValueError("Visualization is only supported for 2D data.")

        x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
        y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1
        x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

        density = self.predict(grid_points).reshape(100, 100)

        plt.figure(figsize=(10, 8))
        plt.contourf(x_grid, y_grid, density, levels=20, cmap='viridis')
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c='red', s=1)
        plt.title(f'Kernel Density Estimate ({self.kernel} kernel, bandwidth={self.bandwidth})')
        plt.grid()
        plt.colorbar()
        plt.tight_layout()

        if output_path is not None:
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            plt.clf()

        else:
            plt.show()
