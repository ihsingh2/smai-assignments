""" Assignment 5: Kernel Density Estimation, Hidden Markov Model and Recurrent Neural Network. """

import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# pylint: disable=wrong-import-position

PROJECT_DIR = '../..'
sys.path.insert(0, PROJECT_DIR)

from models.kde import KDE
from models.gmm import GMM

# pylint: enable=wrong-import-position


def bit_counting() -> None:
    """ """


def kernel_density_estimation() -> None:
    """ Applies KDE on a synthetic dataset and contrasts it with GMM. """

    def sample_points_in_circle(n, r, c_x, c_y):
        """ Samples n points in a circle of radius r, centered at (c_x, c_y). """

        points = []
        for _ in range(n):
            radius = r * np.sqrt(np.random.uniform(0, 1))
            angle = np.random.uniform(0, 2 * np.pi)
            x = c_x + radius * np.cos(angle)
            y = c_y + radius * np.sin(angle)
            points.append([x, y])
        return np.array(points)

    # Log function call
    print('--- kernel_density_estimation')

    points = np.vstack(( \
        sample_points_in_circle(3000, 2, 0, 0),
        sample_points_in_circle(500, 0.25, 1, 1)
    ))
    points += 0.01 * np.random.randn(3500, 2)

    # Plot the distribution of various attributes
    output_path = 'figures/synthetic_data.png'
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], s=1)
    plt.title('Synthetic Dataset')
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    plt.clf()
    print(output_path)

    kde = KDE('gaussian', 0.25).fit(points)
    output_path = 'figures/synthetic_data_kde.png'
    kde.visualize(output_path)
    print(output_path)

    output_path = 'figures/synthetic_data_gmm_2.png'
    gmm = GMM(2).fit(points)
    gmm.visualize(output_path)
    print(output_path)

    output_path = 'figures/synthetic_data_gmm_3.png'
    gmm = GMM(3).fit(points)
    gmm.visualize(output_path)
    print(output_path)

    output_path = 'figures/synthetic_data_gmm_4.png'
    gmm = GMM(4).fit(points)
    gmm.visualize(output_path)
    print(output_path)

    print()


def optimal_character_recognition() -> None:
    """ """


def speech_digit_recognition() -> None:
    """ """


# pylint: disable=duplicate-code

# pylint: disable-next=too-many-arguments, too-many-positional-arguments
def train_val_test_split(
    X: npt.NDArray, y: npt.NDArray, train_size: float = 0.8, val_size: float = 0.1,
    test_size: float = 0.1, random_seed: int | None = 0
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """ Partitions dataset represented as a pair of array, into three groups. """

    # Reinitialize the random number generator
    if random_seed is not None:
        np.random.seed(random_seed)

    # Ensure the sizes form a probability simplex
    assert train_size + val_size + test_size == 1.0, \
                                    'train_size, val_size, and test_size sizes must sum to 1.'
    assert 0.0 <= train_size <= 1.0, 'train_size must lie in (0, 1)'
    assert 0.0 <= val_size <= 1.0, 'val_size must lie in (0, 1)'
    assert 0.0 <= test_size <= 1.0, 'test_size must lie in (0, 1)'

    # Ensure that X and y are of same length
    assert X.shape[0] == y.shape[0], 'Expected X and y to be the same length'

    # Shuffle the indices
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    # Shuffle the dataset as per the indices
    X = X[indices]
    y = y[indices]

    # Compute the splitting indices
    train_end = int(train_size * X.shape[0])
    val_end = train_end + int(val_size * X.shape[0])

    # Split the data
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test

# pylint: enable=duplicate-code


if __name__ == '__main__':

    # 2 KDE
    kernel_density_estimation()

    # 3 HMMs
    speech_digit_recognition()

    # 4 RNNs
    bit_counting()
    optimal_character_recognition()
