""" Assignment 3: Multi Layer Perceptron and AutoEncoders. """

import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

# pylint: disable=wrong-import-position

PROJECT_DIR = '../..'
sys.path.insert(0, PROJECT_DIR)

from models.AutoEncoders import AutoEncoders
from models.knn import KNN
from models.MLP import MLP
from models.MLP.activation import Identity, ReLU, Sigmoid, Tanh
from models.MLP.loss import CrossEntropy, MeanSquaredError
from performance_measures import ClassificationMeasures

# pylint: enable=wrong-import-position

# Library customizations
pd.set_option('display.max_columns', None)


def mlp_classification_hyperparameter_tuning() -> None:
    """ Hyperparameter tuning for Multi Layer Perceptron Classification on Wine Quality Dataset. """

    # Read interim CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/interim/3/WineQT.csv')

    # Convert DataFrame to array
    X = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, -1]

    # Start class labels from zero
    y -= np.min(y)
    y = y.astype(int)

    # Split the array into train, test and split
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    # Initialize and fit model
    mlp = MLP(num_hidden_layers=2, num_neurons_per_layer=5, classify=True,
                                        activation=Sigmoid(), loss=CrossEntropy())
    mlp.fit(X_train, y_train)

    # Evaluate the model on validation set
    mlp.predict(X_val)


def wineqt_analysis_preprocessing() -> None:
    """ Analyze and preprocess Wine Quality Dataset. """

    # Read external CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/external/WineQT.csv', index_col=-1)

    # Description of the dataset
    desc = df.describe()
    print(desc.loc[['mean', 'std', 'min', 'max']])

    # TODO: graph that shows the distribution of the various labels across the entire dataset.

    # Backup of discrete labels
    quality_copy = df['quality'].copy()

    # Convert all columns to floating point
    df = df.astype(float)

    # Apply standardization to all columns
    df = (df - df.mean()) / df.std()

    # Revert back labels to original discrete values
    df['quality'] = df['quality'].astype(int)
    df['quality'] = quality_copy

    # Write processed DataFrames to CSV files
    df.to_csv(f'{PROJECT_DIR}/data/interim/3/WineQT.csv')


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

    # 2 Multi Layer Perceptron Classification

    ## 2.1 Dataset Analysis and Preprocessing
    wineqt_analysis_preprocessing()

    ## 2.3 Model Training & Hyperparameter Tuning
    mlp_classification_hyperparameter_tuning()
