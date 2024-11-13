""" Assignment 5: Kernel Density Estimation, Hidden Markov Model and Recurrent Neural Network. """

import sys
from typing import Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from hmmlearn import hmm

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

    # Generate synthetic dataset
    points = np.vstack(( \
        sample_points_in_circle(3000, 2, 0, 0),
        sample_points_in_circle(500, 0.25, 1, 1)
    ))
    points += 0.01 * np.random.randn(3500, 2)

    # Plot synthetic dataset
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

    # Plot KDE estimate
    kde = KDE('gaussian', 0.25).fit(points)
    output_path = 'figures/synthetic_data_kde.png'
    kde.visualize(output_path)
    print(output_path)

    # Plot GMM memberships for two components
    output_path = 'figures/synthetic_data_gmm_2.png'
    gmm = GMM(2).fit(points)
    gmm.visualize(output_path)
    print(output_path)

    # Plot GMM memberships for three components
    output_path = 'figures/synthetic_data_gmm_3.png'
    gmm = GMM(3).fit(points)
    gmm.visualize(output_path)
    print(output_path)

    # Plot GMM memberships for four components
    output_path = 'figures/synthetic_data_gmm_4.png'
    gmm = GMM(4).fit(points)
    gmm.visualize(output_path)
    print(output_path)

    print()


def optimal_character_recognition() -> None:
    """ """


def speech_digit_recognition() -> None:
    """ Applies HMM on Free Spoken Digit Dataset to recognize spoken digits from audio signals. """

    # Log function call
    print('--- speech_digit_recognition')

    # Plot MFCC features for recordings of same digit
    output_path = 'figures/mfcc_features_same_digit.png'
    fig, axs = plt.subplots(5, 2, figsize=(14, 20))
    fig.suptitle('MFCC Features for Recordings of Same Digit')

    for idx in range(10):
        y, sr = librosa.load(f'{PROJECT_DIR}/data/external/fsdd/0_george_{idx}.wav')
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15)
        spec = librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=axs[idx // 2][idx % 2])
        axs[idx // 2][idx % 2].set_title(f'Recording {idx}')
        fig.colorbar(spec, ax=axs[idx // 2][idx % 2])

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    plt.clf()
    print(output_path)

    # Plot MFCC features for recordings of different digit
    output_path = 'figures/mfcc_features_diff_digit.png'
    fig, axs = plt.subplots(5, 2, figsize=(14, 20))
    fig.suptitle('MFCC Features for Recordings of Different Digits')

    for idx in range(10):
        y, sr = librosa.load(f'{PROJECT_DIR}/data/external/fsdd/{idx}_george_0.wav')
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15)
        spec = librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=axs[idx // 2][idx % 2])
        axs[idx // 2][idx % 2].set_title(f'Digit {idx}')
        fig.colorbar(spec, ax=axs[idx // 2][idx % 2])

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    plt.clf()
    print(output_path)

    # Extract train features and train model
    PERSONS = ['george', 'jackson', 'lucas', 'nicolas', 'theo', 'yweweler']
    list_hmm = []
    for digit in range(10):
        print(f'Training model {digit}')
        features = []
        for person in PERSONS:
            for recording in range(40):
                y, sr = librosa.load(f'{PROJECT_DIR}/data/external/fsdd/{digit}_{person}_{recording}.wav')
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15).T
                features.append(mfcc)
        model = hmm.GaussianHMM(n_components=15)
        model.fit(np.vstack(features))
        list_hmm.append(model)

    # Extract test features and evaluate model
    list_actual = []
    list_prediction = []
    for digit in range(10):
        print(f'Evaluating model {digit}')
        for person in PERSONS:
            for recording in range(40, 50):
                y, sr = librosa.load(f'{PROJECT_DIR}/data/external/fsdd/{digit}_{person}_{recording}.wav')
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15).T

                prediction = 0
                best_score = -np.inf
                for idx, model in enumerate(list_hmm):
                    score = model.score(mfcc)
                    if score > best_score:
                        prediction = idx
                        best_score = score
                list_actual.append(digit)
                list_prediction.append(prediction)

    test_accuracy = (np.array(list_actual) == np.array(list_prediction)).mean()
    print(f'Accuracy on test set: {test_accuracy}')

    # TODO Personal recordings
    # personal_accuracy = 

    print()


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
