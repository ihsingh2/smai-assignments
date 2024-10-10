""" Assignment 1: K-Nearest Neighbours and Linear Regression. """

import sys
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

#pylint: disable=wrong-import-position

PROJECT_DIR = '../..'
sys.path.insert(0, PROJECT_DIR)

from models.knn import KNN, OldKNN
from models.linear_regression import LinearRegression
from performance_measures import ClassificationMeasures, RegressionMeasures

#pylint: enable=wrong-import-position


def knn_drop_columns() -> None:
    """ Test KNN on a reduced dataset, dropping different combinations of columns. """

    # Different combinations of columns to drop
    combinations = [
        None,
        'key',
        'liveness',
        'time_signature',
        'mode',
        'explicit',
        'loudness',
        'acousticness',
        'speechiness',
        'instrumentalness',
        'energy',
        'valence',
        'tempo',
        'danceability',
        'duration_ms',
        'popularity',
        [
            'key',
            'time_signature',
            'mode',
            'explicit'
        ],
        [
            'key',
            'liveness',
            'time_signature',
            'mode',
            'explicit'
        ],
        [
            'key',
            'liveness',
            'time_signature',
            'mode',
            'explicit',
            'loudness',
            'speechiness',
        ],
        [
            'key',
            'liveness',
            'time_signature',
            'mode',
            'explicit',
            'acousticness',
            'instrumentalness',
        ]
    ]

    # Read the best hyperparameters from file
    with open('results/knn_hyper_params.txt', 'r', encoding='utf-8') as file:
        k, metric, _ = file.readline().strip().split(', ')
        k = int(k)

    # Read interim CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/interim/1/spotify.csv', index_col=0)

    # Initialize an empty dictionary for storing accuracy
    all_accuracy = {}

    # Iterate over different combinations
    for choice in combinations:

        # Drop columns based on current choice
        if choice is not None:
            df_reduced = df.drop(columns=choice)
        else:
            df_reduced = df.copy()

        # Convert DataFrame to array
        X = df_reduced.to_numpy()[:, :-1]
        y = df_reduced.to_numpy()[:, -1].astype(int)

        # Split the array into train, validation and test
        X_train, X_val, _, y_train, y_val, _ = train_val_test_split(X, y)

        # Initialize and train the model
        knn = KNN(k, metric)
        knn.fit(X_train, y_train)

        # Compute predictions on the validation set
        y_pred = knn.predict(X_val)

        # Evaluate predictions for the validation set
        cls_measures = ClassificationMeasures(y_val, y_pred)
        accuracy = cls_measures.accuracy_score()

        # Store the accuracy for comparision
        if isinstance(choice, list):
            all_accuracy[tuple(choice)] = accuracy
        else:
            all_accuracy[choice] = accuracy

        # Log progress
        print(choice, accuracy)

    # Extract the top 10 hyperparameters by accuracy
    best_combination = sorted(all_accuracy, key=all_accuracy.get, reverse=True)[:10]

    # Write the top 10 hyperparameters to a file
    print()
    with open('results/knn_drop_columns.txt', 'w', encoding='utf-8') as file:
        for param in best_combination:
            file.write(f'{param}, {all_accuracy[param]}\n')
            print(f'{param}, {all_accuracy[param]}')
    print()


def knn_hyperparameter_tuning() -> None:
    """ Test KNN for different hyperparameters, and save the top 10 hyperparameters. """

    # Set of hyperparameters
    k_list = range(3, 36, 2)
    metric_list = ['manhattan', 'euclidean', 'cosine']

    # Read interim CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/interim/1/spotify.csv', index_col=0)

    # Convert DataFrame to array
    X = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, -1].astype(int)

    # Split the array into train, validation and test
    X_train, X_val, _, y_train, y_val, _ = train_val_test_split(X, y)

    # Initialize an empty dictionary for storing accuracy
    all_accuracy = {}

    # Iterate over all combinations of hyperparameters
    for k in k_list:
        for metric in metric_list:

            # Initialize and train the model
            knn = KNN(k, metric)
            knn.fit(X_train, y_train)

            # Compute predictions on the validation set
            y_pred = knn.predict(X_val)

            # Evaluate predictions for the validation set
            cls_measures = ClassificationMeasures(y_val, y_pred)
            accuracy = cls_measures.accuracy_score()

            # Store the accuracy for comparision
            all_accuracy[(k, metric)] = accuracy

            # Log progress
            print(k, metric, accuracy)

    # Extract the top 10 hyperparameters by accuracy
    best_hyperparameters = sorted(all_accuracy, key=all_accuracy.get, reverse=True)[:10]

    # Write the top 10 hyperparameters to a file
    print()
    with open('results/knn_hyper_params.txt', 'w', encoding='utf-8') as file:
        for param in best_hyperparameters:
            file.write(f'{param[0]}, {param[1]}, {all_accuracy[param]}\n')
            print(f'{param[0]}, {param[1]}, {all_accuracy[param]}')
    print()


def knn_inference_time() -> None:
    """ Run KNN for different models and plot their respective inference times. """

    # Read the best hyperparameters from file
    with open('results/knn_hyper_params.txt', 'r', encoding='utf-8') as file:
        best_k, best_metric, _ = file.readline().strip().split(', ')
        best_k = int(best_k)

    # Categories for testing
    categories = [
        'Initial KNN',
        'Best KNN',
        'Optimized KNN',
        'Sklearn KNN'
    ]

    # Different combinations for testing
    model_combinations = [
        OldKNN(3, 'manhattan'),
        OldKNN(best_k, best_metric),
        KNN(best_k, best_metric),
        KNeighborsClassifier(best_k, metric=best_metric, algorithm='brute')
    ]
    train_size_combinations = [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05
    ]

    # Read interim CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/interim/1/spotify.csv', index_col=0)

    # Convert DataFrame to array
    X = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, -1].astype(int)

    # ------------------------------ COMPLETE DATASET ------------------------------

    # Split the array into train, validation and test
    X_train, X_val, _, y_train, _, _ = train_val_test_split(X, y)

    # Initialize an empty list to store the execution times
    exec_times = []

    # Iterate over all model combinations
    for model in model_combinations:

        # Train model
        model.fit(X_train, y_train)

        # Measure inference time
        start_time = time.time()
        model.predict(X_val)
        end_time = time.time()

        # Compute inference time
        exec_time = end_time - start_time
        exec_times.append(exec_time)

        # Log progress
        print(model, exec_time)

    # Plot inference time as bar graph
    plt.bar(categories, exec_times)
    plt.title('Inference times for different models')
    plt.xlabel('Models')
    plt.ylabel('Inference time (seconds)')
    plt.savefig('figures/knn_inference_time_models.png', bbox_inches='tight')
    plt.close()
    plt.clf()

    # ------------------------------ VARYING DATASET SIZE ------------------------------

    # Initialize empty lists to store the execution times
    exec_times = [[] for _ in range(len(model_combinations))]

    # Iterate over all train size combinations
    for train_size in train_size_combinations:

        # Split the array into train, validation and test
        X_train, X_val, _, y_train, _, _ = train_val_test_split(X, y, \
                        train_size=train_size, val_size=0.1, test_size=1-(train_size+0.1))

        # Iterate over all model combinations
        for idx, model in enumerate(model_combinations):

            # Train model
            model.fit(X_train, y_train)

            # Measure inference time
            start_time = time.time()
            model.predict(X_val)
            end_time = time.time()

            # Compute inference time
            exec_time = end_time - start_time
            exec_times[idx].append(exec_time)

            # Log progress
            print(train_size, model, exec_time)
    print()

    # Map relative dataset size to absolute
    abs_train_sizes = [ train_size * len(X) for train_size in train_size_combinations ]

    # Plot inference time as line graph
    for idx, category in enumerate(categories):
        plt.plot(abs_train_sizes, exec_times[idx], label=category)
    plt.title('Inference times for different training dataset sizes')
    plt.xlabel('Size of training dataset')
    plt.ylabel('Inference time (seconds)')
    plt.legend()
    plt.savefig('figures/knn_inference_time_train_sizes.png', bbox_inches='tight')
    plt.close()
    plt.clf()


def knn_k_values() -> None:
    """ Test KNN for different values of k, and plot the graph for k vs accuracy. """

    # Different k values to try
    k_list = range(5, 20, 2)
    metric = 'manhattan'

    # Read interim CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/interim/1/spotify.csv', index_col=0)

    # Convert DataFrame to array
    X = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, -1].astype(int)

    # Split the array into train, validation and test
    X_train, X_val, _, y_train, y_val, _ = train_val_test_split(X, y)

    # Initialize an empty list for storing accuracy
    all_accuracy = []

    # Iterate over different combinations
    for k in k_list:

        # Initialize and train the model
        knn = KNN(k, metric)
        knn.fit(X_train, y_train)

        # Compute predictions on the validation set
        y_pred = knn.predict(X_val)

        # Evaluate predictions for the validation set
        cls_measures = ClassificationMeasures(y_val, y_pred)
        accuracy = cls_measures.accuracy_score()

        # Store the accuracy for comparision
        all_accuracy.append(accuracy)

        # Log progress
        print(f'{k}, {accuracy}')
    print()

    # Generate plot
    plt.plot(k_list, all_accuracy)
    plt.title('Accuracy for different values of k (manhattan distance)')
    plt.xlabel('k')
    plt.ylabel('Validation accuracy')
    plt.savefig('figures/knn_accuracy_k.png', bbox_inches='tight')
    plt.close()
    plt.clf()


def knn_second_dataset() -> None:
    """ Apply KNN on the second dataset, with predefined train, validation and test split. """

    # Read interim CSV files into DataFrames
    df_train = pd.read_csv(f'{PROJECT_DIR}/data/interim/1/spotify-2/train.csv', index_col=0)
    df_test = pd.read_csv(f'{PROJECT_DIR}/data/interim/1/spotify-2/test.csv', index_col=0)

    # Convert DataFrames to NumPy array
    X_train = df_train.to_numpy()[:, :-1]
    X_test = df_test.to_numpy()[:, :-1]
    y_train = df_train.to_numpy()[:, -1].astype(int)
    y_test = df_test.to_numpy()[:, -1].astype(int)

    # Read the best hyperparameters from file
    with open('results/knn_hyper_params.txt', 'r', encoding='utf-8') as file:
        k, metric, _ = file.readline().strip().split(', ')
        k = int(k)

    # Initialize and train the model
    knn = KNN(k, metric)
    knn.fit(X_train, y_train)

    # Compute predictions on the test set
    y_pred = knn.predict(X_test)

    # Evaluate predictions for the test set
    cls_measures = ClassificationMeasures(y_test, y_pred)
    accuracy = cls_measures.accuracy_score()

    # Log progress
    print(k, metric, accuracy)

    # Write the accuracy to a file
    with open('results/knn_second_dataset.txt', 'w', encoding='utf-8') as file:
        file.write(f'{k}, {metric}, {accuracy}\n')


def preprocess_spotify_dataset() -> None:
    """ Remove redundant rows and columns, and standardize the Spotify dataset. """

    def preprocess_worker(df, categories_map):

        # Remove duplicate rows for same track
        df.drop_duplicates(['track_id'], inplace=True)

        # Remove columns with nominal data and > 1000 values
        df.drop(columns=['track_id', 'artists', 'album_name', 'track_name'], inplace=True)

        # Encode genre as an integer
        df['track_genre'] = df['track_genre'].map(categories_map)
        track_genre_copy = df['track_genre'].copy()

        # Convert boolean columns to integer datatype
        df['explicit'] = df['explicit'].astype(int)

        # Convert all columns to floating point
        df = df.astype(float)

        # Apply standardization to all columns
        df = (df - df.mean()) / df.std()

        # Revert back genre to original integer encoding
        df['track_genre'] = df['track_genre'].astype(int)
        df['track_genre'] = track_genre_copy

        return df

    # Read external CSV files into DataFrames
    df_main = pd.read_csv(f'{PROJECT_DIR}/data/external/spotify.csv', index_col=0)
    df_train = pd.read_csv(f'{PROJECT_DIR}/data/external/spotify-2/train.csv', index_col=0)
    df_validate = pd.read_csv(f'{PROJECT_DIR}/data/external/spotify-2/validate.csv', index_col=0)
    df_test = pd.read_csv(f'{PROJECT_DIR}/data/external/spotify-2/test.csv', index_col=0)

    # Generate encoding for target variable
    categories = pd.Categorical(df_main['track_genre']).categories
    categories_map = dict(zip(categories, range(len(categories))))

    # Process the DataFrames individually
    df_main = preprocess_worker(df_main, categories_map)
    df_train = preprocess_worker(df_train, categories_map)
    df_validate = preprocess_worker(df_validate, categories_map)
    df_test = preprocess_worker(df_test, categories_map)

    # Write processed DataFrames to CSV files
    df_main.to_csv(f'{PROJECT_DIR}/data/interim/1/spotify.csv')
    df_train.to_csv(f'{PROJECT_DIR}/data/interim/1/spotify-2/train.csv')
    df_validate.to_csv(f'{PROJECT_DIR}/data/interim/1/spotify-2/validate.csv')
    df_test.to_csv(f'{PROJECT_DIR}/data/interim/1/spotify-2/test.csv')


def regularization() -> None:
    """ Fit a polynomial of different degress with and without L1 and L2 regularization.
    Record the MSE, standard deviation and variance, and plot the resulting curves. """

    # Read external CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/external/regularisation.csv')

    # Convert DataFrame to array
    x = df.to_numpy()[:, 0]
    y = df.to_numpy()[:, 1]

    # Split the array into train, validation and test
    x_train, _, x_test, y_train, _, y_test = train_val_test_split(x, y)

    # Domain of the training data
    x_train_min = np.min(x_train)
    x_train_max = np.max(x_train)
    x_train_domain = np.linspace(x_train_min, x_train_max, 400)

    # Clear contents of output file
    with open('results/regularization.txt', 'w', encoding='utf-8') as file:
        pass

    # Initialize the minimum test loss and corresponding model
    min_test_mse = np.inf
    min_test_mse_model = None
    min_test_mse_regularizer = None

    # Iterate over different combinations
    for k in range(1,21):
        for regularizer in [None, 'l1', 'l2']:

            # Initialize and fit the model to data
            reg = LinearRegression(k=k, regularizer=regularizer)
            reg.fit(x, y, max_iterations=5000, stop_after_max_iterations=True)

            # Test on train data
            y_train_pred = reg.predict(x_train)
            train_reg_measures = RegressionMeasures(y_train, y_train_pred)

            # Test on test data
            y_test_pred = reg.predict(x_test)
            test_reg_measures = RegressionMeasures(y_test, y_test_pred)

            test_mse = test_reg_measures.mean_squared_error()
            if test_mse < min_test_mse:
                min_test_mse = test_mse
                min_test_mse_model = reg
                min_test_mse_regularizer = regularizer

            # Evaluate measures on train and test predictions
            evaluated_measures = [
                ('Train MSE', train_reg_measures.mean_squared_error()),
                ('Train Standard Deviation', train_reg_measures.standard_deviation()),
                ('Train Variance', train_reg_measures.variance()),
                ('Test MSE', test_reg_measures.mean_squared_error()),
                ('Test Standard Deviation', test_reg_measures.standard_deviation()),
                ('Test Variance', test_reg_measures.variance())
            ]

            # Write the evaluated measures to file
            with open('results/regularization.txt', 'a', encoding='utf-8') as file:
                if regularizer is None:
                    file.write(f'k = {k}, No Regularization\n')
                else:
                    file.write(f'k = {k}, {str.upper(regularizer)} Regularization\n')
                file.write('--------------------------------------------------\n')
                for measure, score in evaluated_measures:
                    file.write(f'{measure}: {score}\n')
                file.write('\n')

            # Plot training points
            y_train_domain = reg.predict(x_train_domain)
            if regularizer is None:
                output_path = f'figures/regularization_{k}_None.png'
                plt.title(f'Regularization (k = {k}, No Regularization)')
            else:
                output_path = f'figures/regularization_{k}_{regularizer}.png'
                plt.title(f'Regularization (k = {k}, {str.upper(regularizer)} Regularization)')
            plt.scatter(x_train, y_train, s=5, label='Training samples')
            plt.plot(x_train_domain, y_train_domain, label='Fitted line', c='darkorange')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            plt.clf()
            print(output_path)

    # Save the result summary to file
    with open('results/regularization.txt', 'a', encoding='utf-8') as file:
        file.write(f'Observed minimum for k = {min_test_mse_model.k}, ' \
                   f'regularizer = {min_test_mse_regularizer}\n')
        file.write(f'Test MSE: {min_test_mse}\n')
        print(f'Observed minimum for k = {min_test_mse_model.k}, ' \
              f'regularizer = {min_test_mse_regularizer}')
        print(f'Test MSE: {min_test_mse}')

    # Save model parameters to a file
    min_test_mse_model.save_parameters('results/regression_params.npy')


def simple_regression_animation() -> None:
    """ Fit a polynomial of different degress and plot the training points with the fitted line,
    for each iteration, and save the sequence of plots as a GIF. """

    # Possible orders for polynomials
    k_combinations = [1, 2, 3, 5, 7, 9, 11]

    # Read external CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/external/linreg.csv')

    # Convert DataFrame to array
    x = df.to_numpy()[:, 0]
    y = df.to_numpy()[:, 1]

    # Iterate over different forms of polynomials
    for k in k_combinations:

        # Output path
        animation_path = f'figures/regression_polynomial_{k}.gif'

        # Initialize and fit the model to data
        reg = LinearRegression(k=k)
        reg.fit(x, y, animation_path=animation_path)

        # Log completion of output generation
        print(animation_path)


def simple_regression_line() -> None:
    """ Fit a polynomial of different degress and plot the training points with the fitted line. """

    # Read external CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/external/linreg.csv')

    # Convert DataFrame to array
    x = df.to_numpy()[:, 0]
    y = df.to_numpy()[:, 1]

    # Split the array into train, validation and test
    x_train, _, x_test, y_train, _, y_test = train_val_test_split(x, y)

    # Initialize and fit the model to data
    reg = LinearRegression()
    reg.fit(x, y)

    # Test on train data
    y_train_pred = reg.predict(x_train)
    train_reg_measures = RegressionMeasures(y_train, y_train_pred)

    # Test on test data
    y_test_pred = reg.predict(x_test)
    test_reg_measures = RegressionMeasures(y_test, y_test_pred)

    # Evaluate measures on train and test predictions
    evaluated_measures = [
        ('Train MSE', train_reg_measures.mean_squared_error()),
        ('Train Standard Deviation', train_reg_measures.standard_deviation()),
        ('Train Variance', train_reg_measures.variance()),
        ('Test MSE', test_reg_measures.mean_squared_error()),
        ('Test Standard Deviation', test_reg_measures.standard_deviation()),
        ('Test Variance', test_reg_measures.variance())
    ]

    # Write the evaluated measures to a file
    with open('results/regression_line.txt', 'w', encoding='utf-8') as file:
        for measure, score in evaluated_measures:
            file.write(f'{measure}: {score}\n')
            print(f'{measure}: {score}')

    # Plot training points
    output_path = 'figures/regression_degree_1.png'
    plt.scatter(x_train, y_train, s=5, label='Training samples')
    plt.plot(x_train, y_train_pred, label='Fitted line', c='darkorange')
    plt.title('Simple Regression (Degree 1)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    plt.clf()
    print(output_path)


def simple_regression_polynomial() -> None:
    """ Fit a polynomial of different degress and save the best model parameters. """

    # Read external CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/external/linreg.csv')

    # Convert DataFrame to array
    x = df.to_numpy()[:, 0]
    y = df.to_numpy()[:, 1]

    # Split the array into train, validation and test
    x_train, _, x_test, y_train, _, y_test = train_val_test_split(x, y)

    # Clear contents of output file
    with open('results/regression_polynomial.txt', 'w', encoding='utf-8') as file:
        pass

    # Initialize the minimum test loss and corresponding model
    min_test_mse = np.inf
    min_test_mse_model = None

    # Iterate over different forms of polynomials
    for k in range(1,21):

        # Initialize and fit the model to data
        reg = LinearRegression(k=k)
        reg.fit(x, y)

        # Test on train data
        y_train_pred = reg.predict(x_train)
        train_reg_measures = RegressionMeasures(y_train, y_train_pred)

        # Test on test data
        y_test_pred = reg.predict(x_test)
        test_reg_measures = RegressionMeasures(y_test, y_test_pred)

        test_mse = test_reg_measures.mean_squared_error()
        if test_mse < min_test_mse:
            min_test_mse = test_mse
            min_test_mse_model = reg

        # Evaluate measures on train and test predictions
        evaluated_measures = [
            ('Train MSE', train_reg_measures.mean_squared_error()),
            ('Train Standard Deviation', train_reg_measures.standard_deviation()),
            ('Train Variance', train_reg_measures.variance()),
            ('Test MSE', test_reg_measures.mean_squared_error()),
            ('Test Standard Deviation', test_reg_measures.standard_deviation()),
            ('Test Variance', test_reg_measures.variance())
        ]

        # Write the evaluated measures to file
        with open('results/regression_polynomial.txt', 'a', encoding='utf-8') as file:
            file.write(f'k = {k}\n')
            file.write('--------------------------------------------------\n')
            for measure, score in evaluated_measures:
                file.write(f'{measure}: {score}\n')
            file.write('\n')

    # Save the result summary to file
    with open('results/regression_polynomial.txt', 'a', encoding='utf-8') as file:
        file.write(f'Observed minimum Test MSE for k = {min_test_mse_model.k}\n')
        file.write(f'Test MSE: {min_test_mse}\n')
        print(f'Observed minimum for k = {min_test_mse_model.k}')
        print(f'Test MSE: {min_test_mse}')

    # Save model parameters to a file
    min_test_mse_model.save_parameters('results/regression_params.npy')


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


def visualize_regression_splits() -> None:
    """ Visualize the dataset splits used for simple regression task. """

    # Read external CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/external/linreg.csv')

    # Convert DataFrame to array
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()

    # Create splits based on predefined random seed
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y)

    # Generate scatter plot
    output_path = 'figures/regression_scatter_plot.png'
    plt.scatter(x_train, y_train, s=5, label='Train')
    plt.scatter(x_val, y_val, s=5, label='Validation')
    plt.scatter(x_test, y_test, s=5, label='Test')
    plt.title('Simple Regression Data Split')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    plt.clf()
    print(output_path)


def visualize_regularization_splits() -> None:
    """ Visualize the dataset splits used for regularization task. """

    # Read external CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/external/regularisation.csv')

    # Convert DataFrame to array
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()

    # Create splits based on predefined random seed
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(x, y)

    # Generate scatter plot
    output_path = 'figures/regularization_scatter_plot.png'
    plt.scatter(x_train, y_train, s=5, label='Train')
    plt.scatter(x_val, y_val, s=5, label='Validation')
    plt.scatter(x_test, y_test, s=5, label='Test')
    plt.title('Regularization Data Split')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    plt.clf()
    print(output_path)


def visualize_spotify_dataset() -> None:
    """ Generate plots to show distributions of various features in the Spotify dataset. """

    # Read external CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/external/spotify.csv', index_col=0)

    # Remove duplicate rows for same track
    df.drop_duplicates(['track_id'], inplace=True)

    # Remove columns with nominal data and > 1000 values
    df.drop(columns=['track_id', 'artists', 'album_name', 'track_name'], inplace=True)

    # Convert boolean columns to integer datatype
    df['explicit'] = df['explicit'].astype(int)

    # Randomly sample 1000 points from DataFrame
    sampled_df = df.sample(n=1000, random_state=0)

    # Generate pairwise scatter plot
    output_path = 'figures/spotify_pair_plot.png'
    sns.pairplot(sampled_df, hue='track_genre', plot_kws={'s': 5})
    plt.title('Spotify Dataset Pair Plot')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    plt.clf()
    print(output_path)


if __name__ == '__main__':

    # 2 K-Nearest Neighbours

    ## 2.1 Music Genre Prediction
    preprocess_spotify_dataset()

    ## 2.2 Exploratory Data Analysis
    visualize_spotify_dataset()

    ## 2.4 Hyperparameter Tuning
    knn_hyperparameter_tuning()
    knn_k_values()
    knn_drop_columns()

    ## 2.5 Optimization
    knn_inference_time()

    ## 2.6 A Second Dataset
    knn_second_dataset()

    # 3 Linear Regression

    ## 3.1 Simple Regression
    visualize_regression_splits()

    ### 3.1.1 Degree 1
    simple_regression_line()

    ### 3.1.2 Degree > 1
    simple_regression_polynomial()

    ### 3.1.3 Animation
    simple_regression_animation()

    ## 3.2 Regularization
    visualize_regularization_splits()

    ### 3.2.1 Tasks
    regularization()
