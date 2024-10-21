""" Assignment 3: Multi Layer Perceptron and AutoEncoders. """

import ast
import json
import os
import shutil
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import wandb

# pylint: disable=wrong-import-position

PROJECT_DIR = '../..'
sys.path.insert(0, PROJECT_DIR)

from models.AutoEncoders import AutoEncoder
from models.knn import KNN
from models.MLP import MLP
from models.MLP.activation import get_activation
from models.MLP.loss import get_loss, BinaryCrossEntropy, CrossEntropy, MeanSquaredError
from performance_measures import ClassificationMeasures, MultiLabelClassificationMeasures, \
                                                                                RegressionMeasures

# pylint: enable=wrong-import-position

# Library customizations
pd.set_option('display.max_columns', None)
os.environ["WANDB_SILENT"] = "true"


def advertisement_data_preprocessing() -> None:
    """ Analyze and preprocess Housing Data Dataset. """

    # Log function call
    print('--- advertisement_data_preprocessing')

    # Read external CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/external/advertisement.csv')

    # Remove columns with less relevant values
    df.drop(columns=['city'], inplace=True)

    # Backup of target column
    labels_copy = df['labels'].copy()

    # Encode categorical values
    object_type_columns = df.select_dtypes(include=['object', 'bool']).columns
    for col in object_type_columns:
        df[col] = pd.factorize(df[col])[0]

    # Convert all columns to floating point
    df = df.astype(float)

    # Apply standardization to all columns
    df = (df - df.mean()) / df.std()

    # Restore labels
    df['labels'] = labels_copy

    # Encode labels
    all_labels = set(label for sublist in df['labels'].str.split() for label in sublist)
    label_to_index = {label: idx for idx, label in enumerate(all_labels)}
    df['labels'] = df['labels'].str.split().apply(lambda x: [label_to_index[label] for label in x])

    # Write processed DataFrame to CSV file
    df.to_csv(f'{PROJECT_DIR}/data/interim/3/advertisement.csv')
    print()


def housing_data_analysis_preprocessing() -> None:
    """ Analyze and preprocess Boston Housing Data Dataset. """

    # Log function call
    print('--- housing_data_analysis_preprocessing')

    # Read external CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/external/HousingData.csv')

    # Description of the dataset
    desc = df.describe()
    print(desc.loc[['mean', 'std', 'min', 'max']])

    # Plot the distribution of various attributes
    output_path = 'figures/boston_housing_distribution.png'
    df.hist(figsize=(12, 12), bins=15)
    plt.suptitle('Boston Housing Dataset: Distribution of Attributes')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    plt.clf()
    print(output_path)

    # Replace missing values with mean
    df.fillna(df.mean(), inplace=True)

    # Backup of target column
    medv_copy = df['MEDV'].copy()

    # Convert all columns to floating point
    df = df.astype(float)

    # Apply standardization to all columns
    df = (df - df.mean()) / df.std()

    # Revert back target column to original values
    df['MEDV'] = df['MEDV'].astype(int)
    df['MEDV'] = medv_copy

    # Write processed DataFrame to CSV file
    df.to_csv(f'{PROJECT_DIR}/data/interim/3/HousingData.csv')

    # Reinitialize the random number generator
    np.random.seed(0)

    # Shuffle the DataFrame
    df = df.sample(frac=1).reset_index(drop=True)

    # Compute the splitting indices
    train_end = int(0.8 * len(df))
    val_end = train_end + int(0.1 * len(df))

    # Split the DataFrame
    df_train, df_val, df_test = df[:train_end], df[train_end:val_end], df[val_end:]

    # Write processed DataFrames to CSV files
    df_train.to_csv(f'{PROJECT_DIR}/data/interim/3/HousingData_train.csv')
    df_val.to_csv(f'{PROJECT_DIR}/data/interim/3/HousingData_val.csv')
    df_test.to_csv(f'{PROJECT_DIR}/data/interim/3/HousingData_test.csv')
    print()


def knn_autoencoder() -> None:
    """ Find the nearest neighbour on the Spotify dataset reduced using AutoEncoder. """

    # Log function call
    print('--- knn_autoencoder')

    # Read interim CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/interim/1/spotify.csv', index_col=0)

    # Convert DataFrame to array
    X = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, -1].astype(int)

    # Split the array into train, validation and test
    X_train, X_val, _, y_train, y_val, _ = train_val_test_split(X, y)

    # Read the best hyperparameters from file
    with open(f'{PROJECT_DIR}/assignments/1/results/knn_hyper_params.txt', 'r', \
                                                                    encoding='utf-8') as file:
        k, metric, _ = file.readline().strip().split(', ')
        k = int(k)

    # Read the optimal number of dimensions from file
    with open(f'{PROJECT_DIR}/assignments/2/results/pca_spotify_optimal_dimensions.txt', 'r', \
                                                                    encoding='utf-8') as file:
        latent_dimension = file.readline().strip()
        latent_dimension = int(latent_dimension)

    # Fit the autoencoder
    autoenc = AutoEncoder(num_hidden_layers=4, activation=get_activation('tanh'), lr=1e-4, \
                                                                                optimizer='sgd')
    autoenc.fit(X_train, X_val, latent_dimension=latent_dimension)

    # Compute the latent representation
    X_train_latent = autoenc.get_latent(X_train)
    X_val_latent = autoenc.get_latent(X_val)

    # Initialize and train the model
    knn = KNN(k, metric)
    knn.fit(X_train_latent, y_train)

    # Compute predictions on the validation set
    y_pred = knn.predict(X_val_latent)

    # Evaluate predictions for the validation set
    cls_measures = ClassificationMeasures(y_val, y_pred)
    print('Accuracy:', cls_measures.accuracy_score())
    print('F1 Score:', cls_measures.f1_score(average='macro'))
    print('Precision:', cls_measures.precision_score(average='macro'))
    print('Recall:', cls_measures.recall_score(average='macro'))
    print()


def mlp_classification_hyperparameter_effects() -> None:
    """ Analyze the effects of hyperparameters in Multi Layer Perceptron Classification. """

    def train_worker():
        """ Trains a MLP with a given configuration. """

        # Initialize logging process
        wandb.init()

        # Initialize and train model
        mlp = MLP(
            num_hidden_layers=wandb.config.num_hidden_layers,
            num_neurons_per_layer=wandb.config.num_neurons_per_layer,
            activation=get_activation(wandb.config.activation),
            lr=wandb.config.lr,
            optimizer=wandb.config.optimizer,
            batch_size=wandb.config.batch_size,
            num_epochs=wandb.config.num_epochs,
            task='single-label-classification',
            loss=CrossEntropy()
        )
        mlp.fit(X_train, y_train, X_val, y_val, wandb_log=True)

    # Log function call
    print('--- mlp_classification_hyperparameter_effects')

    # Read the best hyperparameters from the results file
    with open(f'{PROJECT_DIR}/assignments/3/results/mlp_classification_hyperparameters.json', \
                                                                    'r', encoding='utf-8') as file:
        config = json.load(file)

    # Read interim CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/interim/3/WineQT.csv', index_col=0)

    # Convert DataFrame to array
    X = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, -1]

    # Start class labels from zero
    y -= np.min(y)
    y = y.astype(int)

    # Split the array into train, test and split
    X_train, X_val, _, y_train, y_val, _ = train_val_test_split(X, y)

    # WandB sweep configuration for task 1
    activation_sweep_config = {
        'name': 'hyperparameter-effects-activation',
        'method': 'grid',
        'metric': { 'name': 'val_acc', 'goal': 'maximize' },
        'parameters': {
            'activation': { 'values': ['identity', 'relu', 'sigmoid', 'tanh'] },
            'batch_size': { 'value': 16 },
            'lr': { 'value': config['lr'] },
            'num_epochs': { 'value': config['num_epochs'] },
            'num_hidden_layers': { 'value': config['num_hidden_layers'] },
            'num_neurons_per_layer': { 'value': config['num_neurons_per_layer'] },
            'optimizer': { 'value': config['optimizer'] }
        }
    }

    # WandB sweep configuration for task 2
    lr_sweep_config = {
        'name': 'hyperparameter-effects-lr',
        'method': 'grid',
        'metric': { 'name': 'val_acc', 'goal': 'maximize' },
        'parameters': {
            'activation': { 'value': config['activation'] },
            'batch_size': { 'value': 16 },
            'lr': { 'values': [1e-5, 1e-4, 1e-3, 1e-2] },
            'num_epochs': { 'value': config['num_epochs'] },
            'num_hidden_layers': { 'value': config['num_hidden_layers'] },
            'num_neurons_per_layer': { 'value': config['num_neurons_per_layer'] },
            'optimizer': { 'value': config['optimizer'] }
        }
    }

    # WandB sweep configuration for task 3
    batch_size_sweep_config = {
        'name': 'hyperparameter-effects-batch-size',
        'method': 'grid',
        'metric': { 'name': 'val_acc', 'goal': 'maximize' },
        'parameters': {
            'activation': { 'value': config['activation'] },
            'batch_size': { 'values': [8, 16, 32, 64] },
            'lr': { 'value': config['lr'] },
            'num_epochs': { 'value': config['num_epochs'] },
            'num_hidden_layers': { 'value': config['num_hidden_layers'] },
            'num_neurons_per_layer': { 'value': config['num_neurons_per_layer'] },
            'optimizer': { 'value': 'mini-batch' }
        }
    }

    # Start and finish sweep for task 1
    sweep_id = wandb.sweep(activation_sweep_config, project='smai-m24-mlp-classification')
    wandb.agent(sweep_id, train_worker)
    wandb.finish()

    # Start and finish sweep for task 2
    sweep_id = wandb.sweep(lr_sweep_config, project='smai-m24-mlp-classification')
    wandb.agent(sweep_id, train_worker)
    wandb.finish()

    # Start and finish sweep for task 3
    sweep_id = wandb.sweep(batch_size_sweep_config, project='smai-m24-mlp-classification')
    wandb.agent(sweep_id, train_worker)
    wandb.finish()

    shutil.rmtree('wandb')
    print()


def mlp_classification_best_model() -> None:
    """ Evaluate the best Multi Layer Perceptron Classification model, identified through
    hyperparameter tuning, on Wine Quality Test Dataset. """

    # Log function call
    print('--- mlp_classification_best_model')

    # Read the best hyperparameters from the results file
    with open(f'{PROJECT_DIR}/assignments/3/results/mlp_classification_hyperparameters.json', \
                                                                    'r', encoding='utf-8') as file:
        config = json.load(file)

    # Read interim CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/interim/3/WineQT.csv', index_col=0)

    # Convert DataFrame to array
    X = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, -1]

    # Start class labels from zero
    y -= np.min(y)
    y = y.astype(int)

    # Split the array into train, test and split
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    # Initialize and train model
    mlp = MLP(
        num_hidden_layers=config['num_hidden_layers'],
        num_neurons_per_layer=config['num_neurons_per_layer'],
        activation=get_activation(config['activation']),
        lr=config['lr'],
        num_epochs=config['num_epochs'],
        optimizer=config['optimizer'],
        task='single-label-classification',
        loss=CrossEntropy()
    )
    mlp.fit(X_train, y_train, X_val, y_val)

    # Evaluate metrics for model
    test_measures = ClassificationMeasures(y_test, mlp.predict(X_test))
    print('Accuracy:', test_measures.accuracy_score())
    print('F1 Score:', test_measures.f1_score(average='macro'))
    print('Precision:', test_measures.precision_score(average='macro'))
    print('Recall:', test_measures.recall_score(average='macro'))
    print()


def mlp_classification_hyperparameter_tuning() -> None:
    """ Hyperparameter tuning for Multi Layer Perceptron Classification on Wine Quality Dataset. """

    def train_worker():
        """ Trains a MLP with a given configuration. """

        # Initialize logging process
        wandb.init()

        # Initialize and train model
        mlp = MLP(
            num_hidden_layers=wandb.config.num_hidden_layers,
            num_neurons_per_layer=wandb.config.num_neurons_per_layer,
            activation=get_activation(wandb.config.activation),
            lr=wandb.config.lr,
            num_epochs=wandb.config.num_epochs,
            optimizer=wandb.config.optimizer,
            task='single-label-classification',
            loss=CrossEntropy()
        )
        mlp.fit(X_train, y_train, X_val, y_val, wandb_log=True)

        # Evaluate metrics for model
        train_measures = ClassificationMeasures(y_train, mlp.predict(X_train))
        val_measures = ClassificationMeasures(y_val, mlp.predict(X_val))

        # Log metrics
        wandb.log({
            'train_acc': train_measures.accuracy_score(),
            'train_f1': train_measures.f1_score(average='macro'),
            'train_precision': train_measures.precision_score(average='macro'),
            'train_recall': train_measures.recall_score(average='macro'),
            'val_acc': val_measures.accuracy_score(),
            'val_f1': val_measures.f1_score(average='macro'),
            'val_precision': val_measures.precision_score(average='macro'),
            'val_recall': val_measures.recall_score(average='macro')
        })

    # Log function call
    print('--- mlp_classification_hyperparameter_tuning')

    # Read interim CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/interim/3/WineQT.csv', index_col=0)

    # Convert DataFrame to array
    X = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, -1]

    # Start class labels from zero
    y -= np.min(y)
    y = y.astype(int)

    # Split the array into train, test and split
    X_train, X_val, _, y_train, y_val, _ = train_val_test_split(X, y)

    # WandB sweep configuration
    sweep_config = {
        'name': 'hyperparameter-tuning',
        'method': 'grid',
        'metric': { 'name': 'val_acc', 'goal': 'maximize' },
        'parameters': {
            'activation': { 'values': ['identity', 'relu', 'sigmoid', 'tanh'] },
            'lr': { 'values': [1e-5, 1e-4] },
            'num_epochs': { 'values': [50, 100] },
            'num_hidden_layers': { 'values': [4, 8, 16] },
            'num_neurons_per_layer': { 'values': [4, 8, 16] },
            'optimizer': { 'values': ['sgd', 'batch', 'mini-batch'] }
        }
    }

    # Start and finish WandB sweep
    sweep_id = wandb.sweep(sweep_config, project='smai-m24-mlp-classification')
    wandb.agent(sweep_id, train_worker)
    wandb.finish()

    shutil.rmtree('wandb')
    print()


def mlp_classification_spotify_dataset() -> None:
    """ Multi Layer Perceptron Classification on the Spotify dataset. """

    # Log function call
    print('--- mlp_classification_spotify_dataset')

    # Read the best hyperparameters from the results file
    with open( \
        f'{PROJECT_DIR}/assignments/3/results/mlp_classification_spotify_hyperparameters.json', \
                                                                    'r', encoding='utf-8') as file:
        config = json.load(file)

    # Read interim CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/interim/1/spotify.csv', index_col=0)

    # Convert DataFrame to array
    X = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, -1].astype(int)

    # Split the array into train, validation and test
    X_train, X_val, _, y_train, y_val, _ = train_val_test_split(X, y)

    # Initialize and train model
    mlp = MLP(
        num_hidden_layers=config['num_hidden_layers'],
        num_neurons_per_layer=config['num_neurons_per_layer'],
        activation=get_activation(config['activation']),
        lr=config['lr'],
        optimizer=config['optimizer'],
        task='single-label-classification',
        loss=CrossEntropy()
    )
    mlp.fit(X_train, y_train, X_val, y_val)

    # Evaluate metrics for model
    cls_measures = ClassificationMeasures(y_val, mlp.predict(X_val))
    print('Accuracy:', cls_measures.accuracy_score())
    print('F1 Score:', cls_measures.f1_score(average='macro'))
    print('Precision:', cls_measures.precision_score(average='macro'))
    print('Recall:', cls_measures.recall_score(average='macro'))
    print()


def mlp_logistic_regression() -> None:
    """ Apply Logistic Regression using MLP, on the Pima Indians Diabetes dataset. """

    def train_worker():
        """ Trains a MLP with a given configuration. """

        # Initialize logging process
        wandb.init()

        # Initialize and train model
        mlp = MLP(
            num_hidden_layers=0,
            num_neurons_per_layer=[],
            activation=get_activation('sigmoid'),
            lr=1e-4,
            num_epochs=150,
            task='regression',
            loss=get_loss(wandb.config.loss)
        )
        mlp.fit(X_train, y_train, X_val, y_val, wandb_log=True)

    # Log function call
    print('--- mlp_logistic_regression')

    # Read interim CSVs into DataFrames
    df = pd.read_csv(f'{PROJECT_DIR}/data/external/diabetes.csv')

    # Convert DataFrames to arrays
    X = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, -1]

    # Split the array into train, test and split
    X_train, X_val, _, y_train, y_val, _ = train_val_test_split(X, y)

    # WandB sweep configuration
    sweep_config = {
        'name': 'binary-classification',
        'method': 'grid',
        'metric': { 'name': 'val_loss', 'goal': 'minimize' },
        'parameters': {
            'loss': { 'values': ['binary-cross-entropy', 'mean-squared-error'] }
        }
    }

    # Start and finish sweep
    sweep_id = wandb.sweep(sweep_config, project='smai-m24-mlp-regression')
    wandb.agent(sweep_id, train_worker)
    wandb.finish()

    shutil.rmtree('wandb')
    print()


def mlp_multi_label_classification_hyperparameter_tuning() -> None:
    """ Hyperparameter tuning for Multi Layer Perceptron Classification,
    on multi-label Advertisement Dataset. """

    def train_worker():
        """ Trains a MLP with a given configuration. """

        # Initialize logging process
        wandb.init()

        # Initialize and train model
        mlp = MLP(
            num_hidden_layers=wandb.config.num_hidden_layers,
            num_neurons_per_layer=wandb.config.num_neurons_per_layer,
            activation=get_activation(wandb.config.activation),
            lr=wandb.config.lr,
            num_epochs=100,
            optimizer='sgd',
            task='multi-label-classification',
            loss=BinaryCrossEntropy()
        )
        mlp.fit(X_train, y_train, X_val, y_val, wandb_log=True)

        # Evaluate metrics for model
        train_measures = MultiLabelClassificationMeasures(y_train, mlp.predict(X_train))
        val_measures = MultiLabelClassificationMeasures(y_val, mlp.predict(X_val))

        # Log metrics
        wandb.log({
            'train_acc': train_measures.accuracy_score(),
            'train_f1': train_measures.f1_score(),
            'train_hamming': train_measures.hamming_distance(),
            'train_precision': train_measures.precision_score(),
            'train_recall': train_measures.recall_score(),
            'val_acc': val_measures.accuracy_score(),
            'val_f1': val_measures.f1_score(),
            'val_hamming': val_measures.hamming_distance(),
            'val_precision': val_measures.precision_score(),
            'val_recall': val_measures.recall_score()
        })

    # Log function call
    print('--- mlp_multi_label_classification_hyperparameter_tuning')

    # Read interim CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/interim/3/advertisement.csv', index_col=0)

    # Convert DataFrame to array
    X = df.to_numpy()[:, :-1].astype(float)
    y = df.to_numpy()[:, -1]
    for idx in range(y.shape[0]):
        y[idx] = ast.literal_eval(y[idx])

    # Split the array into train, test and split
    X_train, X_val, _, y_train, y_val, _ = train_val_test_split(X, y)

    # WandB sweep configuration
    sweep_config = {
        'name': 'hyperparameter-tuning-multi-label',
        'method': 'grid',
        'metric': { 'name': 'val_acc', 'goal': 'maximize' },
        'parameters': {
            'activation': { 'values': ['identity', 'relu', 'sigmoid', 'tanh'] },
            'lr': { 'values': [1e-5, 1e-4] },
            'num_hidden_layers': { 'values': [4, 8, 16] },
            'num_neurons_per_layer': { 'values': [4, 8, 16] }
        }
    }

    # Start and finish WandB sweep
    sweep_id = wandb.sweep(sweep_config, project='smai-m24-mlp-classification')
    wandb.agent(sweep_id, train_worker)
    wandb.finish()

    shutil.rmtree('wandb')
    print()


def mlp_multi_label_classification_best_model() -> None:
    """ Evaluate the best Multi Layer Perceptron Classification model, identified through
    hyperparameter tuning, on Advertisement Dataset. """

    # Log function call
    print('--- mlp_multi_label_classification_best_model')

    # Read the best hyperparameters from the results file
    with open(\
        f'{PROJECT_DIR}/assignments/3/results/mlp_classification_multilabel_hyperparameters.json',\
                                                                    'r', encoding='utf-8') as file:
        config = json.load(file)

    # Read interim CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/interim/3/advertisement.csv', index_col=0)

    # Convert DataFrame to array
    X = df.to_numpy()[:, :-1].astype(float)
    y = df.to_numpy()[:, -1]
    for idx in range(y.shape[0]):
        y[idx] = ast.literal_eval(y[idx])

    # Split the array into train, test and split
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    # Initialize and train model
    mlp = MLP(
        num_hidden_layers=config['num_hidden_layers'],
        num_neurons_per_layer=config['num_neurons_per_layer'],
        activation=get_activation(config['activation']),
        lr=config['lr'],
        num_epochs=config['num_epochs'],
        optimizer=config['optimizer'],
        task='multi-label-classification',
        loss=BinaryCrossEntropy()
    )
    mlp.fit(X_train, y_train, X_val, y_val)

    # Evaluate metrics for model
    test_measures = MultiLabelClassificationMeasures(y_test, mlp.predict(X_test))
    test_measures.print_all_measures()
    print()


def mlp_regression_best_model() -> None:
    """ Evaluate the best Multi Layer Perceptron Regression model, identified through
    hyperparameter tuning, on Boston Housing Dataset. """

    # Log function call
    print('--- mlp_regression_best_model')

    # Read the best hyperparameters from the results file
    with open(f'{PROJECT_DIR}/assignments/3/results/mlp_regression_hyperparameters.json', \
                                                                    'r', encoding='utf-8') as file:
        config = json.load(file)

    # Read interim CSVs into DataFrames
    df_train = pd.read_csv(f'{PROJECT_DIR}/data/interim/3/HousingData_train.csv', index_col=0)
    df_val = pd.read_csv(f'{PROJECT_DIR}/data/interim/3/HousingData_val.csv', index_col=0)
    df_test = pd.read_csv(f'{PROJECT_DIR}/data/interim/3/HousingData_test.csv', index_col=0)

    # Convert DataFrames to arrays
    X_train, y_train = df_train.to_numpy()[:, :-1], df_train.to_numpy()[:, -1]
    X_val, y_val = df_val.to_numpy()[:, :-1], df_val.to_numpy()[:, -1]
    X_test, y_test = df_test.to_numpy()[:, :-1], df_test.to_numpy()[:, -1]

    # Initialize and train model
    mlp = MLP(
        num_hidden_layers=config['num_hidden_layers'],
        num_neurons_per_layer=config['num_neurons_per_layer'],
        activation=get_activation(config['activation']),
        lr=config['lr'],
        num_epochs=config['num_epochs'],
        optimizer=config['optimizer'],
        task='regression',
        loss=MeanSquaredError()
    )
    mlp.fit(X_train, y_train, X_val, y_val)

    # Evaluate metrics for model
    test_measures = RegressionMeasures(y_test, mlp.predict(X_test))
    test_measures.print_all_measures()
    print()


def mlp_regression_hyperparameter_tuning() -> None:
    """ Hyperparameter tuning for Multi Layer Perceptron Regression on Boston Housing Dataset. """

    def train_worker():
        """ Trains a MLP with a given configuration. """

        # Initialize logging process
        wandb.init()

        # Initialize and train model
        mlp = MLP(
            num_hidden_layers=wandb.config.num_hidden_layers,
            num_neurons_per_layer=wandb.config.num_neurons_per_layer,
            activation=get_activation(wandb.config.activation),
            lr=wandb.config.lr,
            num_epochs=wandb.config.num_epochs,
            optimizer=wandb.config.optimizer,
            task='regression',
            loss=MeanSquaredError()
        )
        mlp.fit(X_train, y_train, X_val, y_val, wandb_log=True)

        # Evaluate metrics for model
        train_measures = RegressionMeasures(y_train, mlp.predict(X_train))
        val_measures = RegressionMeasures(y_val, mlp.predict(X_val))

        # Log metrics
        wandb.log({
            'train_mae': train_measures.mean_absolute_error(),
            'train_mse': train_measures.mean_squared_error(),
            'train_rmse': train_measures.root_mean_squared_error(),
            'train_r2': train_measures.r_squared(),
            'val_mae': val_measures.mean_absolute_error(),
            'val_mse': val_measures.mean_squared_error(),
            'val_rmse': val_measures.root_mean_squared_error(),
            'val_r2': val_measures.r_squared(),
        })

    # Log function call
    print('--- mlp_regression_hyperparameter_tuning')

    # Read interim CSVs into DataFrames
    df_train = pd.read_csv(f'{PROJECT_DIR}/data/interim/3/HousingData_train.csv', index_col=0)
    df_val = pd.read_csv(f'{PROJECT_DIR}/data/interim/3/HousingData_val.csv', index_col=0)

    # Convert DataFrames to arrays
    X_train, y_train = df_train.to_numpy()[:, :-1], df_train.to_numpy()[:, -1]
    X_val, y_val = df_val.to_numpy()[:, :-1], df_val.to_numpy()[:, -1]

    # WandB sweep configuration
    sweep_config = {
        'name': 'hyperparameter-tuning',
        'method': 'grid',
        'metric': { 'name': 'val_loss', 'goal': 'minimize' },
        'parameters': {
            'activation': { 'values': ['identity', 'relu', 'sigmoid', 'tanh'] },
            'lr': { 'values': [1e-5, 1e-4] },
            'num_epochs': { 'values': [50, 100] },
            'num_hidden_layers': { 'values': [4, 8, 16] },
            'num_neurons_per_layer': { 'values': [4, 8, 16] },
            'optimizer': { 'values': ['sgd', 'batch', 'mini-batch'] }
        }
    }

    # Start and finish WandB sweep
    sweep_id = wandb.sweep(sweep_config, project='smai-m24-mlp-regression')
    wandb.agent(sweep_id, train_worker)
    wandb.finish()

    shutil.rmtree('wandb')
    print()


def wineqt_analysis_preprocessing() -> None:
    """ Analyze and preprocess Wine Quality Dataset. """

    # Log function call
    print('--- wineqt_analysis_preprocessing')

    # Read external CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/external/WineQT.csv', index_col=-1)

    # Description of the dataset
    desc = df.describe()
    print(desc.loc[['mean', 'std', 'min', 'max']])

    # Plot the distribution of various attributes
    output_path = 'figures/wineqt_distribution.png'
    df.hist(figsize=(12, 12), bins=15)
    plt.suptitle('Wine Quality Dataset: Distribution of Attributes')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    plt.clf()
    print(output_path)

    # Backup of discrete labels
    quality_copy = df['quality'].copy()

    # Convert all columns to floating point
    df = df.astype(float)

    # Apply standardization to all columns
    df = (df - df.mean()) / df.std()

    # Revert back labels to original discrete values
    df['quality'] = df['quality'].astype(int)
    df['quality'] = quality_copy

    # Write processed DataFrame to CSV file
    df.to_csv(f'{PROJECT_DIR}/data/interim/3/WineQT.csv')
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

    # 2 Multi Layer Perceptron Classification

    ## 2.1 Dataset Analysis and Preprocessing
    wineqt_analysis_preprocessing()

    ## 2.3 Model Training & Hyperparameter Tuning
    # mlp_classification_hyperparameter_tuning()

    ## 2.4 Evaluating Single-label Classification Model
    mlp_classification_best_model()

    ## 2.5 Analyzing Hyperparameters Effects
    # mlp_classification_hyperparameter_effects()

    ## 2.6 Multi-Label Classification
    advertisement_data_preprocessing()
    # mlp_multi_label_classification_hyperparameter_tuning()
    mlp_multi_label_classification_best_model()

    # 3 Multi Layer Perceptron Regression

    ## 3.1 Data Preprocessing
    housing_data_analysis_preprocessing()

    ## 3.3 Model Training & Hyperparameter Tuning
    # mlp_regression_hyperparameter_tuning()

    ## 3.4 Evaluating Model
    mlp_regression_best_model()

    ## 3.5 Mean Squared Error vs Binary Cross Entropy
    # mlp_logistic_regression()

    # 4 AutoEncoders

    ## 4.3 AutoEncoder + KNN
    knn_autoencoder()

    ## 4.4 MLP Classification
    mlp_classification_spotify_dataset()
