""" Assignment 2: KMeans, Gaussian Mixture Models and Principal Component Analysis. """

import sys
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.cluster.hierarchy as hc

# pylint: disable=wrong-import-position

PROJECT_DIR = '../..'
sys.path.insert(0, PROJECT_DIR)

from models.gmm import GMM
from models.k_means import KMeans
from models.knn import KNN
from models.pca import PCA
from performance_measures import ClassificationMeasures, ClusteringMeasures

# pylint: enable=wrong-import-position


def gmm_2d_visualization() -> None:
    """ Perform GMM using the number of clusters estimated from 2D visualization. """

    # Log function call
    print('--- gmm_2d_visualization')

    # Read the external data into a DataFrame
    df = pd.read_feather(f'{PROJECT_DIR}/data/external/word-embeddings.feather')

    # Extract the words
    y = df.to_numpy()[:, 0]

    # Extract the 512 length embeddings
    X = df.to_numpy()[:, 1]
    X = np.vstack(X)

    # Read the estimate for number of clusters from the 2D visualization
    with open('results/k_2.txt', 'r', encoding='utf-8') as file:
        k = file.readline().strip()
        k = int(k)

    # Perform clustering based on the estimate
    gmm = GMM(k).fit(X)
    y_pred = gmm.getMembership(X)
    likelihood = gmm.getLikelihood()
    cls_measures = ClusteringMeasures(k, X.shape[0], likelihood)
    print(k, likelihood, cls_measures.aic(), cls_measures.bic())

    # Print the clusters
    for cluster in np.unique(y_pred):
        print(y[y_pred == cluster])
    print()


def gmm_dimensionality_reduction() -> None:
    """ Find the optimal number of clusters for the dataset reduced using Scree Plot. """

    # Log function call
    print('--- gmm_dimensionality_reduction')

    # Read the external data into a DataFrame
    df = pd.read_feather(f'{PROJECT_DIR}/data/external/word-embeddings.feather')

    # Extract the words
    y = df.to_numpy()[:, 0]

    # Extract the 512 length embeddings
    X = df.to_numpy()[:, 1]
    X = np.vstack(X)

    # Read the optimal number of dimensions determined manually
    with open('results/pca_wordemb_optimal_dimensions.txt', 'r', encoding='utf-8') as file:
        n_components = file.readline().strip()
        n_components = int(n_components)

    # Perform dimensionality reduction based on the optimal number of dimensions
    pca = PCA(n_components=n_components).fit(X)
    assert pca.checkPCA()
    X_reduced = pca.transform(X)

    # --- Determine the optimal number of clusters for the reduced dataset ---

    # Set of hyperparameters
    k_list = range(1, 16)

    # List to store measures for different values of hyperparameter
    likelihoods = []
    aic_list = []
    bic_list = []

    # Iterate over all hyperparameters
    for k in k_list:

        # Initialize and fit the model
        gmm = GMM(k).fit(X_reduced)

        # Compute the likelihood
        likelihoods.append(gmm.getLikelihood())

        # Compute AIC and BIC
        cls_measures = ClusteringMeasures(k, X_reduced.shape[0], likelihoods[-1])
        aic_list.append(cls_measures.aic())
        bic_list.append(cls_measures.bic())

        # Print all computed measures
        print(k, likelihoods[-1], aic_list[-1], bic_list[-1])

    # Plot the relation between hyperparameter and likelihood
    plt.plot(k_list, likelihoods)
    plt.title('PCA + GMM: k vs Log Likelihood')
    plt.ylabel('Log Likelihood')
    plt.xlabel('Number of components (k)')
    plt.grid()
    plt.savefig('figures/pca_gmm_log_likelihood.png', bbox_inches='tight')
    plt.close()
    plt.clf()
    print('figures/pca_gmm_log_likelihood.png')

    # Plot the relation between hyperparameter and AIC
    plt.plot(k_list, aic_list)
    plt.title('PCA + GMM: k vs AIC')
    plt.ylabel('Akaike Information Criterion')
    plt.xlabel('Number of components (k)')
    plt.grid()
    plt.savefig('figures/pca_gmm_aic.png', bbox_inches='tight')
    plt.close()
    plt.clf()
    print('figures/pca_gmm_aic.png')

    # Plot the relation between hyperparameter and BIC
    plt.plot(k_list, bic_list)
    plt.title('PCA + GMM: k vs BIC')
    plt.ylabel('Bayesian Information Criterion')
    plt.xlabel('Number of components (k)')
    plt.grid()
    plt.savefig('figures/pca_gmm_bic.png', bbox_inches='tight')
    plt.close()
    plt.clf()
    print('figures/pca_gmm_bic.png')
    print()

    # Read the optimal number of clusters determined manually
    with open('results/k_gmm3.txt', 'r', encoding='utf-8') as file:
        k, _, _, _ = file.readline().strip().split(', ')
        k = int(k)

    # Perform clustering based on optimal number of clusters
    gmm = GMM(k).fit(X_reduced)
    y_pred = gmm.getMembership(X_reduced)
    likelihood = gmm.getLikelihood()
    cls_measures = ClusteringMeasures(k, X_reduced.shape[0], likelihood)
    print('Optimal number of clusters:', k, likelihood, cls_measures.aic(), cls_measures.bic())

    # Print the clusters
    for cluster in np.unique(y_pred):
        print(y[y_pred == cluster])
    print()


def gmm_optimal_num_clusters() -> None:
    """ Find the optimal number of clusters for Gaussian Mixture Model using
    Bayesian Information Criterion and Akaike Information Criterion. """

    # Log function call
    print('--- gmm_optimal_num_clusters')

    # Set of hyperparameters
    k_list = range(2, 11)

    # Read the external data into a DataFrame
    df = pd.read_feather(f'{PROJECT_DIR}/data/external/word-embeddings.feather')

    # Extract the words
    y = df.to_numpy()[:, 0]

    # Extract the 512 length embeddings
    X = df.to_numpy()[:, 1]
    X = np.vstack(X)

    # List to store measures for different values of hyperparameter
    likelihoods = []
    aic_list = []
    bic_list = []

    # Iterate over all hyperparameters
    for k in k_list:

        # Initialize and fit the model
        gmm = GMM(k).fit(X)

        # Compute the likelihood
        likelihoods.append(gmm.getLikelihood())

        # Compute AIC and BIC
        cls_measures = ClusteringMeasures(k, X.shape[0], likelihoods[-1])
        aic_list.append(cls_measures.aic())
        bic_list.append(cls_measures.bic())

        # Print all computed measures
        print(k, likelihoods[-1], aic_list[-1], bic_list[-1])

    # Plot the relation between hyperparameter and likelihood
    plt.plot(k_list, likelihoods)
    plt.title('GMM: k vs Log Likelihood')
    plt.ylabel('Log Likelihood')
    plt.xlabel('Number of components (k)')
    plt.grid()
    plt.savefig('figures/gmm_log_likelihood.png', bbox_inches='tight')
    plt.close()
    plt.clf()
    print('figures/gmm_log_likelihood.png')

    # Plot the relation between hyperparameter and AIC
    plt.plot(k_list, aic_list)
    plt.title('GMM: k vs AIC')
    plt.ylabel('Akaike Information Criterion')
    plt.xlabel('Number of components (k)')
    plt.grid()
    plt.savefig('figures/gmm_aic.png', bbox_inches='tight')
    plt.close()
    plt.clf()
    print('figures/gmm_aic.png')

    # Plot the relation between hyperparameter and BIC
    plt.plot(k_list, bic_list)
    plt.title('GMM: k vs BIC')
    plt.ylabel('Bayesian Information Criterion')
    plt.xlabel('Number of components (k)')
    plt.grid()
    plt.savefig('figures/gmm_bic.png', bbox_inches='tight')
    plt.close()
    plt.clf()
    print('figures/gmm_bic.png')
    print()

    # Read the optimal number of clusters determined manually
    with open('results/k_gmm1.txt', 'r', encoding='utf-8') as file:
        k, _, _, _ = file.readline().strip().split(', ')
        k = int(k)

    # Perform clustering based on optimal number of clusters
    gmm = GMM(k).fit(X)
    y_pred = gmm.getMembership(X)
    likelihood = gmm.getLikelihood()
    cls_measures = ClusteringMeasures(k, X.shape[0], likelihood)
    print('Optimal number of clusters:', k, likelihood, cls_measures.aic(), cls_measures.bic())

    # Print the clusters
    for cluster in np.unique(y_pred):
        print(y[y_pred == cluster])
    print()


def hierarchical_clustering() -> None:
    """ Hierarchical clustering, with different linkage methods and distance metrics. """

    # Log function call
    print('--- hierarchical_clustering')

    # Read the external data into a DataFrame
    df = pd.read_feather(f'{PROJECT_DIR}/data/external/word-embeddings.feather')

    # Extract the words
    y = df.to_numpy()[:, 0]

    # Extract the 512 length embeddings
    X = df.to_numpy()[:, 1]
    X = np.vstack(X)

    # Different combinations of linkage methods
    linkage_list = [
        'single',
        'complete',
        'average',
        'weighted',
        'centroid',
        'median',
        'ward'
    ]
    metric_list = ['cityblock', 'euclidean', 'cosine']

    # Iterate over all methods
    for linkage in linkage_list:
        for metric in metric_list:
            try:
                linkage_matrix = hc.linkage(X, method=linkage, metric=metric)
                plt.figure(figsize=(25, 10))
                hc.dendrogram(linkage_matrix)
                plt.savefig(f'figures/hierarchical_{linkage}_linkage_{metric}.png')
                plt.close()
                plt.clf()
                print(f'figures/hierarchical_{linkage}_linkage_{metric}.png')
            except ValueError:
                print(f'Skipping {linkage} linkage for {metric} distance')
    print()

    # Read the best linkage method as per analysis
    with open('results/hierarchical_best_linkage_method.txt', 'r', encoding='utf-8') as file:
        linkage = file.readline().strip()

    # Read the best k from K-Means as per analysis
    with open('results/k_means.txt', 'r', encoding='utf-8') as file:
        k_best1 = file.readline().strip()
        k_best1 = int(k_best1)

    # Cut the dendogram at the points corresponding to k from K-Means
    linkage_matrix = hc.linkage(X, method=linkage, metric='euclidean')
    y_pred = hc.fcluster(linkage_matrix, k_best1, criterion='maxclust')

    # Print the clusters
    print(k_best1, linkage)
    for cluster in np.unique(y_pred):
        print(y[y_pred == cluster])
    print()

    # Read the best k from GMM as per analysis
    with open('results/k_means.txt', 'r', encoding='utf-8') as file:
        k_best2 = file.readline().strip()
        k_best2 = int(k_best2)

    # Cut the dendogram at the points corresponding to k from K-Means
    linkage_matrix = hc.linkage(X, method=linkage, metric='euclidean')
    y_pred = hc.fcluster(linkage_matrix, k_best2, criterion='maxclust')

    # Print the clusters
    print(k_best2, linkage)
    for cluster in np.unique(y_pred):
        print(y[y_pred == cluster])

    print()


def kmeans_2d_visualization() -> None:
    """ Perform KMeans using the number of clusters estimated from 2D visualization. """

    # Log function call
    print('--- kmeans_2d_visualization')

    # Read the external data into a DataFrame
    df = pd.read_feather(f'{PROJECT_DIR}/data/external/word-embeddings.feather')

    # Extract the words
    y = df.to_numpy()[:, 0]

    # Extract the 512 length embeddings
    X = df.to_numpy()[:, 1]
    X = np.vstack(X)

    # Read the estimate for number of clusters from the 2D visualization
    with open('results/k_2.txt', 'r', encoding='utf-8') as file:
        k = file.readline().strip()
        k = int(k)

    # Perform clustering based on the estimate
    kmeans = KMeans(k).fit(X)
    y_pred = kmeans.predict(X)
    cost = kmeans.getCost()
    print(k, cost)

    # Print the clusters
    for cluster in np.unique(y_pred):
        print(y[y_pred == cluster])
    print()


def kmeans_dimensionality_reduction() -> None:
    """ Find the optimal number of clusters for the dataset reduced based on Scree Plot. """

    # Log function call
    print('--- kmeans_dimensionality_reduction')

    # Read the external data into a DataFrame
    df = pd.read_feather(f'{PROJECT_DIR}/data/external/word-embeddings.feather')

    # Extract the words
    y = df.to_numpy()[:, 0]

    # Extract the 512 length embeddings
    X = df.to_numpy()[:, 1]
    X = np.vstack(X)

    # Compute the eigenvalues of the covariance matrix
    X_centered = X - np.mean(X, axis=0)
    covariance = X_centered.T @ X_centered / (X.shape[0] - 1)
    eigenvalues, _ = np.linalg.eig(covariance)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Generate scree plot
    plt.figure(figsize=(15,6))
    plt.plot(range(1, eigenvalues.shape[0] + 1), eigenvalues)
    plt.title('PCA (Word Embeddings): Scree Plot')
    plt.xlabel('Component Number')
    plt.ylabel('Eigenvalue')
    plt.xticks(range(1, 513, 15))
    plt.grid()
    plt.savefig('figures/pca_wordemb_scree_plot.png', bbox_inches='tight')
    plt.close()
    plt.clf()
    print('figures/pca_wordemb_scree_plot.png')

    # Read the optimal number of dimensions determined manually
    with open('results/pca_wordemb_optimal_dimensions.txt', 'r', encoding='utf-8') as file:
        n_components = file.readline().strip()
        n_components = int(n_components)

    # Perform dimensionality reduction based on the optimal number of dimensions
    pca = PCA(n_components=n_components).fit(X)
    assert pca.checkPCA()
    X_reduced = pca.transform(X)

    # --- Determine the optimal number of clusters for the reduced dataset ---

    # Set of hyperparameters
    k_list = range(1, 21)

    # Initialize empty list to store costs for different values of hyperparameter
    costs = []

    # Iterate over all hyperparameters
    for k in k_list:
        kmeans = KMeans(k).fit(X_reduced)
        cost = kmeans.getCost()
        costs.append(cost)
        print(k, cost)

    # Plot the relation between hyperparameter and cost
    plt.figure(figsize=(10,6))
    plt.plot(k_list, costs)
    plt.title('PCA + K-Means: k vs WCSS')
    plt.ylabel('Within Cluster Sum of Squares (WCSS)')
    plt.xlabel('Number of clusters (k)')
    plt.xticks(range(1, 21, 2))
    plt.grid()
    plt.savefig('figures/pca_kmeans_wcss.png', bbox_inches='tight')
    plt.close()
    plt.clf()
    print('figures/pca_kmeans_wcss.png')
    print()

    # Read the elbow point determined manually
    with open('results/k_means3.txt', 'r', encoding='utf-8') as file:
        k, _ = file.readline().strip().split(', ')
        k = int(k)

    # Perform clustering based on elbow point
    kmeans = KMeans(k).fit(X_reduced)
    y_pred = kmeans.predict(X_reduced)
    cost = kmeans.getCost()
    print('Elbow point:', k, cost)

    # Print the clusters
    for cluster in np.unique(y_pred):
        print(y[y_pred == cluster])
    print()


def kmeans_optimal_num_clusters() -> None:
    """ Find the optimal number of clusters for KMeans using Elbow Method. """

    # Log function call
    print('--- kmeans_optimal_num_clusters')

    # Set of hyperparameters
    k_list = range(1, 21)

    # Read the external data into a DataFrame
    df = pd.read_feather(f'{PROJECT_DIR}/data/external/word-embeddings.feather')

    # Extract the words
    y = df.to_numpy()[:, 0]

    # Extract the 512 length embeddings
    X = df.to_numpy()[:, 1]
    X = np.vstack(X)

    # Initialize empty list to store costs for different values of hyperparameter
    costs = []

    # Iterate over all hyperparameters
    for k in k_list:
        kmeans = KMeans(k).fit(X)
        cost = kmeans.getCost()
        costs.append(cost)
        print(k, cost)

    # Plot the relation between hyperparameter and cost
    plt.figure(figsize=(10,6))
    plt.plot(k_list, costs)
    plt.title('K-Means: k vs WCSS')
    plt.ylabel('Within Cluster Sum of Squares (WCSS)')
    plt.xlabel('Number of clusters (k)')
    plt.xticks(range(1, 21, 2))
    plt.grid()
    plt.savefig('figures/kmeans_wcss.png', bbox_inches='tight')
    plt.close()
    plt.clf()
    print('figures/kmeans_wcss.png')
    print()

    # Read the elbow point determined manually
    with open('results/k_means1.txt', 'r', encoding='utf-8') as file:
        k, _ = file.readline().strip().split(', ')
        k = int(k)

    # Perform clustering based on elbow point
    kmeans = KMeans(k).fit(X)
    y_pred = kmeans.predict(X)
    cost = kmeans.getCost()
    print('Elbow point:', k, cost)

    # Print the clusters
    for cluster in np.unique(y_pred):
        print(y[y_pred == cluster])
    print()


def nearest_neighbour_search() -> None:
    """ Find the nearest neighbour on the dataset reduced using PCA. """

    # Log function call
    print('--- nearest_neighbour_search')

    # Read interim CSV into DataFrame
    df = pd.read_csv(f'{PROJECT_DIR}/data/interim/1/spotify.csv', index_col=0)

    # Convert DataFrame to array
    X = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, -1].astype(int)

    # Compute the eigenvalues of the covariance matrix
    X_centered = X - np.mean(X, axis=0)
    covariance = X_centered.T @ X_centered / (X.shape[0] - 1)
    eigenvalues, _ = np.linalg.eig(covariance)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Generate scree plot
    plt.figure(figsize=(15,6))
    plt.plot(range(1, eigenvalues.shape[0] + 1), eigenvalues)
    plt.title('PCA (Spotify): Scree Plot')
    plt.xlabel('Component Number')
    plt.ylabel('Eigenvalue')
    plt.xticks(range(1, 17))
    plt.grid()
    plt.savefig('figures/pca_spotify_scree_plot.png', bbox_inches='tight')
    plt.close()
    plt.clf()
    print('figures/pca_spotify_scree_plot.png')
    print()

    # Read the optimal number of dimensions determined manually
    with open('results/pca_spotify_optimal_dimensions.txt', 'r', encoding='utf-8') as file:
        n_components = file.readline().strip()
        n_components = int(n_components)

    # Perform dimensionality reduction based on the optimal number of dimensions
    pca = PCA(n_components=n_components).fit(X)
    assert pca.checkPCA()
    X_reduced = pca.transform(X)

    # --- Apply K-Nearest Neighbours on the original dataset ---

    # Read the best hyperparameters from file
    with open(f'{PROJECT_DIR}/assignments/1/results/knn_hyper_params.txt', 'r', encoding='utf-8') \
                                                                                        as file:
        k, metric, _ = file.readline().strip().split(', ')
        k = int(k)

    # Split the array into train, test and split
    X_train, X_val, _, y_train, y_val, _ = train_val_test_split(X, y)

    # Initialize and train the model
    knn = KNN(k, metric)
    knn.fit(X_train, y_train)

    # Compute predictions on the test set
    start_time = time.time()
    y_pred = knn.predict(X_val)
    original_exec_time = time.time() - start_time

    # Evaluate predictions for the test set
    print('Original dimension:', n_components, k, metric)
    cls_measures = ClassificationMeasures(y_val, y_pred)
    cls_measures.print_all_measures()
    print()

    # --- Apply K-Nearest Neighbours on the reduced dataset ---

    # Split the array into train, test and split
    X_train, X_val, _, y_train, y_val, _ = train_val_test_split(X_reduced, y)

    # Initialize and train the model
    knn = KNN(k, metric)
    knn.fit(X_train, y_train)

    # Compute predictions on the test set
    start_time = time.time()
    y_pred = knn.predict(X_val)
    reduced_exec_time = time.time() - start_time

    # Evaluate predictions for the test set
    print('Reduced dimension:', n_components, k, metric)
    cls_measures = ClassificationMeasures(y_val, y_pred)
    cls_measures.print_all_measures()

    # Plot inference time as bar graph
    plt.bar(['Complete dataset', 'Reduced dataset'], [original_exec_time, reduced_exec_time])
    plt.title('PCA + KNN: Inference times for different datasets')
    plt.xlabel('Datasets')
    plt.ylabel('Inference time (seconds)')
    plt.savefig('figures/pca_knn_inference_time.png', bbox_inches='tight')
    plt.close()
    plt.clf()
    print()


def pca_dimensionality_reduction() -> None:
    """ Perform dimensionality reduction using PCA and visualize reduced data. """

    # Log function call
    print('--- pca_dimensionality_reduction')

    # Read the external data into a DataFrame
    df = pd.read_feather(f'{PROJECT_DIR}/data/external/word-embeddings.feather')

    # Extract the words
    y = df.to_numpy()[:, 0]

    # Extract the 512 length embeddings
    X = df.to_numpy()[:, 1]
    X = np.vstack(X)

    # Dimensionality reduction to 2D
    pca = PCA(n_components=2).fit(X)
    assert pca.checkPCA()
    X_2d = pca.transform(X)

    # Dimensionality reduction to 3D
    pca = PCA(n_components=3).fit(X)
    assert pca.checkPCA()
    X_3d = pca.transform(X)

    # Visualize 2D representation
    output_path = 'figures/pca_2d.png'
    plt.figure(figsize=(25,25))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], s=5)
    for i in range(X.shape[0]):
        plt.annotate(y[i], (X_2d[i, 0] + 5e-3, X_2d[i, 1]), fontsize=6)
    plt.title('PCA (Word Embeddings): Visualization for n_components=2')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    plt.clf()
    print(output_path)

    # Visualize 3D representation
    output_path = 'figures/pca_3d.png'
    fig = plt.figure(figsize=(30, 40))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], s=5)
    for i in range(X.shape[0]):
        ax.text(X_3d[i, 0] + 5e-3, X_3d[i, 1], X_3d[i, 2], y[i], fontsize=6)
    ax.set_title('PCA (Word Embeddings): Visualization for n_components=3')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    plt.clf()
    print(output_path)
    print()


# pylint: disable=duplicate-code

# pylint: disable-next=too-many-arguments
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

    # 3 K-Means Clustering

    ## 3.2 Optimal Number of Clusters
    kmeans_optimal_num_clusters()

    # 4 Gaussian Mixture Models

    ## 4.2 Optimal Number of Clusters
    gmm_optimal_num_clusters()

    # 5 Dimensionality Reduction and Visualization

    ## 5.2 Perform Dimensionality Reduction
    pca_dimensionality_reduction()

    # 6 PCA + Clustering

    ## 6.1 K-means Clustering Based on 2D Visualization
    kmeans_2d_visualization()

    ## 6.2 PCA + K-Means Clustering
    kmeans_dimensionality_reduction()

    ## 6.3 GMM Clustering Based on 2D Visualization
    gmm_2d_visualization()

    ## 6.4 PCA + GMMs
    gmm_dimensionality_reduction()

    # 8 Hierarchical Clustering
    hierarchical_clustering()

    # 9 Nearest Neighbour Search
    nearest_neighbour_search()
