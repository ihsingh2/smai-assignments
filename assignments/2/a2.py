""" Assignment 2: KMeans, Gaussian Mixture Models and Principal Component Analysis. """

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# pylint: disable=wrong-import-position

PROJECT_DIR = '../..'
sys.path.insert(0, PROJECT_DIR)

from models.gmm import GMM
from models.k_means import KMeans
# from models.knn import KNN
from models.pca import PCA
from performance_measures import ClusteringMeasures

# pylint: enable=wrong-import-position


def gmm_2d_visualization() -> None:
    """ Perform GMM using the number of clusters estimated from 2D visualization. """

    # Log function call
    print('gmm_2d_visualization')

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
    print('gmm_dimensionality_reduction')

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
    pca.checkPCA()
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
    print('gmm_optimal_num_clusters')

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


def kmeans_2d_visualization() -> None:
    """ Perform KMeans using the number of clusters estimated from 2D visualization. """

    # Log function call
    print('kmeans_2d_visualization')

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
    print('kmeans_dimensionality_reduction')

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
    plt.title('PCA: Scree Plot')
    plt.xlabel('Component Number')
    plt.ylabel('Eigenvalue')
    plt.xticks(range(0, 512, 15))
    plt.grid()
    plt.savefig('figures/pca_scree_plot.png', bbox_inches='tight')
    plt.close()
    plt.clf()
    print('figures/pca_scree_plot.png')

    # Read the optimal number of dimensions determined manually
    with open('results/pca_wordemb_optimal_dimensions.txt', 'r', encoding='utf-8') as file:
        n_components = file.readline().strip()
        n_components = int(n_components)

    # Perform dimensionality reduction based on the optimal number of dimensions
    pca = PCA(n_components=n_components).fit(X)
    pca.checkPCA()
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
    print('kmeans_optimal_num_clusters')

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


def pca_dimensionality_reduction() -> None:
    """ Perform dimensionality reduction using PCA and visualize reduced data. """

    # Log function call
    print('pca_dimensionality_reduction')

    # Read the external data into a DataFrame
    df = pd.read_feather(f'{PROJECT_DIR}/data/external/word-embeddings.feather')

    # Extract the words
    y = df.to_numpy()[:, 0]

    # Extract the 512 length embeddings
    X = df.to_numpy()[:, 1]
    X = np.vstack(X)

    # Dimensionality reduction to 2D
    pca = PCA(n_components=2).fit(X)
    pca.checkPCA()
    X_2d = pca.transform(X)

    # Dimensionality reduction to 3D
    pca = PCA(n_components=3).fit(X)
    pca.checkPCA()
    X_3d = pca.transform(X)

    # Visualize 2D representation
    output_path = 'figures/pca_2d.png'
    plt.figure(figsize=(25,25))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], s=5)
    for i in range(X.shape[0]):
        plt.annotate(y[i], (X_2d[i, 0] + 5e-3, X_2d[i, 1]), fontsize=6)
    plt.title('PCA: Visualization for n_components=2')
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
    ax.set_title('PCA: Visualization for n_components=3')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    plt.clf()
    print(output_path)
    print()


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

    # # 8 Hierarchical Clustering
    # hierarchical_clustering()

    # # 9 Nearest Neighbour Search
    # nearest_neighbour_search()
