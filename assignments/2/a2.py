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


def gmm_dimensionality_reduction() -> None:
    """ Find the optimal number of clusters for the dataset reduced using Scree Plot. """


def gmm_optimal_num_clusters() -> None:
    """ Find the optimal number of clusters for Gaussian Mixture Model using
    Bayesian Information Criterion and Akaike Information Criterion. """

    # Log function call
    print('gmm_optimal_num_clusters')

    # Set of hyperparameters
    k_list = range(2, 11)

    # Read the external data into a DataFrame
    df = pd.read_feather(f'{PROJECT_DIR}/data/external/word-embeddings.feather')

    # Extract the 512 length embeddings
    X_train = df.to_numpy()[:, 1]
    X_train = np.vstack(X_train)

    # List to store measures for different values of hyperparameter
    likelihoods = []
    aic_list = []
    bic_list = []

    # Iterate over all hyperparameters
    for k in k_list:

        # Initialize and fit the model
        gmm = GMM(k).fit(X_train)

        # Compute the likelihood
        likelihoods.append(gmm.getLikelihood())

        # Compute AIC and BIC
        cls_measures = ClusteringMeasures(k, X_train.shape[0], likelihoods[-1])
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
    gmm = GMM(k)
    gmm.fit(X_train)
    likelihood = gmm.getLikelihood()
    cls_measures = ClusteringMeasures(k, X_train.shape[0], likelihoods[-1])
    print('Optimal number of clusters:', k, likelihood, cls_measures.aic(), cls_measures.bic())
    print()


def hierarchical_clustering() -> None:
    """ Hierarchical clustering, with different linkage methods and distance metrics. """


def kmeans_2d_visualization() -> None:
    """ Perform KMeans using the number of clusters estimated from 2D visualization. """


def kmeans_dimensionality_reduction() -> None:
    """ Find the optimal number of clusters for the dataset reduced based on Scree Plot. """


def kmeans_optimal_num_clusters() -> None:
    """ Find the optimal number of clusters for KMeans using Elbow Method. """

    # Log function call
    print('kmeans_optimal_num_clusters')

    # Set of hyperparameters
    k_list = range(1, 11)

    # Read the external data into a DataFrame
    df = pd.read_feather(f'{PROJECT_DIR}/data/external/word-embeddings.feather')

    # Extract the 512 length embeddings
    X_train = df.to_numpy()[:, 1]
    X_train = np.vstack(X_train)

    # Initialize empty list to store costs for different values of hyperparameter
    costs = []

    # Iterate over all hyperparameters
    for k in k_list:
        kmeans = KMeans(k).fit(X_train)
        cost = kmeans.getCost()
        costs.append(cost)
        print(k, cost)

    # Plot the relation between hyperparameter and cost
    plt.plot(k_list, costs)
    plt.title('K-Means: k vs WCSS')
    plt.ylabel('Within Cluster Sum of Squares (WCSS)')
    plt.xlabel('Number of clusters (k)')
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
    kmeans = KMeans(k)
    kmeans.fit(X_train)
    y_pred = kmeans.predict(X_train)
    cost = kmeans.getCost()
    print('Elbow point:', k, cost)
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
    Y = df.to_numpy()[:, 0]

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
        plt.annotate(Y[i], (X_2d[i, 0] + 5e-3, X_2d[i, 1]), fontsize=6)
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
        ax.text(X_3d[i, 0] + 5e-3, X_3d[i, 1], X_3d[i, 2], Y[i], fontsize=6)
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

    # 8 Hierarchical Clustering
    hierarchical_clustering()

    # 9 Nearest Neighbour Search
    nearest_neighbour_search()
