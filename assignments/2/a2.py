""" Assignment 2: KMeans, Gaussian Mixture Models and Principal Component Analysis. """

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# pylint: disable=wrong-import-position

PROJECT_DIR = '../..'
sys.path.insert(0, PROJECT_DIR)

# from models.gmm import GMM
from models.k_means import KMeans
# from models.knn import KNN
# from models.pca import PCA

# pylint: enable=wrong-import-position


def gmm_2d_visualization() -> None:
    """ Perform GMM using the number of clusters estimated from 2D visualization. """


def gmm_dimensionality_reduction() -> None:
    """ Find the optimal number of clusters for the dataset reduced using Scree Plot. """


def gmm_optimal_num_clusters() -> None:
    """ Find the optimal number of clusters for Gaussian Mixture Model using
    Bayesian Information Criterion and Akaike Information Criterion. """


def hierarchical_clustering() -> None:
    """ Hierarchical clustering, with different linkage methods and distance metrics. """


def kmeans_2d_visualization() -> None:
    """ Perform KMeans using the number of clusters estimated from 2D visualization. """


def kmeans_dimensionality_reduction() -> None:
    """ Find the optimal number of clusters for the dataset reduced based on Scree Plot. """


def kmeans_optimal_num_clusters() -> None:
    """ Find the optimal number of clusters for KMeans using Elbow Method. """

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
        costs.append(kmeans.getCost())

    # Plot the relation between hyperparameter and cost
    plt.plot(k_list, costs)
    plt.title('k vs WCSS')
    plt.ylabel('Within Cluster Sum of Squares (WCSS)')
    plt.xlabel('Number of clusters (k)')
    plt.grid()
    plt.savefig('figures/kmeans_optimal_num_clusters.png', bbox_inches='tight')
    plt.close()
    plt.clf()
    print('figures/kmeans_optimal_num_clusters.png')


def nearest_neighbour_search() -> None:
    """ Find the nearest neighbour on the dataset reduced using PCA. """


def pca_dimensionality_reduction() -> None:
    """ Perform dimensionality reduction using PCA and visualize reduced data. """


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
