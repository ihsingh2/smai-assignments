""" Assignment 2: KMeans, Gaussian Mixture Models and Principal Component Analysis. """

import sys

# import matplotlib.pyplot as plt
# import numpy as np
# from numpy.typing import NDArray
# import pandas as pd

# pylint: disable=wrong-import-position

PROJECT_DIR = '../..'
sys.path.insert(0, PROJECT_DIR)

# from models.gmm import GMM
# from models.k_means import KMeans
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
