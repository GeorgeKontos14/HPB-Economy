import warnings

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans

from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import laplacian

from tslearn.neighbors import KNeighborsTimeSeries

import Constants

def knn_graph(
        df: pd.DataFrame
    ) -> np.ndarray:
    """
    Constructs the k-nearest neighbors graph of the dataset. k is determined heuristically as floor(n**0.5).

    Parameters:
        df (pd.Dataframe): The dataset for which the graph is constructed

    Returns:
    np.ndarray: The n X n weighted adjacency matrix representing the k-nn graph of the dataset.
    """
    n_neighbors = int(np.floor(Constants.n**0.5))
    W = kneighbors_graph(df, n_neighbors=n_neighbors, mode='distance', metric='euclidean')

    return W.toarray()

def epsilon_graph(
        df: pd.DataFrame,
        q: int,
        verbose: bool = False    
    ) -> np.ndarray:
    """
    Constructs the epsilon-neighborhood graph of the dataset.

    Parameters:
        df (pd.Dataframe): The dataset for which the graph is constructed
        q (int): The percentile of the pairwise distances that is used as the epsilon threshold
        verbose (bool): Indicates whether or not to display information

    Returns:
        np.ndarray: The n X n weighted adjacency matrix representing the epsilon-neighborhood similarity graph of the dataset
    """
    pairwise = pdist(df, metric='euclidean')
    S = squareform(pairwise)
    eps = np.percentile(pairwise, q)
    
    if verbose:
        plt.figure(figsize=(20,6))
        plt.hist(pairwise, bins = int(np.ceil(np.max(pairwise)-np.min(pairwise))), edgecolor = 'black')
        plt.title("Distribution of Pairwise Distances")
        plt.show()
        print(f"epsilon = {eps}")
    
    return np.where(S < eps, S, 0)

def dtw_knn_graph(
        df: pd.DataFrame
    ) -> np.ndarray:
    """
    Constructs the k-nearest neighbors graph of the dataset using DTW as a distance measure. k is determined heuristically as floor(n**0.5).

    Parameters:
        df (pd.Dataframe): The dataset for which the graph is constructed

    Returns:
    np.ndarray: The n X n weighted adjacency matrix representing the k-nn graph of the dataset.
    """
    n_neighbors = int(np.floor(Constants.n**0.5))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        W = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric='dtw').fit(
            df).kneighbors_graph(df, mode='distance')

    return W.toarray()

def laplacian_eigen(
        W: np.ndarray,
        normalized: bool = True,
        verbose: bool = False
    ):
    """
    Calculates the Laplacian matrix of a graph and returns its eigenvalues and eigenvectors.

    Parameters:
        n (int): The number of nodes in the graph
        W (np.ndarray): The n X n weighted adjacency matrix of the graph
        normalized (bool): Indicates whether or not to calculate the normalized Laplacian matrix
        verbose (bool): Indicates whether or not to display information
    
    Returns:
        np.ndarray: The eigenvalues of the Laplacian matrix
        np.ndarray: The eigenvectors of the Laplacian matrix
    """

    L = laplacian(W, normed=normalized)
    eigvals, eigvecs = np.linalg.eig(L)
    
    if verbose:
        plt.figure(figsize=(15,7))
        plt.title("(Ordered) Eigenvalue Plot of the Laplacian Matrix")
        plt.stem(range(Constants.n), np.sort(eigvals.real))
        plt.show()
    
    return eigvals, eigvecs

def smallest_eigenvecs(
        K: int,
        eigvals: np.ndarray,
        eigvecs: np.ndarray,
        verbose: bool = False
    ) -> np.ndarray:
    """
    Creates the final spectral clustering dataset by only maintaining the K eigenvectors that correspond to the K smallest eigenvalues of the Laplacian matrix
    
    Parameters:
        K (int): The nuber of eigenvectors to be kept
        eigvals (np.ndarray): The eigenvalues of the Laplacian Matrix
        eigvecs (np.ndarray): The eigenvectors of the Laplacian Matrix
        verbose (bool): Indicates whether or not to display information
    
    Returns:
        np.ndarray: The n X K matrix H that is used for the final step of spectral clustering
    """
    indices = np.argsort(eigvals)[:K]
    H = np.zeros((Constants.n, K))
    
    for i, ind in enumerate(indices):
        H[:, i] = eigvecs[ind].real
    
    if verbose:
        for i in range(K):
            plt.figure(figsize=(15,2))
            plt.plot(H[:,i])
            plt.show()
    
    return H

def kmeans(
        H: np.ndarray, 
        k: int, 
    ) -> np.ndarray:
    """
    Performs k-means clustering on the reduced dataset

    Parameters:
        H (np.ndarray): The dataset to be clustered
        k (int): The number of clusters
    
    Returns:
        np.ndarray: The cluster labels
    """
    k_means = KMeans(n_clusters=k, n_init=100).fit(H)
    labels = k_means.labels_

    return labels