import pandas as pd

import warnings

import numpy as np

from tslearn.clustering import KernelKMeans, KShape, TimeSeriesKMeans

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from aeon.clustering import TimeSeriesKMedoids

def kmeans_euclidean(
        data: pd.DataFrame, 
        k: int, 
        n_init: int =100
    ):
    """
    Performs k-Means clustering using Euclidean distance

    Parameters:
        k (int): The number of partitions (clusters)
        n_init (int): The number of times the algorithm will be (randomly) initialized

    Returns:
        np.ndarray: The labels assigned by the algorithm
        np.ndarray: The final cluster centers (centroids)
    """

    km = TimeSeriesKMeans(
        n_clusters=k, metric='euclidean', n_init=n_init, verbose=False
    )
    y = km.fit_predict(data)

    return y, km.cluster_centers_

def kmeans_dtw(
        data: pd.DataFrame, 
        k: int, 
        n_init: int =100
    ):
    """
    Performs k-Means clustering using dynamic time warping (DTW)

    Parameters:
        k (int): The number of partitions (clusters)
        n_init (int): The number of times the algorithm will be (randomly) initialized

    Returns:
        np.ndarray: The labels assigned by the algorithm
        np.ndarray: The final cluster centers (centroids)
    """

    km = TimeSeriesKMeans(
        n_clusters=k, metric='dtw', n_init=n_init, verbose=False
    )
    y = km.fit_predict(data)

    return y, km.cluster_centers_

def kshape(
        data: pd.DataFrame, 
        k: int, 
        n_init: int =100
    ):
    """
    Performs k-Shape clustering for the time series

    Parameters:
        k (int): The number of partitions (clusters)
        n_init (int): The number of times the algorithm will be (randomly) initialized

    Returns:
        np.ndarray: The labels assigned by the algorithm
        np.ndarray: The final cluster centers (centroids)
    """

    ks = KShape(n_clusters=k, n_init=n_init, verbose=False)
    y = ks.fit_predict(data)

    return y, ks.cluster_centers_

def kmedoids_dtw(
        data: pd.DataFrame, 
        k: int, 
        n_init: int =100
    ):
    """
    Performs k-Medoids clustering using Dynamic Time Warping (DTW)

    Parameters:
        k (int): The number of partitions (clusters)
        n_init (int): The number of times the algorithm will be (randomly) initialized

    Returns:
        np.ndarray: The labels assigned by the algorithm
        np.ndarray: The final cluster centers (centroids)
    """

    kmed = TimeSeriesKMedoids(
        n_clusters=k, distance='dtw', n_init=n_init, verbose=False
    )
    y = kmed.fit_predict(data)

    return y, kmed.cluster_centers_

def kernel_k_means(
        data: pd.DataFrame, 
        k: int, 
        n_init: int =100
    ) -> np.ndarray:
    """
    Performs kernel k-Means clustering

    Parameters:
        k (int): The number of partitions (clusters)
        n_init (int): The number of times the algorithm will be (randomly) initialized

    Returns:
        np.ndarray: The labels assigned by the algorithm
    """

    kkm = KernelKMeans(
        n_clusters=k, kernel_params={"sigma": "auto"}, n_init=n_init, verbose=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        y = kkm.fit_predict(data)

    return y