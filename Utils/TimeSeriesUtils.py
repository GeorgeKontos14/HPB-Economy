import numpy as np

from tslearn.barycenters import dtw_barycenter_averaging

from pmdarima import auto_arima

def cluster_centroids(
        data: np.ndarray, 
        n_clusters: int, 
        y: np.ndarray, 
        T: int
    ) -> np.ndarray:
    """
    Calculates the centroids of clusters using DTW Barycenter Averaging (DBA)

    Parameters:
        data (np.ndarray): The data that has been clustered
        n_clusters (int): The number of clusters into which the data has been separated
        y (np.ndarray): The labels the clustering algorithm has assigned
        T (int): The length of the data

    Returns:
        np.ndarray: The cluster centroids 
    """
    centroids = np.zeros((n_clusters, T, 1))

    for i in range(n_clusters):
        members = data[y==i]
        if len(members) > 0:
            centroids[i] = dtw_barycenter_averaging(members, max_iter=100)
    
    return centroids

def arima_orders(data: np.ndarray) -> np.ndarray:
    """
    Calculates the ARIMA order most suitable for each time series

    Parameters:
        data (np.ndarray): The data matrix, where each row is treated as a time series
    
    Returns:
        np.ndarray: The (n, 3) array containing the (p,d,q) order tuples for each time series
    """

    n = len(data)
    arima_orders = np.zeros((n,3), dtype=np.int64)
    for i, country in enumerate(data):
        (p,d,q) = auto_arima(country, seasonal=False, trace=False).order
        arima_orders[i] = np.array([p,d,q])
    return arima_orders