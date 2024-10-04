import numpy as np

from tslearn.barycenters import dtw_barycenter_averaging

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
        centroids[i] = dtw_barycenter_averaging(members, max_iter=100)
    
    return centroids
