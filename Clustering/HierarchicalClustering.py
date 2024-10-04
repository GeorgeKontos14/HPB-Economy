import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

def hierarchical_clustering(
        data: pd.DataFrame, 
        method: str,
        verbose: bool=False
    ):
    """
    Creates a clustering hierarchy and splits the clusters approprietly. The clusters are split based on the distance threshold specified by MATLAB(TM) behavior

    Parameters:
        data (pd.DataFrame): The dataset to be clustered
        method (str): The linkage method to be used
        verbose (bool): Indicates whether or not to display information

    Returns:
        np.ndarray: A numpy array containing the clusters
        float: The silhouette score to indicate the clustering's performance
    """
    hierarchy = linkage(data, method=method, metric='euclidean', optimal_ordering=True)
    labels = fcluster(hierarchy, 0.7*np.max(hierarchy[:,2]), criterion='distance')
    
    score = silhouette_score(data, labels)

    if verbose:
        clusters, counts = np.unique(labels, return_counts=True)
        print(f"Number of Clusters: {len(clusters)}")
        print(f"Cluster Sizes: {counts}")
        print(f"Silhouette Score: {score}")

        plt.figure(figsize=(15,7))
        dendrogram(hierarchy)
        plt.show()
    
    return labels, score
