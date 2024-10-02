import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def max_curvature(
        data: np.ndarray, 
        verbose: bool = False
    ) -> float:
    """
    Constructs the k-Distance graph for the dataset. The k-Distance graph is used to tune the epsilon parameter of the DBSCAN algorithm used for outlier detection.

    Parameters:
        data (pd.DataFrame): The dataset for which the graph is constructed
        verbose (bool): Indicates whether or not to plot the graph

    Returns:
        float: Epsilon as the value where the k-Distance graph displays maximum curvature
    """
    neighbor = NearestNeighbors(n_neighbors=2)
    nbrs = neighbor.fit(data)
    distances, _ = nbrs.kneighbors(data)
    distances = np.sort(distances, axis=0)[:,1]

    if verbose:
        plt.figure(figsize=(15,7))
        plt.plot(distances)
        plt.title('k-Distance Graph', fontsize=20)
        plt.xlabel('Data (Sorted by Distance)')
        plt.ylabel('Epsilon')
    
    first_derivative = np.gradient(distances)
    second_derivative = np.gradient(first_derivative)

    return distances[np.argmax(second_derivative)]
    
def dbscan(
        data: pd.DataFrame, 
        verbose: bool = False
    ) -> np.ndarray:
    """
    Tunes and performs the DBSCAN clustering algorithm for outlier labeling

    Parameters:
        data (pd.DataFrame): The dataset for which to detect outliers verbose (bool): Indicates whether or not to display information

    Returns:
        np.ndarray: The labels of the dataset entries, where 0 means the entry is normal and -1 means the entry is an outlier
    """
    if verbose:
        print("-----Original Dataset Review-----")
        print(data.describe())
    
    min_pts = 2*len(data.columns)
    epsilon = max_curvature(data, verbose)
    model = DBSCAN(eps=epsilon, min_samples=min_pts).fit(data)
    return model.labels_

def remove_outliers_dbscan(
        countries: list[str],
        data: pd.DataFrame,
        verbose: bool = False
    ):
    """
    Performs clustering-based outlier detection using the DBSCAN algorithm and removes the outliers from the dataset.

    Parameters:
        countries (list[str]): The list of ISO3 codes for each country
        data (pd.DataFrame): The original dataset
        verbose (bool): Indicates whether or not to display information

    Returns:
        pd.Dataframe: The original dataset with all the outliers removed
        pd.Dataframe: The outliers data
    """
    labels = dbscan(data,verbose)
    without_outliers = data.copy()
    outliers = pd.DataFrame(columns=data.columns)
    for i, country in enumerate(countries):
        if labels[i] == -1:
            row = without_outliers.loc[[country]]
            without_outliers.drop(index=country, inplace=True)
            outliers = pd.concat([outliers, row], ignore_index=False)
    
    if verbose:
        print("-----Dataset without Outliers Review-----")
        print(without_outliers.describe())
        print("-----Outliers-----")
        print(outliers)

    return without_outliers, outliers