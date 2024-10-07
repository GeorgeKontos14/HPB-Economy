import numpy as np

import pandas as pd

import geopandas as gp

import matplotlib.pyplot as plt

from Utils.TimeSeriesUtils import cluster_centroids

def show_multiple_countries(
        codes: list[str],
        names: list[str],
        df: pd.DataFrame,
        start_year: int,
        duration: int    
    ):
    """
    Plots the different time series for multiple countries

    Parameters:
        codes (list[str]): The ISO3 country codes of the selected countries
        names (list[str]): The names of the selected countries
        df (pd.DataFrame): The full dataset
        start_year (int): The first year of the observations
        duration (int): The number of observations for each time series
    """
    selected_data = [df.loc[code].to_list() for code in codes]
    selected_gdp = [country[:duration] for country in selected_data]
    selected_pop = [country[duration:2*duration] for country in selected_data]
    selected_cur = [country[2*duration:] for country in selected_data]

    countries_str = ', '.join(names)

    plt.figure(figsize=(20,5))
    plt.title(f"GDP for {countries_str}")
    for i, row in enumerate(selected_gdp):
        plt.plot(range(start_year, start_year+duration), row, label=names[i])
    plt.xlabel('Year')
    plt.ylabel('GDP')
    plt.show()

    plt.figure(figsize=(20,5))
    plt.title(f"Population for {countries_str}")
    for i, row in enumerate(selected_pop):
        plt.plot(range(start_year, start_year+duration), row, label=names[i])
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.show()

    plt.figure(figsize=(20,5))
    plt.title(f"Currency for {countries_str}")
    for i, row in enumerate(selected_cur):
        plt.plot(range(start_year, start_year+duration), row, label=names[i])
    plt.xlabel('Year')
    plt.ylabel('Currency')

def plot_clusters(
        k: int,
        data: np.ndarray,
        cluster_centers: np.ndarray,
        labels: np.ndarray,
        rows: int,
        columns: int,
        start_year: int,
        T: int,
        title: str = None
    ):
    """
    Plots the clusters and their centroids in separate subplots

    Parameters:
        k (int): The number of clusters
        data (np.ndarray): The dataset that has been clustered
        cluster_centers (np.ndarray): The centroids of the clusters
        labels (np.ndarray): The labels that the clustering algorithm assigns to the data
        rows (int): The number of rows of subplots
        columns (int): The number of columns of subplots 
        start_year (int): The first year of observations
        T (int): The duration of the time series
        title (str): The title to be used in the graph
    """
    time = np.arange(start_year, start_year+T)
    plt.figure(figsize=(20, 10))
    if title is not None:
        plt.suptitle(title)
    for yi in range(k):
        plt.subplot(rows, columns, yi+1)
        for xx in data[labels == yi]:
            plt.plot(time, xx.ravel(), 'k-', alpha=.2)
        plt.plot(time, cluster_centers[yi].ravel(), 'lime')
        plt.text(0.55, 0.85, f'Cluster {yi+1}',
                transform=plt.gca().transAxes)

def show_clustering(
        countries: list[str],
        k: int,
        data: np.ndarray,
        cluster_centers: np.ndarray,
        labels: np.ndarray,
        score: float,
        rows: int,
        columns: int,
        start_year: int,
        T: int,
        title: str = None
    ) -> list[list[str]]:
    """
    Displays all relevant information for a clustering algorithm instance, including the visualization of the clusters, the silhouette score, and which countries are assigned to each cluster

    Parameters:
        countries (list[str]): The ISO3 country codes of all the clustered countries
        k (int): The number of clusters
        data (np.ndarray): The dataset that has been clustered
        cluster_centers (np.ndarray): The centroids of the clusters
        labels (np.ndarray): The labels that the clustering algorithm assigns to the data
        score (float): The silhouette score of the clustering
        rows (int): The number of rows of subplots
        columns (int): The number of columns of subplots
        start_year (int): The first year of observations
        T (int): The duration of the time series
        title (str): The title to be used in the graph
    
    Returns:
        list[list[str]]: The list containing the country names per cluster
    """
    plot_clusters(k, data, cluster_centers, labels, rows, columns, start_year, T, title)

    clusters = summarize_clustering(countries, k, labels, score)
    
    return clusters

def summarize_clustering(
        countries: list[str],
        k: int,
        labels: np.ndarray,
        score: float
    ) -> list[list[str]]:
    """
    Presents a summary of the clustering results

    Parameters:
        countries (list[str]): The ISO3 country codes of all the clustered countries
        k (int): The number of clusters
        labels (np.ndarray): The labels that the clustering algorithm assigns to the data
        score (float): The silhouette score of the clustering
    
    Returns:
        list[list[str]]: The list containing the country names per cluster
    """
    print(f"Silhouette Score: {score}")

    clusters = [[] for _ in range(k)]
    for i in range(k):
        members = np.where(labels == i)[0]
        for member in members:
            clusters[i].append(countries[member])
        print(f"Cluster #{i+1} size: {len(clusters[i])}")
        print(f"Cluster #{i+1} members: {', '.join(clusters[i])}")
    
    return clusters

def plot_centroids_outliers(
        cluster_centers: np.ndarray,
        outliers: np.ndarray,
        start_year: int,
        T: int,
        title: str = None
    ):
    """
    Plots the cluster centers and outliers on the same graph. The outliers are plotted in red and the cluster centers are plotted in green

    Parameters:
        cluster_centers (np.ndarray): The centroids determined by the clustering algorithm
        outliers (np.ndarray): The outlier data, eliminated from the clustered dataset
        start_year (int): The first year of observations
        T (int): The duration of the time series
        title (str): The title to be used in the graph
    """
    time = np.arange(start_year, start_year+T)

    plt.figure(figsize=(20,5))
    if title is not None:
        plt.suptitle(title)
    
    for center in cluster_centers:
        plt.plot(time, center.ravel(), 'lime')
    
    for outlier in outliers:
        plt.plot(time, outlier.ravel(), 'red')

    centers_line = plt.Line2D([0],[0], color='lime', lw=2)
    outliers_line = plt.Line2D([0], [0], color='red', lw=2)

    plt.legend([centers_line, outliers_line], ['Centroids', 'Outliers'])

def show_clusters_on_map(
        countries: list[str],
        labels: np.ndarray,
        world: gp.GeoDataFrame,
        title: str = None  
    ):
    """
    Visualizes the clustering results on a world map

    Parameters:
        countries (list[str]): The ISO3 country codes of all the clustered countries for all countries in the dataset
        labels (np.ndarray): The labels that the clustering algorithm assigns to the data
        world (gp.GeoDataFrame): The dataframe containing map information
        title (str): The title to be used in the graph
    """
    y_frame = pd.DataFrame({
        'CODE': countries,
        'LABEL': labels
    })

    df = world.merge(y_frame, how='inner', left_on='ISO_A3_EH', right_on='CODE')

    fig, ax = plt.subplots(1,1, figsize=(15,10))
    fig.patch.set_facecolor('black')
    if title is not None:
        plt.title(title, color='white')
    
    df.plot(column='LABEL', ax=ax)
    ax.axis('off')

    plt.tight_layout()
    plt.show()
