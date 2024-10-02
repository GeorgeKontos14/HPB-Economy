import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        labels: np.ndarray
    ):
    """
    Plots the clusters and their centroids in separate subplots

    Parameters:
        k (int): The number of clusters
        data (np.ndarray): The dataset that has been clustered
        cluster_centers (np.ndarray): The centroids of the clusters
        labels (np.ndarray): The labels that the clustering algorithm assigns to the data
    """
    plt.figure(figsize=(20, 5))
    for yi in range(k):
        plt.subplot(2, int(k/2)+1, yi+1)
        for xx in data[labels == yi]:
            plt.plot(xx.ravel(), 'k-', alpha=.2)
        plt.plot(cluster_centers[yi].ravel(), 'r-')
        plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                transform=plt.gca().transAxes)

def show_clustering(
        countries: list[str],
        k: int,
        data: np.ndarray,
        cluster_centers: np.ndarray,
        labels: np.ndarray,
        score: float
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
    
    Returns:
        list[list[str]]: The list containing the country names per cluster
    """
    plot_clusters(k, data, cluster_centers, labels)

    print(f"Silhouette Score: {score}")

    clusters = [[] for _ in range(k)]
    for i in range(k):
        members = np.where(labels == i)[0]
        for member in members:
            clusters[i].append(countries[member])
        print(f"Cluster #{i} size: {len(clusters[i])}")
        print(f"Cluster #{i} members: {', '.join(clusters[i])}")
    
    return clusters
