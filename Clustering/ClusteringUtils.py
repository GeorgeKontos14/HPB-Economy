import numpy as np
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

from Clustering.Country import Country
from NeuralNetwork.PreProcessing import preprocess_data

def preprocessing(
        names: list[str], 
        gdp: np.ndarray, 
        locations: np.ndarray,
        areas: np.ndarray,
        pop: np.ndarray,
        labor: np.ndarray,
        currency: np.ndarray,
        T_gdp: int,
        T_labor: int,
        T_currency: int,
        n: int 
    ) -> list[Country]:
    """
    Preprocesses the data into a list of country objects

    Parameters:
    names (list[str]): the ISO-3 codes of the countries
    gdp (np.ndarray): the annual GDP per capita of each country
    locations (np.ndarray): the geographical location of each country
    areas (np.ndarray): the annual area of each country
    pop (np.ndarray): the annual population of each country
    labor (np.ndarray): the annual labor participation percentage of each country
    currency (np.ndarray): the annual bilateral exchange rate of the currency of
        each country
    T_gdp (int): The time period over which GDP data was gathered
    T_labor (int): The time period over which labor participation data was
        gathered
    T_currency (int): The time period over which currency exchange rate data was
        gathered
    n (int): The number of countries considered

    Returns:
    list[Country]: A list of Country objects, where each object contains all 
        relevant information about a country
    """
    q_gdp = 31
    q0_gdp = 16
    log_gdp, low_gdp = preprocess_data(gdp, T_gdp, q_gdp, q0_gdp)

    temp = []
    for c in labor:
        last_ind = T_labor-1
        while np.isnan(c[last_ind]):
            last_ind -= 1
        temp.append(strip_nan(c[:(last_ind+1)]))
    labor = temp

    temp = []
    for c in currency:
        temp.append(strip_nan(c[:last_ind]))
    currency = temp

    countries: list[Country] = []
    for i in range(n):
        countries.append(Country(
            names[i], log_gdp[i, -T_currency:], locations[i][0], locations[i][1],
            areas[i], pop[i, -T_currency:], labor[i], currency[i] 
        ))

    return countries

def strip_nan(y: np.ndarray):
    """
    Removes the leading nans from a numpy array

    Parameters:
    y (np.ndarray): The numpy array in question

    Returns:
    np.ndarray: The original numpy array, starting from the first not-nan element
    """
    if np.isnan(y).any():
        last_nan = np.where(np.isnan(y))[0][-1]
        return y[(last_nan+1):]
    return y

def spectral_clustering(
        data: np.ndarray,
        no_clusters: int,
        affinity: str,
        assign_labels: str
    ):
    """
    Performs spectral clustering

    Parameters:
    data (np.ndarray): The data to be clustered
    no_clusters (int): The number of clusters to be used
    affinity (str): The method with which affinity will be determined
    assign_labels (str): The label assignment method

    Returns:
    np.ndarray: The labels given by the spectral clustering algorithm
    float: The silhouette score of the labels given the data
    """
    model = SpectralClustering(
        n_clusters= no_clusters, affinity=affinity, assign_labels=assign_labels
    )
    labels = model.fit_predict(data)
    return labels, silhouette_score(data, labels)

def kmeans_clustering(
        data: np.ndarray,
        no_clusters: int,
        algorithm: str    
    ):
    """
    Performs kmeans clustering

    Parameters:
    data (np.ndarray): The data to be clustered
    no_clusters (int): The number of clusters to be used
    algorithm (str): the k-means algorithm to be used

    Returns:
    np.ndarray: The labels given by the kmeans clustering algorithm
    float: The silhouette score of the labels given the data
    """
    model = KMeans(
        n_clusters=no_clusters, algorithm=algorithm 
    )
    labels = model.fit_predict(data)
    return labels, silhouette_score(data, labels)

def hierarchical_clustering(
        data: np.ndarray,
        no_clusters: int,
        metric: str,
        linkage: str
    ):
    """
    Performs hierarchical clustering

    Parameters:
    data (np.ndarray): The data to be clustered
    no_clusters (int): The number of clusters to be used
    metric (str): The distance metric to be used
    linkage (str): The linkage to be used

    Returns:
    np.ndarray: The labels given by the agglomerative clustering algorithm
    float: The silhouette score of the labels given the data
    """
    model = AgglomerativeClustering(
        n_clusters=no_clusters, metric=metric, linkage=linkage
    )
    labels = model.fit_predict(data)
    return labels, silhouette_score(data, labels)