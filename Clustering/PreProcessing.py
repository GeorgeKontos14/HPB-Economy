import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer

def fill_data(arr: np.ndarray, neighbors: int = 3) -> np.ndarray:
    """
    Fills empty values in the input array using a KNN Imputer

    Parameters:
        arr (np.ndarray): The array to be filled
        neighbors (int): The number of neighbors for the KNN algorithm

    Returns:
        np.ndarray: The filled array
    """
    imputer = KNNImputer(n_neighbors=neighbors)
    return imputer.fit_transform(arr)

def pca(data: np.ndarray, variance: float = 0.95) -> np.ndarray:
    """
    Performs Principal Component Analysis on the input data

    Parameters:
        data (np.ndarray): The initial dataset
        variance (float): The variance % of the dataset to be maintained. Based on this number, the PCA algorithm determines the number of principal components

    Returns:
        np.ndarray: The principal components of the input data
    """
    filled = data
    if np.sum(np.isnan(data)) > 0:
        filled = fill_data(data)
    return PCA(n_components=variance).fit_transform(filled)

def preprocess_pca(
        countries: list[str],
        gdp: np.ndarray, 
        pop: np.ndarray, 
        currency: np.ndarray, 
        variance: float=0.95
    ):
    """
    Constructs the final dataset to be used for the hierarchical clustering algorithms

    Parameters:
        countries (list[str]): The ISO3 code for each country (to be used as indices)
        gdp (np.ndarray): The matrix containing the annual GDP per capita for each country
        pop (np.ndarray): The matrix containing the annual population of each country
        currency (np.ndarray): The matrix containing the annual bilateral exchange rate of each country's currency and the US Dollar
        variance (float): The variance % of the dataset to be maintained. Based on this number, the PCA algorithm determines the number of principal components

    Returns:
        pd.DataFrame: The dataset where each country is represented by the concatenation of the principal components of its three time series (gdp, population and currency exchange rate)
        pd.DataFrame: The scaled version of the dataset
    """
    df = pd.DataFrame({})
    scaler = StandardScaler()
    
    gdp_components = pca(gdp, variance)
    for i in range(gdp_components.shape[1]):
        df[f"GDP #{i}"] = gdp_components[:,i]
    
    pop_components = pca(pop, variance)
    for i in range(pop_components.shape[1]):
        df[f"Population #{i}"] = pop_components[:,i]
    
    currency_components = pca(currency, variance)
    for i in range(currency_components.shape[1]):
        df[f"Currency #{i}"] = currency_components[:,i]
    
    df.index = countries
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    return df, scaled_df

def preprocess(        
        countries: list[str],
        gdp: np.ndarray, 
        pop: np.ndarray, 
        currency: np.ndarray,
        start_year: int
    ):
    """
    Constructs the final dataset to be used for the spectral clustering algorithm

    Parameters:
        countries (list[str]): The ISO3 code for each country (to be used as indices)
        gdp (np.ndarray): The matrix containing the annual GDP per capita for each country
        pop (np.ndarray): The matrix containing the annual population of each country
        currency (np.ndarray): The matrix containing the annual bilateral exchange rate of each country's currency and the US Dollar
        start_year (int): The start year of the data

    Returns:
        pd.DataFrame: The dataset where each country is represented by the concatenation of the three time series (gdp, population and currency exchange rate)
        pd.DataFrame: The scaled version of the dataset
    """
    scaler = StandardScaler()
    gdp_columns = {f"GDP {i+start_year}": gdp[:,i] for i in range(gdp.shape[1])}
    gdp_df = pd.DataFrame(gdp_columns)

    pop_columns = {f"Population {i+start_year}": pop[:, i] for i in range(pop.shape[1])}
    pop_df = pd.DataFrame(pop_columns)

    currency_data = fill_data(currency)
    currency_columns = {f"Currency {i+start_year}": currency_data[:, i] for i in range(currency_data.shape[1])}
    currency_df = pd.DataFrame(currency_columns)

    df = pd.concat([gdp_df, pop_df, currency_df], axis=1)
    df.index = countries
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

    return df, scaled_df
