from typing import Tuple

import numpy as np

import pandas as pd

import csv

import geopandas as gp

def load_clustering_data(
        names_path: str,
        gdp_path: str,
        population_path: str,
        currency_path: str,
        map_path: str,
        n: int,
        T_gdp: int,
        T_pop: int,
        T_currency: int
    ):
    """
    Loads all data required for clustering from .csv and .txt files

    Parameters:
        names_path (str): The path to the file containing each country's ISO3 code
        gdp_path (str): The path to the file containing each country's GDP data
        population_path (str): The path to the file containing each country's population data
        currency_path (str): The path to the file containing each country's currency exchange rate data
        map_path (str): The path to the map data
        n (int): The number of countries in the dataset
        T_gdp (int): The amount of years GDP data is available for (ending in 2017)
        T_pop (int): The amount of years population data is available for (ending in 2017)
        T_currency (int): The amount of years currency exchange rate data is available for (ending in 2017)

    Returns:
        list[str]: The list of ISO3 codes for each country in the dataset
        np.ndarray: The annual GDP per capita data for each country
        np.ndarray: The annual population data for each country
        np.ndarray: The annual bilateral exchange rate of each country's currency
        gd.GeoDataFrame: The map, used for visualization of clustering results
    """
    names = []
    with open(names_path, 'r') as file:
        rows = file.readlines()
        for row in rows:
            names.append(row[:3])

    gdp = np.zeros((n,T_gdp))
    with open(gdp_path, 'r') as file:
        rows = csv.reader(file)
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                gdp[j][i] = float(val)

    pop = np.zeros((n, T_pop))
    with open(population_path, 'r') as file:
        rows = csv.reader(file)
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                pop[j][i] = float(val)

    currency = np.zeros((n, T_currency))
    with open(currency_path, 'r') as file:
        rows = csv.reader(file)
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                currency[i][j] = float(val)

    world = gp.read_file(map_path)

    return names, gdp, pop, currency, world

def load_locations(
        n: int,
        locations_path: str    
    ) -> np.ndarray:
    """
    Loads the locations of the countries

    Parameters:
        n (int): The number of countries in the dataset
        locations_path (str): The path to the csv file containing the locations
    
    Returns:
        np.ndarray: The (longitude, latitude) matrix of all locations
    """
    locations = np.zeros((n,2))

    with open(locations_path, 'r') as file:
        rows = csv.reader(file)
        for i,row in enumerate(rows):
            for j,val in enumerate(row):
                locations[i][j] = val

    return locations

def load_groups(groups_path: str) -> list:
    """
    Loads the country groups used for post-processing

    Parameters:
        groups_path (str): The path to the .txt file containing group information
    
    Returns:
        list: The list of tuples (name, members), where name is a string and members is a list of ISO3 country codes
    """
    groups = []

    with open(groups_path, 'r') as file:
        rows = file.readlines()
        row_no = 0
        while row_no < len(rows)-1:
            groups.append((rows[row_no][:-1], rows[row_no+1][:-1].split(',')))
            row_no += 2
        
    return groups

def load_forecast_data(
        names_path: str,
        gdp_path: str,
        n: int,
        T_gdp: int,
    ):
    """
    Loads all data required for neural network regression from .csv and .txt files

    Parameters:
        names_path (str): The path to the file containing each country's ISO3 code
        gdp_path (str): The path to the file containing each country's GDP data
        n (int): The number of countries in the dataset
        T_gdp (int): The amount of years GDP data is available for (ending in 2017)

    Returns:
        list[str]: The list of ISO3 codes for each country in the dataset
        np.ndarray: The annual GDP per capita data for each country
    """
    names = []
    with open(names_path, 'r') as file:
        rows = file.readlines()
        for row in rows:
            names.append(row[:3])

    gdp = np.zeros((n,T_gdp))
    with open(gdp_path, 'r') as file:
        rows = csv.reader(file)
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                gdp[j][i] = float(val)

    return names, gdp

def load_labels(path: str) -> np.ndarray:
    """
    Loads the labels produced from clustering

    Parameters:
        path (str): The path to the .csv file containing the labels
    
    Returns:
        np.ndarray: The numpy array containing the labels
    """
    labels = []
    with open(path, 'r') as file:
        rows = csv.reader(file)
        for i, row in enumerate(rows):
            if i > 0:
                labels.append(int(row[1]))
    
    return np.array(labels)

def load_forecast(
        dir: str,
        start_year: int,
        T: int, 
        train_split: float, 
        horizon: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads forecasts from csv files

    Parameters:
        dir (str): The directory containing the csv files
        start_year (int): The start year of observations
        T (int): The number of observations
        train_split (float): The train-test ratio
        horizon (int): The number of future values to be predicted
    
    Returns:
        pd.DataFrame: The predictions on the test set
        pd.DataFrame: The predictions on the horizon
    """

    test_path = f'{dir}/test.csv'
    horizon_path = f'{dir}/future.csv'
    
    test_preds = pd.read_csv(test_path)
    horizon_preds = pd.read_csv(horizon_path)

    split_ind = int(train_split*T)
    T_test = pd.date_range(start=f'{start_year+split_ind}', end=f'{start_year+T}', freq='Y')
    T_horizon = pd.date_range(start=f'{start_year+T}', end=f'{start_year+T+horizon}', freq='Y')

    test_preds.index = T_test
    horizon_preds.index = T_horizon

    return test_preds, horizon_preds

def write_data(data: np.ndarray, path: str):
    """
    Saves data to the specified file

    Parameters:
        data (np.ndarray): The data to be written
        path (str): The file to which data should be written to
    """
    with open(path, 'w', newline='') as file: 
        writer = csv.writer(file)
        writer.writerows(data)

def write_labels(
        countries: list[str], 
        labels: np.ndarray, 
        path: str
    ):
    """
    Writes the results of a clustering algorithm to a file.

    Parameters:
        countries (list[str]): The list containing the ISO-3 codes of all the countries that are clustered
        labels (np.ndarray): The numpy array containing the labels that the algorithm outputed
        path (str): The file to which data should be written to 
    """
    rows = [['Country', 'Cluster']]
    for i, country in enumerate(countries):
        rows.append([country, labels[i]])
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

def write_forecasts(
        test_preds: pd.DataFrame,
        horizon_preds: pd.DataFrame,
        dir: str    
    ):
    """
    Writes predictions for test and horizon in different files

    Parameters:
        test_preds (pd.DataFrame): The predictions on the test set
        horizon_preds (pd.DataFrame): The future predictions
        dir (str): The path to the directory to store the files
    """
    test_path = f"{dir}/test.csv"
    horizon_path = f"{dir}/future.csv"
    test_preds.to_csv(test_path, index=False)
    horizon_preds.to_csv(horizon_path, index=False)

def select_predictions(
        country: str, 
        predictions: pd.DataFrame,
        only_bounds: bool = False
    ) -> pd.DataFrame:
    """
    Select and isolate the predictions of a specified country

    Parameters:
        country (str): The ISO-3 code of the specified country
        predictions (pd.DataFrame): The object containing predictions for multiple countries
        only_bounds (bool): Whether or not to only take the upper and lower bounds
        
    Returns:
        pd.DataFrame: The predictions for the selected country 
    """
    if only_bounds:
        new_cols = {
            country: country,
            f'{country}_lower_bound':'lower_bound',
            f'{country}_upper_bound':'upper_bound',
        }
    else:
        new_cols = {
            country: country,
            f'{country}_lower_bound':'lower_bound',
            f'{country}_upper_bound':'upper_bound',
            f'{country}_q_0.5': 'median',
            f'{country}_mean': 'mean'
        }

    if only_bounds:
        selected_preds = predictions[[
            country,f'{country}_lower_bound', f'{country}_upper_bound'
        ]].rename(columns=new_cols)    
    else:
        selected_preds = predictions[[
            country,f'{country}_lower_bound', f'{country}_upper_bound', f'{country}_q_0.5', f'{country}_mean'
        ]].rename(columns=new_cols)

    return selected_preds

def select_baseline(
        country: str,
        stats: pd.DataFrame,
        use_67: bool = True    
    ) -> dict:
    """
    Selects the baseline predictions for a given country

    Parameters:
        country (str): The ISO-3 code of the country to select
        stats (pd.DataFrame): The dataframe containing prediction statistics
        use_67 (bool): Whether or not to use the 67% prediction intervals; if False, 90% prediction intervals are used
    
    Returns:
        dict: A dictionary containing the bounds, mean and median for the final step of the horizon
    """
    if use_67:
        lower = 0.16
        upper = 0.84
    else:
        lower = 0.05
        upper = 0.95

    row = stats.loc[country]
    curr = row['2017Value']
    results = {
        'curr': curr,
        'lower_bound': np.log(row[lower])+curr,
        'upper_bound': np.log(row[upper])+curr,
        'median': np.log(row[0.5])+curr,
        'mean': np.log(row['Mean'])+curr
    }

    return results