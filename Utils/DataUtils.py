import numpy as np

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