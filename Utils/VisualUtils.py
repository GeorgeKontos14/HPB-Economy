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