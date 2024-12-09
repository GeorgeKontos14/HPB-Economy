from typing import Tuple

import numpy as np

import pandas as pd

from skforecast.base import ForecasterBase

from Utils import DataUtils

def predict_mean(
        forecaster: ForecasterBase,
        to_predict: list[str],
        horizon: int,
        univariate: bool = True
    ) -> pd.DataFrame:
    """
    Calculates the mean of bootstrapped predictions

    Parameters:
        forecaster (ForecasterBase): Trained forecaster object
        to_predict list[str]: The country/countries for which to make predictions
        horizon (int): The number of steps to be predicted
        univariate (bool): Indicates whether or not the forecaster is univariate
    
    Returns:
        pd.DataFrame: The dataframe containing the mean for each country 
    """
    columns = [f'{country}_mean' for country in to_predict]
    if univariate:
        bootstrap_preds = forecaster.predict_bootstrapping(steps=horizon, n_boot=100)
        mean = pd.DataFrame(bootstrap_preds.mean(axis=1), columns=columns)
    else:
        cols = []
        bootstrap_preds = forecaster.predict_bootstrapping(steps=horizon, n_boot=100, levels=to_predict)
        for bootstrap_pred in bootstrap_preds.values():
            cols.append(bootstrap_pred.mean(axis=1))
        mean = pd.concat(cols, axis=1)
        mean.columns=columns

    return mean

def pinball_loss(
        y: np.ndarray, 
        y_h: np.ndarray,
        alpha: float
    ) -> float:
    """
    Calculates the pinball (quantile) loss given a quantile prediction

    Parameters:
        y (np.ndarray): The ground truth values
        y_h (np.ndarray): The quantile predictions
        alpha (float): The quantile for which the prediction has been made. Must be in [0,1]
    
    Returns:
        float: The pinball loss for the given predictions
    """
    n = len(y)
    pinball = 0
    for i, y_i in enumerate(y):
        pinball += alpha*max(y_i-y_h[i], 0) + (1-alpha)*max(y_h[i]-y_i, 0)
    return pinball/n

def probability_coverage(
        y: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray
    ) -> float:
    """
    Calculates the frequency with which the ground truth value falls into the prediction interval

    Parameters:
        y (np.ndarray): The ground truth values
        lower_bound (np.ndarray): The values of the lower bound per annum
        upper_bound (np.ndarray): The values of the upper bound per annum

    Returns:
        float: The probability coverage
    """
    return np.mean((y >= lower_bound) & (y <= upper_bound))

def calculate_metrics(
        data_test: pd.Series, 
        test_preds: pd.DataFrame, 
        lower_bound: int, 
        upper_bound: int,
        country: str
    ) -> pd.DataFrame:
    """
    Calculates Probability Coverage and Quantile losses for a given set of predictions

    Parameters:
        data_test (pd.Series): The data in the test set
        test_preds (pd.DataFrame): The predictions for the test set
        lower_bound (int): The lower quantile
        upper_bound (int): The upper quantile
        country (str): The country for which the predictions are made
    
    Returns:
        pd.Dataframe: A dataframe containing all the calculated metrics
    """
    lower_quantile = lower_bound/100
    upper_quantile = upper_bound/100

    lower_bound_vals = test_preds['lower_bound'].values
    upper_bound_vals = test_preds['upper_bound'].values

    y = data_test.values

    coverage = probability_coverage(y, lower_bound_vals, upper_bound_vals)

    lower_q_loss = pinball_loss(y, lower_bound_vals, lower_quantile)
    upper_q_loss = pinball_loss(y, upper_bound_vals, upper_quantile)
    median_loss = pinball_loss(y, test_preds['median'].values, 0.5)

    idx = [
        'Probability Coverage', 
        f'Pinball Loss for {lower_bound}th Quantile',
        f'Pinball Loss for Median',
        f'Pinball Loss for {upper_bound}th Quantile'
    ]
    metrics = pd.DataFrame(data = np.array([
        coverage, lower_q_loss, median_loss, upper_q_loss
    ]), columns=[country], index=idx)

    return metrics

def interval_width(preds: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    Calculates the length of the prediction intervals for all predicted samples for a specific country

    Parameters:
        preds (pd.DataFrame): The predictions for the specified country
        country (str): The country in question
    
    Returns:
        pd.DataFrame: The data frame containing the interval lengths per annum
    """

    intervals = pd.DataFrame(data=preds['upper_bound']-preds['lower_bound'], columns=[country])
    return intervals

def evaluate(
        data_test: pd.DataFrame,
        test_preds: pd.DataFrame,
        horizon_preds: pd.DataFrame,
        lower_bound: int,
        upper_bound: int,
        countries: list[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates a set of predictions for multiple countries

    Parameters:
        data_test (pd.DataFrame): The test set
        test_preds: (pd.DataFrame): The predictions on the test set
        horizon_preds (pd.DataFrame): The predictions for the horizon
        lower_bound (int): The lower quantile
        upper_bound (int): The upper quantile
        countries (list[str]): The countries for which to evaluate
    
    Returns:
        pd.DataFrame: The metrics for each country (see `calculate_metrics`)
        pd.DataFrame: The annual interval widths for each country
    """
    metrics = pd.DataFrame()
    intervals = pd.DataFrame()
    for country in countries:
        y = data_test[country]
        country_test = DataUtils.select_predictions(country, test_preds)
        country_horizon = DataUtils.select_predictions(country, horizon_preds)
        country_metrics = calculate_metrics(
            data_test=y,
            test_preds=country_test,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            country=country
        )
        country_intervals = interval_width(
            preds=country_horizon,
            country=country
        )

        metrics = pd.concat([metrics, country_metrics], axis=1)
        intervals = pd.concat([intervals, country_intervals], axis=1)
    
    return metrics, intervals