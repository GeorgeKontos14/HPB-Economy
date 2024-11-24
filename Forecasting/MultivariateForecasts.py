import numpy as np

import pandas as pd

from scipy.stats import mode

from sklearn.ensemble import GradientBoostingRegressor

from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirectMultiVariate

def multiseries_independent_forecasts(
        y: np.ndarray,
        countries: list[str],
        arima_orders: np.ndarray,
        train_split: float,
        start_year: int,
        horizon: int,
        lower_quantile: float,
        upper_quantile: float    
    ):
    """
    Performs probabilistic forecasting on multiple time series without considering the relations between different time series

    Parameters:
        y (np.ndarray): The input time series matrix of dimensions (m,T)
        countries (list[str]): The list containing the ISO3 codes for each country in the dataset
        arima_order (np.ndarray): The (p,d,q) ARIMA orders of y, of shape (m,3)
        train_split (float): The split between train and test set (must be in (0,1))
        start_year (int): The start year of the time series
        horizon (int): The number of future values to be predicted
        lower_quantile (float): The lower bound quantile to be predicted
        upper_quantile (float): The upper bound quantile to be predicted

    Returns:
        pd.DataFrame: The indexed training set
        pd.DataFrame: The indexed test set
        pd.DataFrame: The prediction intervals for the test set
        pd.DataFrame: The prediction intervals for the horizon
    """
    T = y.shape[1]

    lags = [int(max(1,p)) for p in arima_orders[:,0]]
    d = int(mode(arima_orders[:,1])[0])
    q = int(mode(arima_orders[:,2])[0])
    
    differentiation =  d if d > 0 else None
    window_features = RollingFeatures(stats=['mean'], window_sizes=q) if q > 0 else None

    split_ind = int(train_split*T)
    test_steps = T-split_ind

    T_train = pd.date_range(start=f'{start_year}', end=f'{start_year+split_ind}', freq='Y')
    T_test = pd.date_range(start=f'{start_year+split_ind}', end=f'{start_year+T}', freq='Y')
    T_all = pd.date_range(start=f'{start_year}', end=f'{start_year+T}', freq='Y')    

    data_train = pd.DataFrame(y[:, :split_ind].T, index=T_train, columns=countries)
    data_test = pd.DataFrame(y[:, split_ind:].T, index=T_test, columns=countries)
    data_all = pd.DataFrame(y.T, index=T_all, columns=countries)    

    forecaster = ForecasterRecursiveMultiSeries(
        regressor = GradientBoostingRegressor(loss='quantile'),
        lags=lags,
        window_features=window_features,
        differentiation=differentiation
    )

    forecaster.fit(series=data_train)
    test_preds = forecaster.predict_interval(steps=test_steps, interval=[
            lower_quantile, upper_quantile], n_boot=100)

    forecaster.fit(series=data_all)
    horizon_preds = forecaster.predict_interval(steps=horizon, interval=[
            lower_quantile, upper_quantile], n_boot=100)
    
    return data_train, data_test, test_preds, horizon_preds

def many_to_one_forecasts(
        y: np.ndarray,
        countries: list[str],
        arima_orders: np.ndarray,
        train_split: float,
        start_year: int,
        horizon: int,
        lower_quantile: float,
        upper_quantile: float,
        countries_to_predict: list[str] = None 
    ):
    """
    Performs probabilistic forecasting on multiple time series by creating multiple many-to-one models (i.e. one for each time series)

    Parameters:
        y (np.ndarray): The input time series matrix of dimensions (m,T)
        countries (list[str]): The list containing the ISO-3 codes for each country in the dataset
        arima_order (np.ndarray): The (p,d,q) ARIMA orders of y, of shape (m,3)
        train_split (float): The split between train and test set (must be in (0,1))
        start_year (int): The start year of the time series
        horizon (int): The number of future values to be predicted
        lower_quantile (float): The lower bound quantile to be predicted
        upper_quantile (float): The upper bound quantile to be predicted
        countries_to_predict (list[str]): The codes of countries for which predictions should be made. If None, predictions for the entire dataset are performed
        
    Returns:
        pd.DataFrane: The indexed training set
        pd.DataFrane: The indexed test set
        pd.DataFrame: The prediction intervals for the test set
        pd.DataFrame: The prediction intervals for the horizon
    """
    T = y.shape[1]

    lags = [int(max(1,p)) for p in arima_orders[:,0]]
    d = int(mode(arima_orders[:,1])[0])
    q = int(mode(arima_orders[:,2])[0])
    
    differentiation =  d if d > 0 else None
    window_features = RollingFeatures(stats=['mean'], window_sizes=q) if q > 0 else None

    split_ind = int(train_split*T)
    test_steps = T-split_ind

    T_train = pd.date_range(start=f'{start_year}', end=f'{start_year+split_ind}', freq='Y')
    T_test = pd.date_range(start=f'{start_year+split_ind}', end=f'{start_year+T}', freq='Y')
    T_all = pd.date_range(start=f'{start_year}', end=f'{start_year+T}', freq='Y')
    T_horizon = pd.date_range(start=f'{start_year+T}', end=f'{start_year+T+horizon}', freq='Y')

    data_train = pd.DataFrame(y[:, :split_ind].T, index=T_train, columns=countries)
    data_test = pd.DataFrame(y[:, split_ind:].T, index=T_test, columns=countries)
    data_all = pd.DataFrame(y.T, index=T_all, columns=countries)    

    test_preds = pd.DataFrame(index=T_test)
    horizon_preds = pd.DataFrame(index=T_horizon)

    if countries_to_predict is not None:
        to_predict = countries_to_predict
    else:
        to_predict = countries

    for country in to_predict:
        test_forecaster = ForecasterDirectMultiVariate(
            regressor=GradientBoostingRegressor(loss='quantile'),
            level=country,
            steps=test_steps,
            lags = lags,
            window_features=window_features,
            differentiation=differentiation
        )
        test_forecaster.fit(series=data_train)
        country_test_preds = test_forecaster.predict_interval(
            steps=test_steps,
            interval=[lower_quantile, upper_quantile],
            n_boot = 100
        )

        horizon_forecaster = ForecasterDirectMultiVariate(
            regressor=GradientBoostingRegressor(loss='quantile'),
            level=country,
            steps=horizon,
            lags = lags,
            window_features=window_features,
            differentiation=differentiation
        )
        horizon_forecaster.fit(series=data_all)
        country_horizon_preds = horizon_forecaster.predict_interval(
            steps=horizon,
            interval=[lower_quantile, upper_quantile],
            n_boot = 100
        )

        test_preds = pd.concat([test_preds, country_test_preds], axis=1)
        horizon_preds = pd.concat([horizon_preds, country_horizon_preds], axis=1)

    return data_train, data_test, test_preds, horizon_preds
