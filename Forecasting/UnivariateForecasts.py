import numpy as np

import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive, ForecasterSarimax
from skforecast.sarimax import Sarimax

from Utils import ForecastingUtils

def univariate_forecast(
        y: np.ndarray,
        arima_order: np.ndarray,
        train_split: float,
        start_year: int,
        horizon: int,
        lower_quantile: float,
        upper_quantile: float,
        use_gbr: float = True
    ):
    """
    Performs probabilistic forecasting on a single univariate time series, using Gradient Boosting

    Parameters:
        y (np.ndarray): The input time series of length T
        arima_order (np.ndarray): The (p,d,q) ARIMA orders of y
        train_split (float): The split between train and test set (must be in (0,1))
        start_year (int): The start year of the time series
        horizon (int): The number of future values to be predicted
        lower_quantile (float): The lower bound quantile to be predicted
        upper_quantile (float): The upper bound quantile to be predicted
        use_gbr (float): Indicates what model should be used. (True for GBR, False for ARIMA)

    Returns:
        pd.Series: The indexed training set
        pd.Series: The indexed test set
        pd.DataFrame: The prediction intervals for the test set
        pd.DataFrame: The prediction intervals for the horizon
    """
    p = int(arima_order[0])
    d = int(arima_order[1])
    q = int(arima_order[2])

    T = len(y)
    split_ind = int(train_split*T)
    test_steps = T-split_ind

    T_train = pd.date_range(start=f'{start_year}', end=f'{start_year+split_ind}', freq='Y')
    T_test = pd.date_range(start=f'{start_year+split_ind}', end=f'{start_year+T}', freq='Y')
    T_all = pd.date_range(start=f'{start_year}', end=f'{start_year+T}', freq='Y')

    data_train = pd.Series(y[:split_ind], index=T_train)
    data_test = pd.Series(y[split_ind:], index=T_test)
    data_all = pd.Series(y, index=T_all)

    if use_gbr:
        lags = None
        window_features = None
        differentiation = None
        if p >= 1:
            lags = p
        if q >= 1:
            window_features = RollingFeatures(stats=['mean'], window_sizes=q)
        if d >= 1:
            differentiation = d
        if lags is None and window_features is None:
            lags=1

        forecaster = ForecasterRecursive(
            regressor = GradientBoostingRegressor(loss='quantile'),
            lags = lags,
            window_features = window_features,
            differentiation=differentiation
        )
    else:
        forecaster = ForecasterSarimax(regressor = Sarimax(order=(p,d,q)))

    forecaster.fit(y=data_train)
    if use_gbr:
        test_preds = forecaster.predict_interval(steps=test_steps, interval=[
            lower_quantile, upper_quantile], n_boot=100)
        test_med = forecaster.predict_quantiles(steps=test_steps, quantiles=[0.5])
        test_med.columns = ['median']
    else:
        test_preds = forecaster.predict_interval(steps=test_steps, interval=[
            lower_quantile, upper_quantile])
        # not actual median; just for consistency
        test_med = (test_preds['lower_bound']+test_preds['upper_bound'])/2
        test_med.name = 'median'
        test_med = pd.DataFrame(test_med)

    test_mean = ForecastingUtils.predict_mean(
        forecaster=forecaster, to_predict=['NA'], horizon=test_steps, univariate=True
    )
    test_mean.columns = ['mean']

    test_preds = pd.concat([test_preds, test_med, test_mean], axis=1)

    forecaster.fit(y=data_all)
    if use_gbr:
        horizon_preds = forecaster.predict_interval(steps=horizon, interval=[
            lower_quantile, upper_quantile], n_boot=100)
        horizon_med = forecaster.predict_quantiles(steps=horizon, quantiles=[0.5])
        horizon_med.columns = ['median']
    else:
        horizon_preds = forecaster.predict_interval(steps=horizon, interval=[
            lower_quantile, upper_quantile])
        horizon_med = (horizon_preds['lower_bound']+horizon_preds['upper_bound'])/2
        horizon_med.name = 'median'
        horizon_med = pd.DataFrame(horizon_med)

    horizon_mean = ForecastingUtils.predict_mean(
        forecaster=forecaster, to_predict=['NA'], horizon=horizon, univariate=True
    )
    horizon_mean.columns = ['mean']

    horizon_preds = pd.concat([horizon_preds, horizon_med, horizon_mean], axis=1)

    return data_train, data_test, test_preds, horizon_preds