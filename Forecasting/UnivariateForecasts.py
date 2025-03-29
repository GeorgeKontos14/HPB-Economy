import numpy as np

import pandas as pd

from Utils import ForecastingUtils, PreProcessing

def univariate_forecast(
        y: np.ndarray,
        train_split: float,
        start_year: int,
        horizon: int,
        country: str
    ):
    """
    Performs probabilistic forecasting on a single univariate time series, using Gradient Boosting

    Parameters:
        y (np.ndarray): The input time series of length T
        train_split (float): The split between train and test set (must be in (0,1))
        start_year (int): The start year of the time series
        horizon (int): The number of future values to be predicted
        lower_quantile (float): The lower bound quantile to be predicted
        upper_quantile (float): The upper bound quantile to be predicted
        country (str): The country for which predictions are made
        use_gbr (float): Indicates what model should be used. (True for GBR, False for ARIMA)

    Returns:
        pd.Series: The indexed training set
        pd.Series: The indexed test set
        pd.DataFrame: The prediction intervals for the test set
        pd.DataFrame: The prediction intervals for the horizon
        pd.DataFrame: The in-sample prediction intervals
    """

    test_steps, data_train, data_test, data_all = PreProcessing.preprocess_univariate_forecast(
        y = y, start_year = start_year, train_split = train_split
    )

    forecaster, _ = ForecastingUtils.tree_parzen_univariate(
        data_train=data_train,
        data_test=data_test,
        lags_bound=4,
        difference_bound=2,
        average_bound=3,
        country = country
    )

    forecaster.fit(y=data_train)
    test_preds = forecaster.predict_quantiles(steps=test_steps, quantiles=[0.05, 0.16, 0.84, 0.95], n_boot=100)
    test_med = forecaster.predict_quantiles(steps=test_steps, quantiles=[0.5])
    test_med.columns = ['median']

    test_mean = ForecastingUtils.predict_mean(
        forecaster=forecaster, to_predict=['NA'], horizon=test_steps, univariate=True
    )
    test_mean.columns = ['mean']

    test_preds = pd.concat([test_preds, test_med, test_mean], axis=1)

    forecaster.fit(y=data_all)
    horizon_preds = forecaster.predict_quantiles(steps=horizon, quantiles=[0.05, 0.16, 0.84, 0.95], n_boot=100)
    horizon_med = forecaster.predict_quantiles(steps=horizon, quantiles=[0.5])
    horizon_med.columns = ['median']

    horizon_mean = ForecastingUtils.predict_mean(
        forecaster=forecaster, to_predict=['NA'], horizon=horizon, univariate=True
    )
    horizon_mean.columns = ['mean']

    horizon_preds = pd.concat([horizon_preds, horizon_med, horizon_mean], axis=1)

    in_sample_preds = ForecastingUtils.predict_in_sample(data_train, forecaster)

    columns = [f'{country}_q_0.05', f'{country}_q_0.16', f'{country}_q_0.84', f'{country}_q_0.95']
    in_sample_preds.columns = columns
    columns.append(f'{country}_q_0.5')
    columns.append(f'{country}_mean')
    test_preds.columns = columns
    horizon_preds.columns=columns

    return data_train, data_test, test_preds, horizon_preds, in_sample_preds