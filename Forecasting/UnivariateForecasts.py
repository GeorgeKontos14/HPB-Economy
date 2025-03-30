import numpy as np

import pandas as pd

import Constants
from Utils import ForecastingUtils, PreProcessing

def univariate_forecast(
        y: np.ndarray,
        country: str
    ):
    """
    Performs probabilistic forecasting on a single univariate time series, using Gradient Boosting

    Parameters:
        y (np.ndarray): The input time series of length T
        country (str): The country for which predictions are made

    Returns:
        pd.Series: The indexed training set
        pd.Series: The indexed test set
        pd.DataFrame: The prediction intervals for the test set
        pd.DataFrame: The prediction intervals for the horizon
        pd.DataFrame: The in-sample prediction intervals
        dict: The best parameters
    """

    test_steps, data_train, data_test, data_all = PreProcessing.preprocess_univariate_forecast(y = y)

    forecaster, best_params = ForecastingUtils.tree_parzen_univariate(
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

    test_preds = pd.concat([test_preds, test_med], axis=1)

    forecaster.fit(y=data_all)
    horizon_preds = forecaster.predict_quantiles(steps=Constants.horizon, quantiles=[0.05, 0.16, 0.84, 0.95], n_boot=100)
    horizon_med = forecaster.predict_quantiles(steps=Constants.horizon, quantiles=[0.5])
    horizon_med.columns = ['median']

    horizon_preds = pd.concat([horizon_preds, horizon_med], axis=1)

    in_sample_preds = ForecastingUtils.predict_in_sample(data_train, forecaster)

    columns = [f'{country}_q_0.05', f'{country}_q_0.16', f'{country}_q_0.84', f'{country}_q_0.95']
    in_sample_preds.columns = columns
    columns.append(f'{country}_q_0.5')
    columns.append(f'{country}_mean')
    test_preds.columns = columns
    horizon_preds.columns=columns

    return data_train, data_test, test_preds, horizon_preds, in_sample_preds, best_params