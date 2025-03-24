import numpy as np

import pandas as pd

from skforecast.preprocessing import TimeSeriesDifferentiator

from Utils import ForecastingUtils, PreProcessing

def multiseries_independent_forecasts(
        y: np.ndarray,
        countries: list[str],
        train_split: float,
        start_year: int,
        horizon: int, 
    ):
    """
    Performs probabilistic forecasting on multiple time series without considering the relations between different time series

    Parameters:
        y (np.ndarray): The input time series matrix of dimensions (m,T)
        countries (list[str]): The list containing the ISO3 codes for each country in the dataset
        train_split (float): The split between train and test set (must be in (0,1))
        start_year (int): The start year of the time series
        horizon (int): The number of future values to be predicted

    Returns:
        pd.DataFrame: The indexed training set
        pd.DataFrame: The indexed test set
        pd.DataFrame: The prediction intervals for the test set
        pd.DataFrame: The prediction intervals for the horizon
        pd.DataFrame: The in-sample prediction intervals
    """  

    test_steps, data_train, data_test, data_all = PreProcessing.preprocess_multivariate_forecast(
        countries=countries, y=y, start_year=start_year, train_split=train_split
    )

    forecaster, _ = ForecastingUtils.grid_search_multiple_inputs(
        data_train=data_train,
        data_test=data_test,
        lags_bound=1,
        difference_bound=0,
        ma_bound=0,
        countries_to_predict=countries, 
        model_type='ForecasterRecursiveMultiSeries',
        horizon=horizon
    )

    forecaster.fit(series=data_train)
    test_preds = forecaster.predict_quantiles(steps=test_steps, quantiles=[0.05,0.16,0.84,0.95], n_boot=100)
    test_med = forecaster.predict_quantiles(steps=test_steps, quantiles=[0.5])
    test_mean = ForecastingUtils.predict_mean(
        forecaster=forecaster, to_predict=countries, horizon=test_steps, univariate=False
    )
    test_preds = pd.concat([test_preds, test_med, test_mean], axis=1)

    forecaster.fit(series=data_all)
    horizon_preds = forecaster.predict_quantiles(steps=horizon, quantiles=[0.05,0.16,0.84,0.95], n_boot=100)
    horizon_med = forecaster.predict_quantiles(steps=horizon, quantiles=[0.5])
    horizon_mean = ForecastingUtils.predict_mean(
        forecaster=forecaster, to_predict=countries, horizon=horizon, univariate=False
    )
    horizon_preds = pd.concat([horizon_preds, horizon_med, horizon_mean], axis=1)
    
    in_sample_preds = ForecastingUtils.predict_in_sample(data_train, forecaster)

    return data_train, data_test, test_preds, horizon_preds, in_sample_preds

def many_to_one_forecasts(
        y: np.ndarray,
        countries: list[str],
        train_split: float,
        start_year: int,
        horizon: int,
        countries_to_predict: list[str] = None 
    ):
    """
    Performs probabilistic forecasting on multiple time series by creating multiple many-to-one models (i.e. one for each time series)

    Parameters:
        y (np.ndarray): The input time series matrix of dimensions (m,T)
        countries (list[str]): The list containing the ISO-3 codes for each country in the dataset
        train_split (float): The split between train and test set (must be in (0,1))
        start_year (int): The start year of the time series
        horizon (int): The number of future values to be predicted
        countries_to_predict (list[str]): The codes of countries for which predictions should be made. If None, predictions for the entire dataset are performed
        
    Returns:
        pd.DataFrane: The indexed training set
        pd.DataFrane: The indexed test set
        pd.DataFrame: The prediction intervals for the test set
        pd.DataFrame: The prediction intervals for the horizon
        pd.DataFrame: The in-sample prediction intervals
    """
    T = y.shape[1]
    T_horizon = pd.date_range(start=f'{start_year+T}', end=f'{start_year+T+horizon}', freq='Y')
 
    test_steps, data_train, data_test, data_all = PreProcessing.preprocess_multivariate_forecast(
        countries=countries, y=y, start_year=start_year, train_split=train_split
    )

    test_preds = pd.DataFrame(index=data_test.index)
    horizon_preds = pd.DataFrame(index=T_horizon)

    if countries_to_predict is not None:
        to_predict = countries_to_predict
    else:
        to_predict = countries
    in_sample = []
    for country in to_predict:
        test_forecaster, horizon_forecaster = ForecastingUtils.grid_search_multiple_inputs(
            data_train=data_train,
            data_test=data_test,
            lags_bound=1,
            difference_bound=0,
            ma_bound=0,
            countries_to_predict=[country],
            model_type='ForecasterDirectMultiVariate',
            horizon=horizon
        )

        test_forecaster.fit(series=data_train)
        country_test_preds = test_forecaster.predict_quantiles(
            steps=test_steps,
            quantiles=[0.05,0.16,0.84,0.95],
            n_boot = 100
        )
        country_test_med = test_forecaster.predict_quantiles(steps=test_steps, quantiles=[0.5])
        country_test_mean = ForecastingUtils.predict_mean(
            forecaster = test_forecaster, to_predict=[country], horizon=test_steps, univariate=True
        )
        country_test_preds = pd.concat([country_test_preds, country_test_med, country_test_mean], axis=1)

        horizon_forecaster.fit(series=data_all)
        country_horizon_preds = horizon_forecaster.predict_quantiles(
            steps=horizon,
            quantiles=[0.05,0.16,0.84,0.95],
            n_boot = 100
        )
        country_horizon_med = horizon_forecaster.predict_quantiles(steps=horizon, quantiles=[0.5])
        country_horizon_mean = ForecastingUtils.predict_mean(
            forecaster=horizon_forecaster, to_predict=[country], horizon=horizon, univariate=True
        )
        country_horizon_preds = pd.concat([country_horizon_preds, country_horizon_med, country_horizon_mean], axis=1)
        country_in_sample_preds = ForecastingUtils.predict_in_sample(data_train, horizon_forecaster)
        in_sample.append(country_in_sample_preds)
        test_preds = pd.concat([test_preds, country_test_preds], axis=1)
        horizon_preds = pd.concat([horizon_preds, country_horizon_preds], axis=1)
        

    return data_train, data_test, test_preds, horizon_preds, pd.concat(in_sample, axis=1)

def many_to_many_forecasts(
        y: np.ndarray,
        countries: list[str],
        train_split: float,
        start_year: int,
        horizon: int,
        countries_to_predict: list[str] = None 
    ):
    """
    Performs probabilistic forecasting on multiple time series by creating a direct model for multiple series

    Parameters:
        y (np.ndarray): The input time series matrix of dimensions (m,T)
        countries (list[str]): The list containing the ISO-3 codes for each country in the dataset
        train_split (float): The split between train and test set (must be in (0,1))
        start_year (int): The start year of the time series
        horizon (int): The number of future values to be predicted
        countries_to_predict (list[str]): The codes of countries for which predictions should be made. If None, predictions for the entire dataset are performed
        
    Returns:
        pd.DataFrane: The indexed training set
        pd.DataFrane: The indexed test set
        pd.DataFrame: The prediction intervals for the test set
        pd.DataFrame: The prediction intervals for the horizon
        pd.DataFrame: The in-sample prediction intervals
    """   

    test_steps, data_train, data_test, data_all = PreProcessing.preprocess_multivariate_forecast(
        countries=countries, y=y, start_year=start_year, train_split=train_split
    )

    if countries_to_predict is not None:
        to_predict = countries_to_predict
    else:
        to_predict = countries

    test_forecaster, horizon_forecaster = ForecastingUtils.grid_search_multiple_inputs(
        data_train=data_train,
        data_test=data_test,
        lags_bound=1,
        difference_bound=0,
        ma_bound=1,
        countries_to_predict=to_predict,
        model_type='ForecastDirectMultiOutput',
        horizon=horizon
    )

    test_forecaster.fit(series=data_train)

    test_preds = test_forecaster.predict_quantiles(
        steps=test_steps, 
        quantiles=[0.05,0.16,0.84,0.95], 
        n_boot=100
    )

    test_med = test_forecaster.predict_quantiles(steps=test_steps, quantiles=[0.5])
    test_mean = ForecastingUtils.predict_mean(
        forecaster=test_forecaster, to_predict=to_predict, horizon=test_steps, univariate=False
    )
    test_preds = pd.concat([test_preds, test_med, test_mean], axis=1)

    horizon_forecaster.fit(series=data_all)

    horizon_preds = horizon_forecaster.predict_quantiles(
        steps=horizon,
        quantiles=[0.05,0.16,0.84,0.95],
        n_boot = 100
    )

    horizon_med = horizon_forecaster.predict_quantiles(steps=horizon, quantiles=[0.5])
    horizon_mean = ForecastingUtils.predict_mean(
        forecaster=horizon_forecaster, to_predict=to_predict, horizon=horizon, univariate=False
    )
    horizon_preds = pd.concat([horizon_preds, horizon_med, horizon_mean], axis=1)
    in_sample_preds = ForecastingUtils.predict_in_sample(data_train, horizon_forecaster)

    return data_train, data_test, test_preds, horizon_preds, in_sample_preds

def rnn_forecasts(
        y: np.ndarray,
        countries: list[str],
        layer_type: str,
        train_split: float,
        start_year: int,
        horizon: int,
        differentiation: int = None,
        countries_to_predict: list[str] = None 
    ):
    """
    Performs probabilistic forecasting on multiple time series by creating a recurrent neural network model for multiple series

    Parameters:
        y (np.ndarray): The input time series matrix of dimensions (m,T)
        countries (list[str]): The list containing the ISO-3 codes for each country in the dataset
        arima_order (np.ndarray): The (p,d,q) ARIMA orders of y, of shape (m,3)
        layer_type (str): The type of recurrent layer to be used, either LSTM or RNN
        train_split (float): The split between train and test set (must be in (0,1))
        start_year (int): The start year of the time series
        horizon (int): The number of future values to be predicted
        differentiation (int): The order of differentiation
        countries_to_predict (list[str]): The codes of countries for which predictions should be made. If None, predictions for the entire dataset are performed
        
    Returns:
        pd.DataFrane: The indexed training set
        pd.DataFrane: The indexed test set
        pd.DataFrame: The prediction intervals for the test set
        pd.DataFrame: The prediction intervals for the horizon
        pd.DataFrame: The in-sample prediction intervals
    """

    T = y.shape[1]

    split_ind = int(train_split*T) 

    test_steps, data_train_pure, data_test_pure, data_all_pure = PreProcessing.preprocess_multivariate_forecast(
        countries=countries, y=y, start_year=start_year, train_split=train_split
    )

    if countries_to_predict is not None:
        to_predict = countries_to_predict
    else:
        to_predict = countries

    if differentiation is None:
        data_train = data_train_pure
        data_test = data_test_pure
        data_all = data_all_pure
    else:
        differentiators = {country: TimeSeriesDifferentiator(order=differentiation)
                            for country in countries}
        differentiators_train = {country: TimeSeriesDifferentiator(order=differentiation)
                            for country in countries}
        differenced = np.zeros_like(y)
        for i, country in enumerate(countries):
            differenced[i] = differentiators[
                country
            ].fit_transform(y[i])
            differentiators_train[country].fit(y[i, :split_ind])
        differenced[:,:differentiation] = 0

        data_train = pd.DataFrame(differenced[:, :split_ind].T, index=data_train_pure.index, columns=countries)
        data_test = pd.DataFrame(differenced[:, split_ind:].T, index=data_test_pure.index, columns=countries)
        data_all = pd.DataFrame(differenced.T, index=data_all_pure.index, columns=countries) 

    test_forecaster, horizon_forecaster = ForecastingUtils.grid_search_rnn(
        data_train=data_train,
        data_test=data_test,
        data_all=data_all,
        lags_bound=1,
        layer_type=layer_type,
        recurrent_layers = np.array([4]),
        dense_layers = np.array([16]),
        countries_to_predict=to_predict,
        horizon=horizon
    )

    test_forecaster.fit(series=data_train)

    test_preds = test_forecaster.predict_quantiles(
        steps=test_steps, 
        quantiles=[0.05, 0.16, 0.84, 0.95], 
        n_boot=100
    )

    test_med = test_forecaster.predict_quantiles(steps=test_steps, quantiles=[0.5])
    test_mean = ForecastingUtils.predict_mean(
        forecaster=test_forecaster, to_predict=to_predict, horizon=test_steps, univariate=False
    )
    test_preds = pd.concat([test_preds, test_med, test_mean], axis=1)

    if differentiation is not None:
        for col in test_preds.columns:
            country = col[:3]
            test_preds[col] = differentiators_train[
                country
            ].inverse_transform_next_window(test_preds[col].values)

    horizon_forecaster.fit(series=data_all)

    horizon_preds = horizon_forecaster.predict_quantiles(
        steps=horizon,
        quantiles=[0.05, 0.16, 0.84, 0.95],
        n_boot = 100
    )

    horizon_med = horizon_forecaster.predict_quantiles(steps=horizon, quantiles=[0.5])
    horizon_mean = ForecastingUtils.predict_mean(
        forecaster=horizon_forecaster, to_predict=to_predict, horizon=horizon, univariate=False
    )
    horizon_preds = pd.concat([horizon_preds, horizon_med, horizon_mean], axis=1)
    in_sample_preds = ForecastingUtils.predict_in_sample(data_train, horizon_forecaster)

    if differentiation is not None:
        for col in horizon_preds.columns:
            country = col[:3]
            horizon_preds[col] = differentiators[
                country
            ].inverse_transform_next_window(horizon_preds[col].values)
            in_sample_preds[col] = differentiators[
                country
            ].inverse_transform_training(in_sample_preds[col].values)

    return data_train_pure, data_test_pure, test_preds, horizon_preds, in_sample_preds