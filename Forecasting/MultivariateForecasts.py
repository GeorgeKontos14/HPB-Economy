import numpy as np

import pandas as pd

import Constants
from Utils import ForecastingUtils, PreProcessing, PostProcessing

def multiseries_independent_forecasts(
        y: np.ndarray,
        countries: list[str],
    ):
    """
    Performs probabilistic forecasting on multiple time series without considering the relations between different time series

    Parameters:
        y (np.ndarray): The input time series matrix of dimensions (m,T)
        countries (list[str]): The list containing the ISO3 codes for each country in the dataset

    Returns:
        pd.DataFrame: The indexed training set
        pd.DataFrame: The indexed test set
        pd.DataFrame: The prediction intervals for the test set
        pd.DataFrame: The prediction intervals for the horizon
        pd.DataFrame: The in-sample prediction intervals
        dict: The best parameters
    """  

    test_steps, data_train, data_test, data_all = PreProcessing.preprocess_multivariate_forecast(
        countries=countries, y=y
    )

    forecaster, _, best_params = ForecastingUtils.tree_parzen_multivariate(
        data_train=data_train,
        data_test=data_test,
        countries_to_predict=countries, 
        model_type='ForecasterRecursiveMultiSeries',
    )

    forecaster.fit(series=data_train)
    test_preds = forecaster.predict_quantiles(steps=test_steps, quantiles=[0.05,0.16,0.5,0.84,0.95], n_boot=100)
    test_preds = PostProcessing.pivot_dataframe(test_preds, 'level', 'pred')

    forecaster.fit(series=data_all)
    horizon_preds = forecaster.predict_quantiles(steps=Constants.horizon, quantiles=[0.05,0.16,0.5,0.84,0.95], n_boot=100)
    horizon_preds = PostProcessing.pivot_dataframe(horizon_preds, 'level', 'pred')
    
    in_sample_preds = ForecastingUtils.predict_in_sample(data_train, forecaster)

    return data_train, data_test, test_preds, horizon_preds, in_sample_preds, best_params

def many_to_one_forecasts(
        y: np.ndarray,
        countries: list[str],
        countries_to_predict: list[str] = None 
    ):
    """
    Performs probabilistic forecasting on multiple time series by creating multiple many-to-one models (i.e. one for each time series)

    Parameters:
        y (np.ndarray): The input time series matrix of dimensions (m,T)
        countries (list[str]): The list containing the ISO-3 codes for each country in the dataset
        countries_to_predict (list[str]): The codes of countries for which predictions should be made. If None, predictions for the entire dataset are performed
        
    Returns:
        pd.DataFrane: The indexed training set
        pd.DataFrane: The indexed test set
        pd.DataFrame: The prediction intervals for the test set
        pd.DataFrame: The prediction intervals for the horizon
        pd.DataFrame: The in-sample prediction intervals
        dict: The best parameters
    """
    T = y.shape[1]
    T_horizon = pd.date_range(start=f'{Constants.start_year+T}', end=f'{Constants.start_year+T+Constants.horizon}', freq='Y')
 
    test_steps, data_train, data_test, data_all = PreProcessing.preprocess_multivariate_forecast(
        countries=countries, y=y
    )

    test_preds = pd.DataFrame(index=data_test.index)
    horizon_preds = pd.DataFrame(index=T_horizon)

    if countries_to_predict is not None:
        to_predict = countries_to_predict
    else:
        to_predict = countries
    in_sample = []
    best_params = {}
    for country in to_predict:
        test_forecaster, horizon_forecaster, params_countries = ForecastingUtils.tree_parzen_multivariate(
            data_train=data_train,
            data_test=data_test,
            countries_to_predict=[country],
            model_type='ForecasterDirectMultiVariate',
        )
        best_params[country] = params_countries
        test_forecaster.fit(series=data_train)
        country_test_preds = test_forecaster.predict_quantiles(
            steps=test_steps,
            quantiles=[0.05,0.16,0,5,0.84,0.95],
            n_boot = 100
        )
        country_test_preds = PostProcessing.pivot_dataframe(country_test_preds, 'level', 'pred')

        horizon_forecaster.fit(series=data_all)
        country_horizon_preds = horizon_forecaster.predict_quantiles(
            steps=Constants.horizon,
            quantiles=[0.05,0.16,0.5,0.84,0.95],
            n_boot = 100
        )
        country_horizon_preds = PostProcessing.pivot_dataframe(country_horizon_preds, 'level', 'pred')
        country_in_sample_preds = ForecastingUtils.predict_in_sample(data_train, horizon_forecaster)
        in_sample.append(country_in_sample_preds)
        test_preds = pd.concat([test_preds, country_test_preds], axis=1)
        horizon_preds = pd.concat([horizon_preds, country_horizon_preds], axis=1)
        

    return data_train, data_test, test_preds, horizon_preds, pd.concat(in_sample, axis=1), best_params

def many_to_many_forecasts(
        y: np.ndarray,
        countries: list[str],
        countries_to_predict: list[str] = None 
    ):
    """
    Performs probabilistic forecasting on multiple time series by creating a direct model for multiple series

    Parameters:
        y (np.ndarray): The input time series matrix of dimensions (m,T)
        countries (list[str]): The list containing the ISO-3 codes for each country in the dataset
        countries_to_predict (list[str]): The codes of countries for which predictions should be made. If None, predictions for the entire dataset are performed
        
    Returns:
        pd.DataFrane: The indexed training set
        pd.DataFrane: The indexed test set
        pd.DataFrame: The prediction intervals for the test set
        pd.DataFrame: The prediction intervals for the horizon
        pd.DataFrame: The in-sample prediction intervals
        dict: The best parameters
    """   

    test_steps, data_train, data_test, data_all = PreProcessing.preprocess_multivariate_forecast(
        countries=countries, y=y
    )

    if countries_to_predict is not None:
        to_predict = countries_to_predict
    else:
        to_predict = countries

    test_forecaster, horizon_forecaster, best_params = ForecastingUtils.tree_parzen_multivariate(
        data_train=data_train,
        data_test=data_test,
        countries_to_predict=to_predict,
        model_type='ForecastDirectMultiOutput',
    )

    test_forecaster.fit(series=data_train)

    test_preds = test_forecaster.predict_quantiles(
        steps=test_steps, 
        quantiles=[0.05,0.16,0.84,0.95], 
        n_boot=100
    )

    test_med = test_forecaster.predict_quantiles(steps=test_steps, quantiles=[0.5])
    test_preds = pd.concat([test_preds, test_med], axis=1)

    horizon_forecaster.fit(series=data_all)

    horizon_preds = horizon_forecaster.predict_quantiles(
        steps=Constants.horizon,
        quantiles=[0.05,0.16,0.84,0.95],
        n_boot = 100
    )

    horizon_med = horizon_forecaster.predict_quantiles(steps=Constants.horizon, quantiles=[0.5])
    horizon_preds = pd.concat([horizon_preds, horizon_med], axis=1)
    in_sample_preds = ForecastingUtils.predict_in_sample(data_train, horizon_forecaster)

    return data_train, data_test, test_preds, horizon_preds, in_sample_preds, best_params

def multiseries_quantiles(
        y: np.ndarray,
        countries: list[str],
    ):
    """
    Performs quantile forecasting on multiple time series without considering the relations between different time series

    Parameters:
        y (np.ndarray): The input time series matrix of dimensions (m,T)
        countries (list[str]): The list containing the ISO3 codes for each country in the dataset

    Returns:
        pd.DataFrame: The indexed training set
        pd.DataFrame: The indexed test set
        pd.DataFrame: The prediction intervals for the test set
        pd.DataFrame: The prediction intervals for the horizon
        pd.DataFrame: The in-sample prediction intervals
        dict: The best parameters
    """  
    test_steps, data_train, data_test, data_all = PreProcessing.preprocess_multivariate_forecast(
        countries=countries, y=y
    )

    forecaster, _, best_params = ForecastingUtils.tree_parzen_quantile(
        data_train=data_train,
        data_test=data_test,
        countries_to_predict=countries, 
        model_type='ForecasterMultiSeriesQuantile',
    )
    forecaster.fit(series=data_train)
    test_preds = forecaster.predict(steps=test_steps)
    quantiles = [0.05,0.16,0.5,0.84,0.95]
    test_pred_list = []
    for q in quantiles:
        df = PostProcessing.pivot_dataframe(test_preds[q], 'level', 'pred')
        df.columns = [f'{column}_q_{q}' for column in df.columns]
        test_pred_list.append(df)
    test_preds = pd.concat(test_pred_list, axis=1)

    forecaster.fit(series=data_all)
    horizon_preds = forecaster.predict(steps=Constants.horizon)
    horizon_preds_list = []
    for q in quantiles:
        df = PostProcessing.pivot_dataframe(horizon_preds[q], 'level', 'pred')
        df.columns = [f'{column}_q_{q}' for column in df.columns]
        horizon_preds_list.append(df)
    horizon_preds = pd.concat(horizon_preds_list, axis=1)
    
    in_sample_preds = ForecastingUtils.predict_in_sample(data_train, forecaster)

    return data_train, data_test, test_preds, horizon_preds, in_sample_preds, best_params


def rnn_forecasts(
        y: np.ndarray,
        countries: list[str],
        layer_type: str,
        countries_to_predict: list[str] = None 
    ):
    """
    Performs probabilistic forecasting on multiple time series by creating a recurrent neural network model for multiple series

    Parameters:
        y (np.ndarray): The input time series matrix of dimensions (m,T)
        countries (list[str]): The list containing the ISO-3 codes for each country in the dataset
        arima_order (np.ndarray): The (p,d,q) ARIMA orders of y, of shape (m,3)
        layer_type (str): The type of recurrent layer to be used, either LSTM or RNN
        countries_to_predict (list[str]): The codes of countries for which predictions should be made. If None, predictions for the entire dataset are performed
        
    Returns:
        pd.DataFrane: The indexed training set
        pd.DataFrane: The indexed test set
        pd.DataFrame: The prediction intervals for the test set
        pd.DataFrame: The prediction intervals for the horizon
        pd.DataFrame: The in-sample prediction intervals
    """

    T = y.shape[1]

    split_ind = int(Constants.train_split*T) 

    test_steps, data_train_pure, data_test_pure, data_all_pure = PreProcessing.preprocess_multivariate_forecast(
        countries=countries, y=y
    )

    if countries_to_predict is not None:
        to_predict = countries_to_predict
    else:
        to_predict = countries

    data_train = data_train_pure
    data_test = data_test_pure
    data_all = data_all_pure

    test_forecaster, horizon_forecaster = ForecastingUtils.grid_search_rnn(
        data_train=data_train,
        data_test=data_test,
        data_all=data_all,
        lags_bound=1,
        layer_type=layer_type,
        recurrent_layers = np.array([4]),
        dense_layers = np.array([16]),
        countries_to_predict=to_predict,
    )

    test_forecaster.fit(series=data_train)

    test_preds = test_forecaster.predict_quantiles(
        steps=test_steps, 
        quantiles=[0.05, 0.16, 0.84, 0.95], 
        n_boot=100
    )

    test_med = test_forecaster.predict_quantiles(steps=test_steps, quantiles=[0.5])
    test_preds = pd.concat([test_preds, test_med], axis=1)

    horizon_forecaster.fit(series=data_all)

    horizon_preds = horizon_forecaster.predict_quantiles(
        steps=Constants.horizon,
        quantiles=[0.05, 0.16, 0.84, 0.95],
        n_boot = 100
    )

    horizon_med = horizon_forecaster.predict_quantiles(steps=Constants.horizon, quantiles=[0.5])
    horizon_preds = pd.concat([horizon_preds, horizon_med], axis=1)
    in_sample_preds = ForecastingUtils.predict_in_sample(data_train, horizon_forecaster)

    return data_train_pure, data_test_pure, test_preds, horizon_preds, in_sample_preds