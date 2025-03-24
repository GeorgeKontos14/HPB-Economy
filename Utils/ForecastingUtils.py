from typing import Tuple, Union

import numpy as np

import pandas as pd

import skforecast
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from keras.optimizers import Adam # type: ignore
from keras.losses import MeanSquaredError # type: ignore

from skforecast.base import ForecasterBase
from skforecast.deep_learning.utils import create_and_compile_model
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries

from Forecasting.ForecasterMultioutput import ForecastDirectMultiOutput
from Forecasting.ForecasterRNNProb import ForecastRNNProb
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
        f'Probability Coverage {upper_bound-lower_bound}%', 
        f'Pinball Loss for {lower_bound}th Quantile',
        f'Pinball Loss for Median',
        f'Pinball Loss for {upper_bound}th Quantile'
    ]
    metrics = pd.DataFrame(data = np.array([
        coverage, lower_q_loss, median_loss, upper_q_loss
    ]), columns=[country], index=idx)

    return metrics

def interval_width(lower_bound: pd.Series, upper_bound: pd.Series, country: str) -> pd.DataFrame:
    """
    Calculates the length of the prediction intervals for all predicted samples for a specific country

    Parameters:
        lower_bound (pd.Series): The lower bound of the interval
        upper_bound (pd.Series): The upper bound of the interval
        country (str): The country in question
    
    Returns:
        pd.DataFrame: The data frame containing the interval lengths per annum
    """

    intervals = pd.DataFrame(data=upper_bound-lower_bound, columns=[country])
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

def interval_overlap_ratio(
        T_horizon: int, 
        preds: pd.DataFrame, 
        baseline_low: np.ndarray, 
        baseline_up: np.ndarray
    ) -> float:
    """
    Calculate the interval overlap ratio between the predictions of a model and the baseline for a specific country

    Parameters:
        T_horizon (int): The number of years on the horizon
        preds (pd.DataFrame): The predictions for that country
        baseline_low (np.ndarray): The lower bound of the baseline prediction interval
        baseline_up (np.ndarray): The upper bound of the baseline prediction interval

    Returns:
        float: The (average) interval overlap ratio across the predictions of that country 
    """
    ratio = 0
    L = preds['lower_bound'].to_numpy()
    U = preds['upper_bound'].to_numpy()
    for i in range(T_horizon):
        num = max(0, min(U[i], baseline_up[i])-max(L[i], baseline_low[i]))
        denom = max(U[i], baseline_up[i])-min(L[i], baseline_low[i])
        ratio += num/denom

    return ratio/T_horizon

def relative_interval_width_ratio(
        T_horizon: int, 
        preds: pd.DataFrame, 
        baseline_low: np.ndarray, 
        baseline_up: np.ndarray
    ) -> float:
    """
    Calculate the relative interval width ratio between the predictions of a model and the baseline for a specific country

    Parameters:
        T_horizon (int): The number of years on the horizon
        preds (pd.DataFrame): The predictions for that country
        baseline_low (np.ndarray): The lower bound of the baseline prediction interval
        baseline_up (np.ndarray): The upper bound of the baseline prediction interval

    Returns:
        float: The (average) relative interval width ratio across the predictions of that country 
    """
    ratio = 0
    L = preds['lower_bound'].to_numpy()
    U = preds['upper_bound'].to_numpy()
    for i in range(T_horizon):
        num = U[i]-L[i]
        denom = baseline_up[i]-baseline_low[i]
        ratio += num/denom

    return ratio/T_horizon

def compare_to_baseline(
        countries: list[str],
        T_horizon: int,
        preds67: pd.DataFrame,
        preds90: pd.DataFrame,
        q05: np.ndarray,
        q16: np.ndarray,
        q84: np.ndarray,
        q95: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compare the results of a model to the baseline predictions

    Parameters:
        countries (list[str]): The ISO3 codes of all countries
        T_horizon (int): The number of predicted observations
        preds67 (pd.DataFrame): The 67% prediction intervals of the model
        preds90 (pd.DataFrame): The 90% prediction intervals of the model
        q05 (np.ndarray): The 5th baseline quantile
        q16 (np.ndarray): The 16th baseline quantile
        q84 (np.ndarray): The 84th baseline quantile
        q95 (np.ndarray): The 95th baseline quantile

    Returns:
        np.ndarray: The interval overlap ratio (per country) for 67% intervals
        np.ndarray: The relative width ratio (per country) for 67% intervals        
        np.ndarray: The interval overlap ratio (per country) for 90% intervals
        np.ndarray: The relative width ratio (per country) for 90% intervals
    """
    n = len(countries)
    overlap_ratios67 = np.zeros(n)
    width_ratios67 = np.zeros(n)
    overlap_ratios90 = np.zeros(n)
    width_ratios90 = np.zeros(n)
    for j, country in enumerate(countries):
        pred67 = DataUtils.select_predictions(country, preds67)
        pred90 = DataUtils.select_predictions(country, preds90)
        baseline05 = q05[:, j]
        baseline16 = q16[:, j]
        baseline84 = q84[:, j]
        baseline95 = q95[:, j]
        overlap_ratios67[j] = interval_overlap_ratio(T_horizon, pred67, baseline16, baseline84)
        width_ratios67[j] = relative_interval_width_ratio(T_horizon, pred67, baseline16, baseline84)
        overlap_ratios90[j] = interval_overlap_ratio(T_horizon, pred90, baseline05, baseline95)
        width_ratios90[j] = relative_interval_width_ratio(T_horizon, pred90, baseline05, baseline95)

    print("Interval Overlap Ratios")
    print(f"67% Prediction Intervals")
    print(f"Mean: {np.mean(overlap_ratios67)}\nStandard Deviation: {np.std(overlap_ratios67)}")
    print(f"90% Prediction Intervals")
    print(f"Mean: {np.mean(overlap_ratios90)}\nStandard Deviation: {np.std(overlap_ratios90)}")
    print("----------------------------------------")
    print("Relative Width Ratios")
    print(f"67% Prediction Intervals")
    print(f"Mean: {np.mean(width_ratios67)}\nStandard Deviation: {np.std(width_ratios67)}")
    print(f"90% Prediction Intervals")
    print(f"Mean: {np.mean(width_ratios90)}\nStandard Deviation: {np.std(width_ratios90)}")    

    return overlap_ratios67, width_ratios67, overlap_ratios90, width_ratios90

def predict_in_sample(data_train: pd.DataFrame, forecaster: ForecasterBase) -> pd.DataFrame:
    """
    Perform in sample predictions for a given (trained) forecaster

    Parameters:
        data_all (pd.DataFrame): All the observed data
        forecaster (ForecasterBase): The trained forecaster model. Could be univariate or multivariate
        T (int): The number of observations per time series
    
    Returns:
        pd.DataFrame: The in-sample 67% and 90% prediction intervals
    """
    if isinstance(forecaster, ForecasterRecursiveMultiSeries):
        size = len(list(forecaster.last_window_.values())[0])
    else:
        size = forecaster.last_window_.shape[0]
    remaining_steps = len(data_train)-size
    first_window = data_train[:size]
    all_preds = forecaster.predict_quantiles(steps = remaining_steps, last_window=first_window, quantiles = [0.05, 0.16, 0.84, 0.95])
    if isinstance(forecaster, ForecasterDirectMultiVariate):
        return all_preds
    return all_preds[:remaining_steps]

def create_univariate_forecaster(
        p: int,
        d: int,
        q: int    
    ) -> ForecasterRecursive:
    """
    Creates a univariate forecaster with the given configuration

    Parameters:
        p (int): The number of lags
        d (int): The order of differentiation
        q (int): The rolling window size
    """
    differentiation = d if d > 0 else None
    window_features = RollingFeatures(stats=['mean'], window_sizes=q) if q > 0 else None

    return ForecasterRecursive(
        regressor=GradientBoostingRegressor(loss='quantile'),
        lags=p,
        window_features=window_features,
        differentiation=differentiation
    )

def create_multivariate_forecaster(
        p: list[int],
        d: int,
        q: int,
        model_type: str,
        steps: int = None,
        level: list[str] = None   
    ) -> ForecasterBase:
    """
    Creates a multivariate forecaster object

    Parameters:
        p (list[int]): The number of lags for each variable
        d (int): The order of differentiation
        q (int): The rolling window size
        model_type (str): The type of model to create. One of 'ForecasterRecursiveMultiSeries', 'ForecasterDirectMultiVariate', 'ForecastDirectMultiOutput'
        steps (int): The number of steps to be predicted
        level (list[str]): The variables to be predicted
    """

    differentiation = d if d > 0 else None
    window_features = RollingFeatures(stats=['mean'], window_sizes=q) if q > 0 else None
    if model_type == 'ForecasterRecursiveMultiSeries':
        forecaster = ForecasterRecursiveMultiSeries(
            regressor=GradientBoostingRegressor(loss='quantile'),
            lags = p,
            window_features=window_features,
            differentiation=differentiation
        )
    elif model_type == 'ForecasterDirectMultiVariate':
        forecaster = ForecasterDirectMultiVariate(
            regressor = GradientBoostingRegressor(loss='quantile'),
            level=level[0],
            steps=steps,
            lags = p,
            window_features=window_features,
            differentiation=differentiation
        )
    elif model_type == 'ForecastDirectMultiOutput':
        forecaster = ForecastDirectMultiOutput(
            regressor= MultiOutputRegressor(
                GradientBoostingRegressor(loss='quantile')
            ),
            levels=level,
            lags=p,
            window_features=window_features,
            differentiation=differentiation,
            steps = steps
        )
    return forecaster

def create_rnn_forecaster(
        series: pd.DataFrame,
        p: list[int],
        steps: int,
        level: list[str],
        layer_type: str,
        r_u: int,
        d_u: int    
    ) -> ForecastRNNProb:
    """
    Creates an RNN Forecaster:

    Parameters:
        series (pd.DataFrame): The input data
        p (list[int]): The lags for each variable
        steps (int): The number of future steps to be predicted
        level (list[str]): The variables to be predicted
        layer_type (str): The type of recurrent layer to be used, either LSTM or RNN
        r_u (int): Number of units in the recurrent layer(s)
        d_u (int): Number of units in each dense layer
    """

    model = create_and_compile_model(
        series = series,
        lags = p,
        steps = steps,
        levels = level,
        recurrent_layer = layer_type,
        recurrent_units = r_u,
        dense_units = d_u,
        optimizer = Adam(learning_rate=0.01),
        loss=MeanSquaredError()
    )

    forecaster = ForecastRNNProb(
        regressor = model,
        levels = level,
        lags = p,
        steps = steps
    )

    return forecaster

def grid_search_univariate(
        data_train: pd.Series,
        data_test: pd.Series,
        lags_bound: int,
        difference_bound: int,
        ma_bound: int,
        country: str   
    ) -> ForecasterRecursive:
    """
    Performs exhaustive search to find the model configuration that is best suited to perform probabilistic forecasting

    Parameters:
        data_train (pd.Series): The training set
        data_test (pd.Series): The test set
        lags_bound (int): The maximum number of lags to consider
        difference_bound (int): The maximum differentiation order to consider
        ma_bound (int): The maximum rolling window size to consider
        country (str): The country for which predictions are made
        
    Returns:
        ForecasterRecursive: The forecaster that achieved the best performance the test set
    """
    test_steps = len(data_test)

    y = data_test.values

    p_list = np.arange(start=1, stop=lags_bound+1)
    d_list = np.arange(difference_bound+1)
    q_list = np.arange(ma_bound+1)
    
    # Create all possible combinations
    A, B, C = np.meshgrid(p_list, d_list, q_list, indexing='ij')
    configurations = np.stack([A.ravel(), B.ravel(), C.ravel()], axis=-1)

    interval_lengths = np.zeros((len(configurations), 2))
    cov = np.zeros((len(configurations),2))

    for i, config in enumerate(configurations):
        p = int(config[0])
        d = int(config[1])
        q = int(config[2])
        forecaster = create_univariate_forecaster(p, d, q)
        forecaster.fit(y=data_train)
        preds = forecaster.predict_quantiles(steps=test_steps, quantiles=[0.05, 0.16, 0.84, 0.95], n_boot=100)
        pred_lengths67 = interval_width(preds['q_0.16'], preds['q_0.84'], country)
        pred_lengths90 = interval_width(preds['q_0.05'], preds['q_0.95'], country)
        interval_lengths[i][0] = pred_lengths67.iloc[-1, 0]
        interval_lengths[i][1] = pred_lengths90.iloc[-1, 0]
        cov[i][0] = probability_coverage(y, preds['q_0.16'], preds['q_0.84'])
        cov[i][1] = probability_coverage(y, preds['q_0.05'], preds['q_0.95'])

    cov_ratios = cov[:,0]/interval_lengths[:,0]+cov[:,1]/interval_lengths[:,1]
    ind = np.argmax(cov_ratios)
    p = int(configurations[ind][0])
    d = int(configurations[ind][1])
    q = int(configurations[ind][2])

    return create_univariate_forecaster(p,d,q)

def grid_search_multiple_inputs(
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        lags_bound: int,
        difference_bound: int,
        ma_bound: int,
        countries_to_predict: list[str],
        model_type: str,
        horizon: int        
    ) -> Tuple[ForecasterBase, ForecasterBase]:
    """
    Performs exhaustive search to find the best hyperparameter configuration for a multivariate model

    Parameters:
        data_train (pd.DataFrame): The training set
        data_test (pd.DataFrame): The test set
        lags_bound (int): The maximum number of lags to consider
        difference_bound (int): The maximum differentiation order to consider
        ma_bound (int): The maximum rolling window size to consider
        countries_to_predict (list[str]): The countries for which predictions are made
        model_type (str): The type of model to create. One of 'ForecasterRecursiveMultiSeries', 'ForecasterDirectMultiVariate', 'ForecastDirectMultiOutput'
        horizon (int): The number of future values to ultimately predict

    Returns:
        ForecasterBase: The forecaster for the test set
        ForecasterBase: The forecaster for the horizon
    """

    test_steps = len(data_test)
    n = data_train.shape[1]
    m = len(countries_to_predict)
    
    p_list = np.arange(start=1, stop=lags_bound+1)
    d_list = np.arange(difference_bound+1)
    q_list = np.arange(ma_bound+1)

    arrays = [p_list, d_list, q_list]
    grids = np.meshgrid(*arrays, indexing='ij')
    configurations = np.stack([grid.ravel() for grid in grids], axis=-1)

    no_configs = len(configurations)
    interval_lengths = np.zeros((no_configs, m, 2))
    cov = np.zeros((no_configs, m, 2))

    for i, config in enumerate(configurations):
        p = int(config[0])
        d = int(config[1])
        q = int(config[2])
        
        forecaster = create_multivariate_forecaster(
            p = p,
            d = d,
            q = q,
            model_type=model_type,
            steps = test_steps,
            level=countries_to_predict
        )

        forecaster.fit(series = data_train)
        preds = forecaster.predict_quantiles(steps=test_steps, quantiles=[0.05, 0.16, 0.84, 0.95], n_boot=100)
        for j, country in enumerate(countries_to_predict):
            y = data_test[country]
            pred_lengths67 = interval_width(preds[f'{country}_q_0.16'], preds[f'{country}_q_0.84'], country)
            pred_lengths90 = interval_width(preds[f'{country}_q_0.05'], preds[f'{country}_q_0.95'], country)
            interval_lengths[i][j][0] = pred_lengths67.iloc[-1, 0]
            interval_lengths[i][j][1] = pred_lengths90.iloc[-1, 0]
            cov[i][j][0] = probability_coverage(y, preds[f'{country}_q_0.16'], preds[f'{country}_q_0.84'])
            cov[i][j][1] = probability_coverage(y, preds[f'{country}_q_0.05'], preds[f'{country}_q_0.95'])
    
    cov_ratios = cov[:,:,0]/interval_lengths[:,:,0]+cov[:,:,1]/interval_lengths[:,:,1]
    means = np.mean(cov_ratios, axis=1)
    ind = np.argmax(means)
    config = configurations[ind]
    p = int(config[0])
    d = int(config[1])
    q = int(config[2])

    forecaster_test = create_multivariate_forecaster(
        p = p,
        d = d,
        q = q,
        model_type=model_type,
        steps = test_steps,
        level = countries_to_predict
    )

    forecaster_horizon = create_multivariate_forecaster(
        p = p,
        d = d,
        q = q,
        model_type = model_type,
        steps = horizon,
        level = countries_to_predict
    )

    return forecaster_test, forecaster_horizon

def grid_search_rnn(
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        data_all: pd.DataFrame,
        lags_bound: int,
        layer_type: str,
        recurrent_layers: np.ndarray,
        dense_layers: np.ndarray,
        countries_to_predict: list[str],
        horizon: int
    ) -> Tuple[ForecastRNNProb, ForecastRNNProb]:
    """
    Performs exhaustive search to find the best hyperparameter configuration for a recurrent neural network

    Parameters:
        data_train (pd.DataFrame): The training set
        data_test (pd.DataFrame): The test set
        data_all (pd.DataFrame): The complete dataset
        lags_bound (int): The maximum number of lags to consider
        layer_type (str): The type of recurrent layer to be used, either LSTM or RNN
        recurrent_layers (np.ndarray): The possible numbers of recurrent units
        dense_layers (np.ndarray): The possible numbers of dense units
        countries_to_predict (list[str]): The countries for which predictions are made
        model_type (str): The type of model to create. One of 'ForecasterRecursiveMultiSeries', 'ForecasterDirectMultiVariate', 'ForecastDirectMultiOutput'
        horizon (int): The number of future values to ultimately predict

    Returns:
        ForecastRNNPron: The forecaster for the test set
        ForecastRNNPron: The forecaster for the horizon
    """

    test_steps = len(data_test)
    n = data_train.shape[1]
    m = len(countries_to_predict)
    
    p_list = np.arange(start=1, stop=lags_bound+1)
    arrays = [p_list, recurrent_layers, dense_layers]
    grids = np.meshgrid(*arrays, indexing='ij')
    configurations = np.stack([grid.ravel() for grid in grids], axis=-1)

    interval_lengths = np.zeros((len(configurations), m, 2))
    cov = np.zeros((len(configurations), m, 2))

    for i, config in enumerate(configurations):
        p = int(config[0])
        ru = int(config[1])
        du = int(config[2])

        forecaster = create_rnn_forecaster(
            series = data_train,
            p = p,
            steps = test_steps,
            level = countries_to_predict,
            layer_type = layer_type,
            r_u = ru,
            d_u = du
        )

        forecaster.fit(series = data_train)
        preds = forecaster.predict_quantiles(steps=test_steps, quantiles=[0.05, 0.16, 0.84, 0.95], n_boot=100)
        for j, country in enumerate(countries_to_predict):
            y = data_test[country]
            pred_lengths67 = interval_width(preds[f'{country}_q_0.16'], preds[f'{country}_q_0.84'], country)
            pred_lengths90 = interval_width(preds[f'{country}_q_0.05'], preds[f'{country}_q_0.95'], country)
            interval_lengths[i][j][0] = pred_lengths67.iloc[-1, 0]
            interval_lengths[i][j][1] = pred_lengths90.iloc[-1, 0]
            cov[i][j][0] = probability_coverage(y, preds[f'{country}_q_0.16'], preds[f'{country}_q_0.84'])
            cov[i][j][1] = probability_coverage(y, preds[f'{country}_q_0.05'], preds[f'{country}_q_0.95'])
    
    cov_ratios = cov[:,:,0]/interval_lengths[:,:,0]+cov[:,:,1]/interval_lengths[:,:,1]
    means = np.mean(cov_ratios, axis=1)
    ind = np.argmax(means)
    config = configurations[ind]
    p = int(config[0])
    ru = int(config[1])
    du = int(config[2])

    forecaster_test = create_rnn_forecaster(
        series = data_train,
        p = p,
        steps = test_steps,
        level = countries_to_predict,
        layer_type = layer_type,
        r_u = ru,
        d_u = du
    )

    forecaster_horizon = create_rnn_forecaster(
        series = data_all,
        p = p,
        steps = horizon,
        level = countries_to_predict,
        layer_type = layer_type,
        r_u = ru,
        d_u = du
    )

    return forecaster_test, forecaster_horizon