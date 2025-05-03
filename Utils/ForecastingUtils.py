from typing import Tuple

import numpy as np

import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

from keras.optimizers import Adam # type: ignore
from keras.losses import MeanSquaredError # type: ignore

from skforecast.base import ForecasterBase
from skforecast.deep_learning.utils import create_and_compile_model
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries

from hyperopt import hp, fmin, tpe, Trials
from hyperopt.early_stop import no_progress_loss

import Constants
from Forecasting.ForecasterMultioutput import ForecastDirectMultiOutput
from Forecasting.ForecasterRNNProb import ForecastRNNProb
from Forecasting.ForecasterQuantile import ForecasterMultiSeriesQuantile
from Utils import DataUtils, PostProcessing

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
        country: str
    ) -> pd.DataFrame:
    """
    Calculates Probability Coverage and Quantile losses for a given set of predictions

    Parameters:
        data_test (pd.Series): The data in the test set
        test_preds (pd.DataFrame): The predictions for the test set
        country (str): The country for which the predictions are made
    
    Returns:
        pd.Dataframe: A dataframe containing all the calculated metrics
    """
    intervals = [(67, 0.16, 0.84), (90, 0.05, 0.95)]
    metrics = []
    for length, lower_quantile, upper_quantile in intervals:
        lower_bound = lower_quantile*100
        upper_bound = upper_quantile*100
        lower_bound_vals = test_preds[f'{country}_q_{lower_quantile}'].values
        upper_bound_vals = test_preds[f'{country}_q_{upper_quantile}'].values

        y = data_test.values

        coverage = probability_coverage(y, lower_bound_vals, upper_bound_vals)

        lower_q_loss = pinball_loss(y, lower_bound_vals, lower_quantile)
        upper_q_loss = pinball_loss(y, upper_bound_vals, upper_quantile)
        median_loss = pinball_loss(y, test_preds[f'{country}_q_0.5'].values, 0.5)

        idx = [
            f'Probability Coverage {upper_bound-lower_bound}%', 
            f'Pinball Loss for {lower_bound}th Quantile',
            f'Pinball Loss for Median',
            f'Pinball Loss for {upper_bound}th Quantile'
        ]
        metrics.append(pd.DataFrame(data = np.array([
        coverage, lower_q_loss, median_loss, upper_q_loss
    ]), columns=[country], index=idx))
    
    metrics = pd.concat(metrics, axis=0)

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

def evaluate_forecast(
        data: pd.DataFrame,
        preds: pd.DataFrame,
        countries: list[str]
    ) -> pd.DataFrame:
    """
    Evaluate a single forecast

    Parameters:
        data (pd.DataFrame): The observed values
        preds (pd.DataFrame): The predictions for the observed values
        countries (list[str]): The countries for which to evaluate

    Returns:
        pd.DataFrame: The metrics for the model
    """
    metrics = []
    for country in countries:
        y = data[country]
        metrics.append(calculate_metrics(y, preds, country))
    metrics = pd.concat(metrics, axis=1)
    return metrics

def evaluate_multiple_forecasts(
        data: pd.DataFrame,
        predictions: list[pd.DataFrame],
        models: list[str],
        countries: list[str]
    ) -> pd.DataFrame:
    """
    Evaluate multiple forecasts

    Parameters:
        data (pd.DataFrame): The observed values
        predictions (list[pd.DataFrame]): The predictions of each forecast for the observed values
        models (list[str]): The underlying models of each forecaster
        countries (list[str]): The countries for which to evaluate

    Returns:
        pd.DataFrame: The (average) metrics for the model
    """
    metrics = []
    for preds in predictions:
        model_metrics = evaluate_forecast(data, preds, countries)
        metrics.append(model_metrics.mean(axis=1))
    metrics = pd.concat(metrics, axis=1)
    metrics.columns = models
    return metrics

def interval_overlap_ratio(
        preds: pd.DataFrame, 
        baseline_low: np.ndarray, 
        baseline_up: np.ndarray,
        country: str,
        use67: bool = True
    ) -> float:
    """
    Calculate the interval overlap ratio between the predictions of a model and the baseline for a specific country

    Parameters:
        preds (pd.DataFrame): The predictions for that country
        baseline_low (np.ndarray): The lower bound of the baseline prediction interval
        baseline_up (np.ndarray): The upper bound of the baseline prediction interval
        country (str): The country for which to make the calculations
        use67 (bool): Flag indicating whether to use the 67% intervals

    Returns:
        float: The (average) interval overlap ratio across the predictions of that country 
    """
    ratio = 0
    if use67:
        L = preds[f'{country}_q_0.16'].to_numpy()
        U = preds[f'{country}_q_0.84'].to_numpy()
    else:
        L = preds[f'{country}_q_0.05'].to_numpy()
        U = preds[f'{country}_q_0.95'].to_numpy()        
    for i in range(Constants.horizon):
        num = max(0, min(U[i], baseline_up[i])-max(L[i], baseline_low[i]))
        denom = max(U[i], baseline_up[i])-min(L[i], baseline_low[i])
        ratio += num/denom

    return ratio/Constants.horizon

def relative_interval_width_ratio(
        preds: pd.DataFrame, 
        baseline_low: np.ndarray, 
        baseline_up: np.ndarray,
        country: str,
        use67: bool = True
    ) -> float:
    """
    Calculate the relative interval width ratio between the predictions of a model and the baseline for a specific country

    Parameters:
        T_horizon (int): The number of years on the horizon
        preds (pd.DataFrame): The predictions for that country
        baseline_low (np.ndarray): The lower bound of the baseline prediction interval
        baseline_up (np.ndarray): The upper bound of the baseline prediction interval
        country (str): The country for which to make the calculations
        use67 (bool): Flag indicating whether to use the 67% intervals

    Returns:
        float: The (average) relative interval width ratio across the predictions of that country 
    """
    ratio = 0
    if use67:
        L = preds[f'{country}_q_0.16'].to_numpy()
        U = preds[f'{country}_q_0.84'].to_numpy()
    else:
        L = preds[f'{country}_q_0.05'].to_numpy()
        U = preds[f'{country}_q_0.95'].to_numpy()  
    for i in range(Constants.horizon):
        num = U[i]-L[i]
        denom = baseline_up[i]-baseline_low[i]
        ratio += num/denom

    return ratio/Constants.horizon

def compare_to_baseline(
        countries: list[str],
        horizon_preds: pd.DataFrame,
        q05: np.ndarray,
        q16: np.ndarray,
        q84: np.ndarray,
        q95: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compare the results of a model to the baseline predictions

    Parameters:
        countries (list[str]): The ISO3 codes of all countries
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
        pred67 = horizon_preds[[f'{country}_q_0.16', f'{country}_q_0.84']]
        pred90 = horizon_preds[[f'{country}_q_0.05', f'{country}_q_0.95']]
        baseline05 = q05[:, j]
        baseline16 = q16[:, j]
        baseline84 = q84[:, j]
        baseline95 = q95[:, j]
        overlap_ratios67[j] = interval_overlap_ratio(pred67, baseline16, baseline84, country)
        width_ratios67[j] = relative_interval_width_ratio(pred67, baseline16, baseline84, country)
        overlap_ratios90[j] = interval_overlap_ratio(pred90, baseline05, baseline95, country, use67 =  False)
        width_ratios90[j] = relative_interval_width_ratio(pred90, baseline05, baseline95, country, use67 =  False)

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
    elif isinstance(forecaster, ForecasterMultiSeriesQuantile):
        size = list(forecaster.last_window_.values())[0].shape[0]
    else:
        size = forecaster.last_window_.shape[0]
    remaining_steps = len(data_train)-size
    first_window = data_train[:size]
    if isinstance(forecaster, ForecasterMultiSeriesQuantile):
        all_preds = forecaster.predict(steps = remaining_steps, last_window=first_window)
    else:
        all_preds = forecaster.predict_quantiles(steps = remaining_steps, last_window=first_window, quantiles = [0.05, 0.16, 0.84, 0.95])
    if isinstance(forecaster, ForecasterMultiSeriesQuantile):
        quantiles=[0.05,0.16,0.84,0.95]
        pred_list = []
        for q in quantiles:
            df = PostProcessing.pivot_dataframe(all_preds[q], 'level','pred')
            df.columns = [f'{column}_q_{q}' for column in df.columns]
            pred_list.append(df)
        all_preds = pd.concat(pred_list,axis=1)
    elif not isinstance(forecaster, ForecastDirectMultiOutput):
        all_preds = PostProcessing.pivot_dataframe(all_preds, pivot_column='level', target_column='pred')
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
        regressor=GradientBoostingRegressor(loss='absolute_error'),
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
        model_type (str): The type of model to create. One of 'ForecasterRecursiveMultiSeries', 'ForecasterDirectMultiVariate', 'ForecastDirectMultiOutput', 'ForecasterMultiSeriesQuantile
        steps (int): The number of steps to be predicted
        level (list[str]): The variables to be predicted
    """

    differentiation = d if d > 0 else None
    window_features = RollingFeatures(stats=['mean'], window_sizes=q) if q > 0 else None
    if model_type == 'ForecasterRecursiveMultiSeries':
        forecaster = ForecasterRecursiveMultiSeries(
            regressor=GradientBoostingRegressor(loss='absolute_error'),
            lags = p,
            window_features=window_features,
            differentiation=differentiation
        )
    elif model_type == 'ForecasterDirectMultiVariate':
        forecaster = ForecasterDirectMultiVariate(
            regressor = GradientBoostingRegressor(loss='absolute_error'),
            level=level[0],
            steps=steps,
            lags = p,
            window_features=window_features,
            differentiation=differentiation
        )
    elif model_type == 'ForecastDirectMultiOutput':
        forecaster = ForecastDirectMultiOutput(
            regressor= MultiOutputRegressor(
                GradientBoostingRegressor(loss='absolute_error')
            ),
            levels=level,
            lags=p,
            window_features=window_features,
            differentiation=differentiation,
            steps = steps
        )
    elif model_type == 'ForecasterMultiSeriesQuantile':
        forecaster = ForecasterMultiSeriesQuantile(
            quantiles=[0.05,0.16,0.5,0.84,0.95],
            levels = level,
            lags = p,
            window_features=window_features,
            differentiation=differentiation
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

def grid_search_rnn(
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        data_all: pd.DataFrame,
        lags_bound: int,
        layer_type: str,
        recurrent_layers: np.ndarray,
        dense_layers: np.ndarray,
        countries_to_predict: list[str]
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

    Returns:
        ForecastRNNPron: The forecaster for the test set
        ForecastRNNPron: The forecaster for the horizon
    """

    test_steps = len(data_test)
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
        steps = Constants.horizon,
        level = countries_to_predict,
        layer_type = layer_type,
        r_u = ru,
        d_u = du
    )

    return forecaster_test, forecaster_horizon

def tree_parzen_univariate(
        data_train: pd.Series,
        data_test: pd.Series,
        country: str
    )-> Tuple[ForecasterRecursive, dict]:
    """
    Performs hyperparameter tuning for univriate forecasts using tree-structured Parzen estimation

    Parameters:
        data_train (pd.Series): The training set
        data_test (pd.Series): The test set
        country (str): The country for which predictions are made

    Returns:
        ForecasterRecursive: The best performing forecaster
        dict: The best parameters      
    """

    test_steps = len(data_test)
    search_space = {
        'p': hp.quniform('p', 1, Constants.lags_bound, 1),
        'd': hp.quniform('d', 0, Constants.difference_bound, 1),
        'q': hp.quniform('q', 0, Constants.average_bound, 1)
    }
    N = Constants.average_bound*Constants.difference_bound*(Constants.lags_bound-1)

    if N <= 250:
        stop = 6
    elif N < 750:
        stop = 10
    else:
        stop = 15    
    
    def objective(params):
        p, d, q = int(params['p']), int(params['d']), int(params['q'])
        model = create_univariate_forecaster(p, d, q)
        model.fit(y=data_train)
        q_preds = model.predict_quantiles(
            steps=test_steps,
            quantiles=[0.05, 0.16, 0.5, 0.84, 0.95]
        )
        preds = model.predict(steps=test_steps)
        p_lengths_67 = interval_width(q_preds['q_0.16'], q_preds['q_0.84'], country)
        p_lengths_90 = interval_width(q_preds['q_0.05'], q_preds['q_0.95'], country)
        lengths_67 = p_lengths_67.iloc[-1,0]
        lengths_90 = p_lengths_90.iloc[-1,0]
        cov_67 = probability_coverage(data_test, q_preds['q_0.16'], q_preds['q_0.84'])
        cov_90 = probability_coverage(data_test, q_preds['q_0.05'], q_preds['q_0.95'])        
        int_score_67 = np.sqrt(lengths_67)*np.abs(cov_67-0.67)
        int_score_90 = np.sqrt(lengths_90)*np.abs(cov_90-0.9)
        loss_median = pinball_loss(data_test, q_preds['q_0.5'], 0.5)
        mse = mean_squared_error(data_test, preds)
        return {'loss': loss_median+(int_score_67+int_score_90)/2+0.2*mse, 'status': 'ok'}
    
    trials = Trials()
    best_params = fmin(
        fn = objective,
        space = search_space,
        algo = tpe.suggest,
        max_evals = int(0.3*N),
        trials=trials,
        early_stop_fn=no_progress_loss(stop)
    )

    lags = int(best_params['p'])
    difference = int(best_params['d'])
    ma = int(best_params['q'])  

    forecaster = create_univariate_forecaster(lags, difference, ma)

    return forecaster, best_params  


def tree_parzen_multivariate(
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        countries_to_predict: list[str],
        model_type: str,
    )-> Tuple[ForecasterBase, ForecasterBase]:
    """
    Performs hyperparameter tuning for multivariate forecasts using tree-structured Parzen estimation

    Parameters:
        data_train (pd.DataFrame): The training set
        data_test (pd.DataFrame): The test set
        countries_to_predict (list[str]): The countries for which predictions are made
        model_type (str): The type of model to create. One of 'ForecasterRecursiveMultiSeries', 'ForecasterDirectMultiVariate', 'ForecastDirectMultiOutput'
        horizon (int): The number of future values to ultimately predict

    Returns:
        ForecasterBase: The forecaster for the test set
        ForecasterBase: The forecaster for the horizon
        dict: The best parameters 
    """

    test_steps = len(data_test)
    n = data_train.shape[1]
    m = len(countries_to_predict)

    search_space = {
        'd': hp.quniform('d', 0, Constants.difference_bound, 1),
        'q': hp.quniform('q', 0, Constants.average_bound, 1),
    }
    N = Constants.difference_bound*Constants.average_bound
    search_space['p'] = [hp.choice(f"p_{i}", list(range(1,Constants.lags_bound))) for i in range(n)]
    N = N*(Constants.lags_bound-1)**n

    if N <= 250:
        stop = 6
    elif N < 750:
        stop = 10
    else:
        stop = 15    

    def objective(params):
        p, d, q = params['p'], int(params['d']), int(params['q'])
        model = create_multivariate_forecaster(
            p, d, q, model_type, test_steps, countries_to_predict
        )
        model.fit(series=data_train)
        int_score_67 = 0
        int_score_90 = 0
        loss_median = 0
        q_preds = model.predict_quantiles(
            steps=test_steps,
            quantiles=[0.05,0.16,0.5,0.84, 0.95],
            n_boot=100
        )
        preds = model.predict(steps=test_steps)
        if not isinstance(model, ForecastDirectMultiOutput):
            q_preds = PostProcessing.pivot_dataframe(q_preds, 'level', 'pred')
            preds = PostProcessing.pivot_dataframe(preds, 'level', 'pred')
        mse = 0        
        for country in countries_to_predict:
            y = data_test[country]
            p_lengths_67 = interval_width(q_preds[f'{country}_q_0.16'], q_preds[f'{country}_q_0.84'], country)
            p_lengths_90 = interval_width(q_preds[f'{country}_q_0.05'], q_preds[f'{country}_q_0.95'], country)
            lengths_67 = p_lengths_67.iloc[-1,0]
            lengths_90 = p_lengths_90.iloc[-1,0]
            cov_67 = probability_coverage(y, q_preds[f'{country}_q_0.16'], q_preds[f'{country}_q_0.84'])
            cov_90 = probability_coverage(y, q_preds[f'{country}_q_0.05'], q_preds[f'{country}_q_0.95'])
            int_score_67 += np.sqrt(lengths_67)*np.abs(cov_67-0.67)
            int_score_90 += np.sqrt(lengths_90)*np.abs(cov_90-0.9)
            loss_median += pinball_loss(y, q_preds[f'{country}_q_0.5'], 0.5)
            mse += mean_squared_error(y, preds[country]) 
        int_score_67 = int_score_67/m
        int_score_90 = int_score_90/m
        loss_median = loss_median/m
        mse = mse/m
        return {'loss': loss_median+(int_score_67+int_score_90)/2+0.2*mse, 'status': 'ok'}

    trials = Trials()
    best_params = fmin(
        fn = objective,
        space = search_space,
        algo = tpe.suggest,
        max_evals = int(0.3*N),
        trials=trials,
        early_stop_fn=no_progress_loss(stop)
    )

    lags = [int(best_params[f"p_{i}"])+1 for i in range(n)]
    difference = int(best_params['d'])
    ma = int(best_params['q'])

    forecaster_test = create_multivariate_forecaster(
        p = lags,
        d = difference,
        q = ma,
        model_type=model_type,
        steps = test_steps,
        level = countries_to_predict
    )

    forecaster_horizon = create_multivariate_forecaster(
        p = lags,
        d = difference,
        q = ma,
        model_type = model_type,
        steps = Constants.horizon,
        level = countries_to_predict
    )

    return forecaster_test, forecaster_horizon, best_params

def tree_parzen_quantile(
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        countries_to_predict: list[str],
        model_type: str = "ForecasterMultiSeriesQuantile",
    )-> Tuple[ForecasterBase, ForecasterBase]:
    """
    Performs hyperparameter tuning for multivariate quantile forecasts using tree-structured Parzen estimation

    Parameters:
        data_train (pd.DataFrame): The training set
        data_test (pd.DataFrame): The test set
        countries_to_predict (list[str]): The countries for which predictions are made
        model_type (str): The type of model to create. One of 'ForecasterMultiSeriesQuantile'
        horizon (int): The number of future values to ultimately predict

    Returns:
        ForecasterBase: The forecaster for the test set
        ForecasterBase: The forecaster for the horizon
        dict: The best parameters 
    """
    test_steps = len(data_test)
    n = data_train.shape[1]
    m = len(countries_to_predict)

    search_space = {
        'd': hp.quniform('d', 0, Constants.difference_bound, 1),
        'q': hp.quniform('q', 0, Constants.average_bound, 1),
    }
    N = Constants.difference_bound*Constants.average_bound
    search_space['p'] = [hp.choice(f"p_{i}", list(range(1,Constants.lags_bound))) for i in range(n)]
    N = N*(Constants.lags_bound-1)**n

    if N <= 250:
        stop = 6
    elif N < 750:
        stop = 10
    else:
        stop = 15  

    def objective(params):
        p, d, q = params['p'], int(params['d']), int(params['q'])
        model = create_multivariate_forecaster(
            p, d, q, model_type, test_steps, countries_to_predict
        )
        model.fit(series=data_train)
        int_score_67 = 0
        int_score_90 = 0
        loss_median = 0
        q_preds = model.predict(steps=test_steps)
        quantiles = [0.05,0.16,0.5,0.84,0.95]
        predictions = []
        for q in quantiles:
            df = PostProcessing.pivot_dataframe(q_preds[q], 'level', 'pred')
            df.columns = [f'{column}_q_{q}' for column in df.columns]
            predictions.append(df)
        q_preds = pd.concat(predictions, axis=1)
        for country in countries_to_predict:
            y = data_test[country]
            p_lengths_67 = interval_width(q_preds[f'{country}_q_0.16'], q_preds[f'{country}_q_0.84'], country)
            p_lengths_90 = interval_width(q_preds[f'{country}_q_0.05'], q_preds[f'{country}_q_0.95'], country)
            lengths_67 = p_lengths_67.iloc[-1,0]
            lengths_90 = p_lengths_90.iloc[-1,0]
            cov_67 = probability_coverage(y, q_preds[f'{country}_q_0.16'], q_preds[f'{country}_q_0.84'])
            cov_90 = probability_coverage(y, q_preds[f'{country}_q_0.05'], q_preds[f'{country}_q_0.95'])
            int_score_67 += np.sqrt(lengths_67)*np.abs(cov_67-0.67)
            int_score_90 += np.sqrt(lengths_90)*np.abs(cov_90-0.9)
            loss_median += pinball_loss(y, q_preds[f'{country}_q_0.5'], 0.5)
        int_score_67 = int_score_67/m
        int_score_90 = int_score_90/m
        loss_median = loss_median/m
        return {'loss': loss_median+(int_score_67+int_score_90)/2, 'status': 'ok'}

    trials = Trials()
    best_params = fmin(
        fn = objective,
        space = search_space,
        algo = tpe.suggest,
        max_evals = int(0.3*N),
        trials=trials,
        early_stop_fn=no_progress_loss(stop)
    )

    lags = [int(best_params[f"p_{i}"])+1 for i in range(n)]
    difference = int(best_params['d'])
    ma = int(best_params['q'])

    forecaster_test = create_multivariate_forecaster(
        p = lags,
        d = difference,
        q = ma,
        model_type=model_type,
        steps = test_steps,
        level = countries_to_predict
    )

    forecaster_horizon = create_multivariate_forecaster(
        p = lags,
        d = difference,
        q = ma,
        model_type = model_type,
        steps = Constants.horizon,
        level = countries_to_predict
    )

    return forecaster_test, forecaster_horizon, best_params