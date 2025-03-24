from typing import Any, Callable, Optional, Tuple, Union

from copy import copy, deepcopy

import keras
from keras.models import clone_model # type: ignore
from keras.optimizers import Adam # type: ignore
from keras.losses import MeanSquaredError # type: ignore

import numpy as np

import pandas as pd

import sys

from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler

import warnings

import skforecast
from skforecast.base import ForecasterBase
from skforecast.exceptions import IgnoredArgumentWarning
from skforecast.preprocessing import TimeSeriesDifferentiator
from skforecast.utils import (
    check_interval,
    check_select_fit_kwargs,
    check_predict_input,
    check_y,
    expand_index,
    set_skforecast_warnings,
    transform_numpy,
    transform_series,
    preprocess_last_window,
    preprocess_y
)

class ForecastRNNProb(ForecasterBase):

    def __init__(
        self,
        regressor: object,
        levels: Union[str, list],
        lags: Optional[Union[int, list, str]] = "auto",
        steps: Optional[Union[int, list, str]] = "auto",
        transformer_series: Optional[Union[object, dict]] = MinMaxScaler(
            feature_range=(0, 1)
        ),
        weight_func: Optional[Callable] = None,
        fit_kwargs: Optional[dict] = {},
        forecaster_id: Optional[Union[str, int]] = None,
        n_jobs: Any = None,
        transformer_exog: Any = None
    ) -> None:
        
        self.levels = levels
        self.transformer_series = transformer_series
        self.transformer_series_ = None
        self.transforemr_exog = None
        self.weight_func = weight_func
        self.source_code_weight_func = None
        self.max_lag = None
        self.window_size = None
        self.index_type_ = None
        self.index_freq_ = None
        self.training_range_ = None
        self.exog_in_ = False
        self.exog_type_in_ = None
        self.exog_dtypes_in_ = None
        self.exog_names_in_ = None
        self.series_names_in_ = None
        self.X_train_dim_names_ = None
        self.y_train_dim_names_ = None
        self.is_fitted = False
        self.creation_date = pd.Timestamp.today().strftime("%Y-%m-%d %H:%M:%S")
        self.fit_date = None
        self.skforecast_version = skforecast.__version__
        self.python_version = sys.version.split(" ")[0]
        self.forecaster_id = forecaster_id
        self.history = None
        self.dropna_from_series = False
        self.encoding = None
        self.differentiation = None
        self.differentiator = None
        self.differentiator_ = None

        self.regressor = clone_model(regressor)
        self.regressor.compile(optimizer=Adam(learning_rate=0.01), loss=MeanSquaredError())
        layer_init = self.regressor.layers[0]

        if lags == "auto":
            if keras.__version__ < "3.0":
                self.lags = np.arange(layer_init.input_shape[0][1]) + 1
            else:
                self.lags = np.arange(layer_init.output.shape[1]) + 1

            warnings.warn(
                "Setting `lags` = 'auto'. `lags` are inferred from the regressor " 
                "architecture. Avoid the warning with lags=lags."
            )
        elif isinstance(lags, int):
            self.lags = np.arange(lags) + 1
        elif isinstance(lags, list):
            self.lags = np.array(lags)
        else:
            raise TypeError(
                f"`lags` argument must be an int, list or 'auto'. Got {type(lags)}."
            )

        self.max_lag = np.max(self.lags)
        self.window_size = self.max_lag

        layer_end = self.regressor.layers[-1]

        try:
            if keras.__version__ < "3.0":
                self.series = layer_end.output_shape[-1]
            else:
                self.series = layer_end.output.shape[-1]

        except:
            raise TypeError(
                "Input shape of the regressor should be Input(shape=(lags, n_series))."
            )
        
        if steps == "auto":
            if keras.__version__ < "3.0":
                self.steps = np.arange(layer_end.output_shape[1]) + 1
            else:
                self.steps = np.arange(layer_end.output.shape[1]) + 1
            warnings.warn(
                "`steps` default value = 'auto'. `steps` inferred from regressor "
                "architecture. Avoid the warning with steps=steps."
            )
        elif isinstance(steps, int):
            self.steps = np.arange(steps) + 1
        elif isinstance(steps, list):
            self.steps = np.array(steps)
        else:
            raise TypeError(
                f"`steps` argument must be an int, list or 'auto'. Got {type(steps)}."
            )
        
        self.max_step = np.max(self.steps)
        if keras.__version__ < "3.0":
            self.outputs = layer_end.output_shape[-1]
        else:
            self.outputs = layer_end.output.shape[-1]

        if not isinstance(levels, (list, str, type(None))):
            raise TypeError(
                f"`levels` argument must be a string, list or. Got {type(levels)}."
            )

        if isinstance(levels, str):
            self.levels = [levels]
        elif isinstance(levels, list):
            self.levels = levels
        else:
            raise TypeError(
                f"`levels` argument must be a string or a list. Got {type(levels)}."
            )

        self.series_val = None
        if "series_val" in fit_kwargs:
            self.series_val = fit_kwargs["series_val"]
            fit_kwargs.pop("series_val")

        self.in_sample_residuals_ = {step: None for step in self.steps}
        self.out_sample_residuals_ = None

        self.fit_kwargs = check_select_fit_kwargs(
            regressor=self.regressor, fit_kwargs=fit_kwargs
        )

    def _create_lags(
        self,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_splits = len(y)-self.max_lag-self.max_step + 1
        if n_splits <= 0:
            raise ValueError(
                (
                    f"The maximum lag ({self.max_lag}) must be less than the length "
                    f"of the series minus the maximum of steps ({len(y) - self.max_step})."
                )
            )

        X_data = np.full(
            shape=(n_splits, (self.max_lag)), fill_value=np.nan, order='F', dtype=float
        )
        for i, lag in enumerate(range(self.max_lag - 1, -1, -1)):
            X_data[:, i] = y[self.max_lag - lag - 1 : -(lag + self.max_step)]

        y_data = np.full(
            shape=(n_splits, self.max_step), fill_value=np.nan, order='F', dtype=float
        )
        for step in range(self.max_step):
            y_data[:, step] = y[self.max_lag + step : self.max_lag + step + n_splits]

        X_data = X_data[:, self.lags - 1]

        y_data = y_data[:, self.steps - 1]

        return X_data, y_data

    def create_train_X_y(
        self, series: pd.DataFrame, exog: Any = None
    ) -> Tuple[np.ndarray, np.ndarray, dict]:

        if not isinstance(series, pd.DataFrame):
            raise TypeError(f"`series` must be a pandas DataFrame. Got {type(series)}.")

        series_names_in_ = list(series.columns)

        if not set(self.levels).issubset(set(series.columns)):
            raise ValueError(
                (
                    f"`levels` defined when initializing the forecaster must be included "
                    f"in `series` used for trainng. {set(self.levels) - set(series.columns)} "
                    f"not found."
                )
            )

        if len(series) < self.max_lag + self.max_step:
            raise ValueError(
                (
                    f"Minimum length of `series` for training this forecaster is "
                    f"{self.max_lag + self.max_step}. Got {len(series)}. Reduce the "
                    f"number of predicted steps, {self.max_step}, or the maximum "
                    f"lag, {self.max_lag}, if no more data is available."
                )
            )

        if self.transformer_series is None:
            self.transformer_series_ = {serie: None for serie in series_names_in_}
        elif not isinstance(self.transformer_series, dict):
            self.transformer_series_ = {
                serie: clone(self.transformer_series) for serie in series_names_in_
            }
        else:
            self.transformer_series_ = {serie: None for serie in series_names_in_}
            # Only elements already present in transformer_series_ are updated
            self.transformer_series_.update(
                (k, v)
                for k, v in deepcopy(self.transformer_series).items()
                if k in self.transformer_series_
            )
            series_not_in_transformer_series = set(series.columns) - set(
                self.transformer_series.keys()
            )
            if series_not_in_transformer_series:
                warnings.warn(
                    (
                        f"{series_not_in_transformer_series} not present in "
                        f"`transformer_series`. No transformation is applied to "
                        f"these series."
                    ),
                    IgnoredArgumentWarning,
                )
        
        if self.differentiation is None:
            self.differentiator_ = {
                serie: None for serie in series_names_in_
            }
        else:
            if not self.is_fitted:
                self.differentiator_ = {
                    serie: copy(self.differentiator)
                    for serie in series_names_in_
                }

        X_train = []
        y_train = []

        for i, serie in enumerate(series.columns):
            x = series[serie]
            check_y(y=x)
            x = transform_series(
                series=x,
                transformer=self.transformer_series_[serie],
                fit=True,
                inverse_transform=False,
            )

            X, _ = self._create_lags(x)
            X_train.append(X)

        for i, serie in enumerate(self.levels):
            y = series[serie]
            check_y(y=y)
            y = transform_series(
                series=y,
                transformer=self.transformer_series_[serie],
                fit=True,
                inverse_transform=False,
            )

            _, y = self._create_lags(y)
            y_train.append(y)
    
        X_train = np.stack(X_train, axis=2)
        y_train = np.stack(y_train, axis=2)

        train_index = series.index.to_list()[
            self.max_lag : (len(series.index.to_list()) - self.max_step + 1)
        ]
        dimension_names = {
            "X_train": {
                0: train_index,
                1: ["lag_" + str(l) for l in self.lags],
                2: series.columns.to_list(),
            },
            "y_train": {
                0: train_index,
                1: ["step_" + str(l) for l in self.steps],
                2: self.levels,
            },
        }

        return X_train, y_train, dimension_names
    
    def fit(
        self,
        series: pd.DataFrame,
        store_in_sample_residuals: bool = True,
        exog: Any = None,
        suppress_warnings: bool = False,
        store_last_window: str = "Ignored",
    ) -> None:
        
        set_skforecast_warnings(False, action='ignore')

        self.index_type_ = None
        self.index_freq_ = None
        self.last_window_ = None
        self.exog_in_ = None
        self.exog_type_in_ = None
        self.exog_dtypes_in_ = None
        self.exog_names_in_ = None
        self.X_train_dim_names_ = None
        self.y_train_dim_names_ = None
        self.in_sample_residuals_ = None
        self.is_fitted = False
        self.training_range_ = None

        self.series_names_in_ = list(series.columns)

        X_train, y_train, X_train_dim_names_ = self.create_train_X_y(series=series)
        self.X_train_dim_names_ = X_train_dim_names_['X_train']
        self.y_train_dim_names_ = X_train_dim_names_['y_train']

        if self.series_val is not None:
            X_val, y_val, _ = self.create_train_X_y(series=self.series_val)
            history = self.regressor.fit(
                x=X_train, y=y_train, validation_data=(X_val, y_val), verbose=0, epochs=100, **self.fit_kwargs
            )
        else:
            history = self.regressor.fit(
                x=X_train, y=y_train, verbose=0, epochs=100, **self.fit_kwargs
            )
        self.history=history.history
        self.is_fitted=True
        self.fit_date=pd.Timestamp.today().strftime("%Y-%m-%d %H:%M:%S")

        _, y_index= preprocess_y(y=series[self.levels], return_values=False)
        self.training_range_ = y_index[[0,-1]]
        self.index_type_ = type(y_index)
        if isinstance(y_index, pd.DatetimeIndex):
            self.index_freq_ = y_index.freqstr
        else:
            self.index_freq_ = y_index.step
        
        self.last_window_ = series.iloc[-self.max_lag:].copy()

        if store_in_sample_residuals:
            residuals = y_train-self.regressor.predict(x=X_train, verbose=0)
            self.in_sample_residuals_ = {step: residuals[:,i,:] for i, step in enumerate(self.steps)}

        return
    
    def _create_predict_inputs(
        self,
        steps: Optional[Union[int, list]] = None,
        last_window: Optional[pd.DataFrame] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
        check_inputs: bool = True
    ) -> Tuple[np.ndarray, pd.Index]:
        
        if last_window is None:
            last_window = self.last_window_
        
        if check_inputs:
            check_predict_input(
                forecaster_name=type(self).__name__,
                steps=steps,
                is_fitted=self.is_fitted,
                exog_in_=self.exog_in_,
                index_type_=self.index_type_,
                index_freq_=self.index_freq_,
                window_size=self.window_size,
                last_window=last_window,
                exog=None,
                exog_type_in_=None,
                exog_names_in_=None,
                interval=None,
                max_steps=self.max_step,
                levels=self.levels,
                levels_forecaster=self.levels,
                series_names_in_=self.series_names_in_,
            )

        last_window = last_window.iloc[-self.window_size :,].copy()

        for serie_name in self.series_names_in_:
            last_window_serie = transform_series(
                series=last_window[serie_name],
                transformer=self.transformer_series_[serie_name],
                fit=False,
                inverse_transform=False,
            )

            last_window_values, last_window_index = preprocess_last_window(
                last_window=last_window_serie
            )
            last_window.loc[:, serie_name] = last_window_values

        X = np.reshape(last_window.to_numpy(), (1, self.max_lag, last_window.shape[1]))

        idx = expand_index(index=last_window_index, steps=max(steps))

        return X, idx


    def predict(
        self,
        steps: Optional[Union[int, list]] = None,
        levels: Optional[Union[str, list]] = None,
        last_window: Optional[pd.DataFrame] = None,
        exog: Any = None,
        suppress_warnings: bool = False
    ) -> pd.DataFrame:
        set_skforecast_warnings(suppress_warnings, action='ignore')
                
        if levels is None:
            levels = self.levels
        elif isinstance(levels, str):
            levels = [levels]
        if isinstance(steps, int):
            steps = list(np.arange(steps) + 1)
        elif steps is None:
            if isinstance(self.steps, int):
                steps = list(np.arange(self.steps) + 1)
            elif isinstance(self.steps, (list, np.ndarray)):
                steps = list(np.array(self.steps))
        elif isinstance(steps, list):
            steps = list(np.array(steps))

        for step in steps:
            if not isinstance(step, (int, np.int64, np.int32)):
                raise TypeError(
                    (
                        f"`steps` argument must be an int, a list of ints or `None`. "
                        f"Got {type(steps)}."
                    )
                )

        X, idx =  self._create_predict_inputs(steps=steps, last_window=last_window)
        predictions = self.regressor.predict(X, verbose=0)
        predictions_reshaped = np.reshape(
            predictions, (predictions.shape[1], predictions.shape[2])
        )

        predictions = pd.DataFrame(
            data=predictions_reshaped[np.array(steps) - 1],
            columns=self.levels,
            index=idx[np.array(steps) - 1],
        )
        predictions = predictions[levels]    

        for serie in levels:
            x = predictions[serie]
            check_y(y=x)
            x = transform_series(
                series=x,
                transformer=self.transformer_series_[serie],
                fit=False,
                inverse_transform=True,
            )
            predictions.loc[:, serie] = x

        set_skforecast_warnings(suppress_warnings, action='default')
                
        return predictions

    def predict_bootstrapping(
        self,
        steps: Optional[Union[int, list]] = None,
        last_window: Optional[pd.DataFrame] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
        n_boot: int = 250,
        random_state: int = 123,
        use_in_sample_residuals: bool = True,
        suppress_warnings: bool = False,
        levels: Any = None
    ) -> dict:
        
        set_skforecast_warnings(suppress_warnings, action='ignore')

        if self.is_fitted:
            steps = self.steps

            if use_in_sample_residuals:
                if not set(steps).issubset(set(self.in_sample_residuals_.keys())):
                    raise ValueError(
                        f"Not `forecaster.in_sample_residuals_` for steps: "
                        f"{set(steps) - set(self.in_sample_residuals_.keys())}."
                    )
                residuals = self.in_sample_residuals_
            else:
                if self.out_sample_residuals_ is None:
                    raise ValueError(
                        "`forecaster.out_sample_residuals_` is `None`. Use "
                        "`use_in_sample_residuals=True` or the "
                        "`set_out_sample_residuals()` method before predicting."
                    )
                else:
                    if not set(steps).issubset(set(self.out_sample_residuals_.keys())):
                        raise ValueError(
                            f"Not `forecaster.out_sample_residuals_` for steps: "
                            f"{set(steps) - set(self.out_sample_residuals_.keys())}. "
                            f"Use method `set_out_sample_residuals()`."
                        )
                residuals = self.out_sample_residuals_              
            
            check_residuals = (
                'forecaster.in_sample_residuals_' if use_in_sample_residuals
                else 'forecaster.out_sample_residuals_'
            )
            for step in steps:
                if residuals[step] is None:
                    raise ValueError(
                        f"forecaster residuals for step {step} are `None`. "
                        f"Check {check_residuals}."
                    )
                elif (any(element is None for element in residuals[step]) or
                      np.any(np.isnan(residuals[step]))):
                    raise ValueError(
                        f"forecaster residuals for step {step} contains `None` "
                        f"or `NaNs` values. Check {check_residuals}."
                    )

        X, prediction_index = self._create_predict_inputs(steps=steps, last_window=last_window)

        predictions = self.regressor.predict(X, verbose=0)
        predictions = np.squeeze(predictions, axis=0)

        boot_predictions = {}
        boot_columns = [f"pred_boot_{i}" for i in range(n_boot)]
        rng = np.random.default_rng(seed=random_state)

        for j, level in enumerate(self.levels):
            boot_level = np.tile(predictions[:,j], (n_boot, 1)).T

            for i, step in enumerate(steps):
                sampled_residuals = residuals[step][rng.integers(low=0, high=len(residuals[step]), size=n_boot), j]
                boot_level[i, :] += sampled_residuals

            if self.transformer_series_[level]:
                boot_level = np.apply_along_axis(
                    func1d = transform_numpy,
                    axis = 0,
                    arr = boot_level,
                    transformer = self.transformer_series_[level],
                    fit = False,
                    inverse_transform = True
                )

            boot_level = pd.DataFrame(
                data = boot_level,
                index=prediction_index,
                columns = boot_columns
            )

            boot_predictions[level] = boot_level

        set_skforecast_warnings(suppress_warnings, action='default')

        return boot_predictions

    def predict_interval(
        self,
        steps: Optional[Union[int, list]] = None,
        last_window: Optional[pd.DataFrame] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
        interval: list = [5, 95],
        n_boot: int = 250,
        random_state: int = 123,
        use_in_sample_residuals: bool = True,
        suppress_warnings: bool = False,
        levels: Any = None
    ) -> pd.DataFrame:
        
        set_skforecast_warnings(suppress_warnings, action='ignore')

        check_interval(interval=interval)

        boot_predictions = self.predict_bootstrapping(
                               steps                   = steps,
                               last_window             = last_window,
                               exog                    = exog,
                               n_boot                  = n_boot,
                               random_state            = random_state,
                               use_in_sample_residuals = use_in_sample_residuals
                           )

        preds = self.predict(steps=steps, last_window=last_window, exog=exog)

        interval = np.array(interval) / 100
        predictions = []        

        for level in preds.columns:
            preds_interval = boot_predictions[level].quantile(q=interval, axis=1).transpose()
            preds_interval.columns = [f'{level}_lower_bound', f'{level}_upper_bound']
            predictions.append(preds[level])
            predictions.append(preds_interval)  

        predictions = pd.concat(predictions, axis=1)
        
        set_skforecast_warnings(suppress_warnings, action='default')

        return predictions

    def predict_quantiles(
        self,
        steps: Optional[Union[int, list]] = None,
        last_window: Optional[pd.DataFrame] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
        quantiles: list = [0.05, 0.5, 0.95],
        n_boot: int = 250,
        random_state: int = 123,
        use_in_sample_residuals: bool = True,
        suppress_warnings: bool = False,
        levels: Any = None
    ) -> pd.DataFrame:
        
        set_skforecast_warnings(suppress_warnings, action='ignore')

        check_interval(quantiles=quantiles)

        boot_predictions = self.predict_bootstrapping(
                               steps                   = steps,
                               last_window             = last_window,
                               exog                    = exog,
                               n_boot                  = n_boot,
                               random_state            = random_state,
                               use_in_sample_residuals = use_in_sample_residuals
                           )
        
        predictions = []

        for level in boot_predictions.keys():
            preds_quantiles = (
                boot_predictions[level].quantile(q=quantiles, axis=1).transpose()
            )
            preds_quantiles.columns = [f'{level}_q_{q}' for q in quantiles]
            predictions.append(preds_quantiles)
        
        predictions = pd.concat(predictions, axis=1)

        set_skforecast_warnings(suppress_warnings, action='default')

        return predictions

    def set_params(
        self, 
        params: dict
    ) -> None:
        self.regressor = clone(self.regressor)
        self.regressor.reset_states()
        self.regressor.compile(**params)