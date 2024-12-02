from typing import Union, Optional, Callable, Tuple, Any

import sys

from copy import copy

import numpy as np

import pandas as pd

from joblib import Parallel, cpu_count, delayed

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from itertools import chain

import warnings

import skforecast
from skforecast.base import ForecasterBase
from skforecast.exceptions import DataTransformationWarning
from skforecast.preprocessing import TimeSeriesDifferentiator
from skforecast.utils import (
    check_exog,
    check_exog_dtypes,
    check_interval,
    check_select_fit_kwargs,
    check_predict_input,
    check_y,
    exog_to_direct,
    exog_to_direct_numpy,
    expand_index,
    get_exog_dtypes,
    initialize_lags,
    initialize_transformer_series,
    initialize_window_features,
    initialize_weights,
    input_to_frame,
    prepare_steps_direct,
    preprocess_last_window,
    preprocess_y,
    select_n_jobs_fit_forecaster,
    set_skforecast_warnings,
    transform_dataframe,
    transform_numpy,
    transform_series
)

class ForecastDirectMultiOutput(ForecasterBase):

    def __init__(
        self,
        regressor: object,
        levels: Union[str, list],
        steps: int,
        lags: Optional[Union[int, list, np.ndarray, range, dict]] = None,
        window_features: Optional[Union[object, list]] = None,
        transformer_series: Optional[Union[object, dict]] = StandardScaler(),
        transformer_exog: Optional[object] = None,
        weight_func: Optional[Callable] = None,
        differentiation: Optional[int] = None,
        fit_kwargs: Optional[dict] = None,
        n_jobs: Union[int, str] = 'auto',
        forecaster_id: Optional[Union[str, int]] = None
    ) -> None:
        
        self.regressor = copy(regressor)
        self.levels = levels
        self.steps = steps
        self.lags_ = None
        self.transformer_series = transformer_series
        self.transformer_series_ = None
        self.transformer_exog = transformer_exog
        self.weight_func = weight_func
        self.source_code_weight_func = None
        self.differentiation = differentiation
        self.differentiator = None
        self.differentiator_ = None
        self.last_window_ = None
        self.index_type_ = None
        self.index_freq_ = None
        self.training_range_ = None
        self.series_names_in_ = None
        self.exog_in_ = None
        self.exog_names_in_ = None
        self.exog_type_in_ = None
        self.exog_dtypes_in_ = None
        self.X_train_series_names_in_ = None
        self.X_train_window_features_names_out_ = None
        self.X_train_exog_names_out_ = None
        self.X_train_direct_exog_names_out_ = None
        self.X_train_features_names_out_ = None
        self.creation_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.is_fitted = False
        self.fit_date = None
        self.skforecast_version = skforecast.__version__
        self.python_version = sys.version.split(" ")[0]
        self.forecaster_id = forecaster_id
        self.dropna_from_series = False  # Ignored in this forecaster
        self.encoding = None   # Ignored in this forecaster

        if isinstance(levels, str):
            self.levels = [levels]
        elif isinstance(levels, list):
            self.levels = levels
        else:
            raise TypeError(
                f"`levels` argument must be a string or a list. Got {type(levels)}."
            )
        
        if not isinstance(steps, int):
            raise TypeError(
                f"`steps` argument must be an int greater than or equal to 1. "
                f"Got {type(steps)}."
            )

        if steps < 1:
            raise ValueError(
                f"`steps` argument must be greater than or equal to 1. Got {steps}."
            )
        
        self.regressors_ = {step: clone(self.regressor) for step in range(1, steps+1)}

        if isinstance(lags, dict):
            self.lags = {}
            self.lags_names = {}
            list_max_lags = []
            for key in lags:
                if lags[key] is None:
                    self.lags[key] = None
                    self.lags_names[key] = None
                else:
                    self.lags[key], lags_names, max_lag = initialize_lags(
                        forecaster_name = type(self).__name__,
                        lags            = lags[key]
                    )
                    self.lags_names[key] = (
                        [f'{key}_{lag}' for lag in lags_names] 
                         if lags_names is not None 
                         else None
                    )
                    if max_lag is not None:
                        list_max_lags.append(max_lag)
            
            self.max_lag = max(list_max_lags) if len(list_max_lags) != 0 else None
        else:
            self.lags, self.lags_names, self.max_lag = initialize_lags(
                forecaster_name = type(self).__name__, 
                lags            = lags
            )

        self.window_features, self.window_features_names, self.max_size_window_features = (
            initialize_window_features(window_features)
        )
        if self.window_features is None and (self.lags is None or self.max_lag is None):
            raise ValueError(
                "At least one of the arguments `lags` or `window_features` "
                "must be different from None. This is required to create the "
                "predictors used in training the forecaster."
            )
        
        self.window_size = max(
            [ws for ws in [self.max_lag, self.max_size_window_features] 
             if ws is not None]
        )
        self.window_features_class_names = None
        if window_features is not None:
            self.window_features_class_names = [
                type(wf).__name__ for wf in self.window_features
            ]

        if self.differentiation is not None:
            if not isinstance(differentiation, int) or differentiation < 1:
                raise ValueError(
                    f"Argument `differentiation` must be an integer equal to or "
                    f"greater than 1. Got {differentiation}."
                )
            self.window_size += self.differentiation
            self.differentiator = TimeSeriesDifferentiator(
                order=self.differentiation, window_size=self.window_size
            )
    
        self.weight_func, self.source_code_weight_func, _ = initialize_weights(
            forecaster_name = type(self).__name__, 
            regressor       = regressor, 
            weight_func     = weight_func, 
            series_weights  = None
        )

        self.fit_kwargs = check_select_fit_kwargs(
                              regressor  = regressor,
                              fit_kwargs = fit_kwargs
                          )
        
        self.in_sample_residuals_ = {step: None for step in range(1, self.steps + 1)}
        self.out_sample_residuals_ = None

        if n_jobs == 'auto':
            self.n_jobs = select_n_jobs_fit_forecaster(
                              forecaster_name = type(self).__name__,
                              regressor       = self.regressor
                          )
        else:
            if not isinstance(n_jobs, int):
                raise TypeError(
                    f"`n_jobs` must be an integer or `'auto'`. Got {type(n_jobs)}."
                )
            self.n_jobs = n_jobs if n_jobs > 0 else cpu_count()

      
    def _create_data_to_return_dict(self, series_names_in_: list) -> Tuple[dict, list]:
        
        if isinstance(self.lags, dict):
            lags_keys = list(self.lags.keys())
            if set(lags_keys) != set(series_names_in_):  # Set to avoid order
                raise ValueError(
                    (f"When `lags` parameter is a `dict`, its keys must be the "
                     f"same as `series` column names. If don't want to include lags, "
                      "add '{column: None}' to the lags dict.\n"
                     f"  Lags keys        : {lags_keys}.\n"
                     f"  `series` columns : {series_names_in_}.")
                )
            self.lags_ = copy(self.lags) 
        else:
            self.lags_ = {serie: self.lags for serie in series_names_in_}
            if self.lags is not None:
                # Defined `lags_names` here to avoid overwriting when fit and then create_train_X_y
                lags_names = [f'lag_{i}' for i in self.lags]
                self.lags_names = {
                    serie: [f'{serie}_{lag}' for lag in lags_names]
                    for serie in series_names_in_
                }
            else:
                self.lags_names = {serie: None for serie in series_names_in_}
        
        X_train_series_names_in_ = series_names_in_
        if self.lags is None:
            data_to_return_dict = {level: 'y' for level in self.levels}
        else:
            data_to_return_dict = {
                col: ('both' if col in self.levels else 'X')
                for col in series_names_in_
                if col in self.levels or self.lags_.get(col) is not None
            }

            for level in self.levels:
                if self.lags_.get(level) is None:
                    data_to_return_dict[level] = 'y'

            if self.window_features is None:
                # X_train_series_names_in_ include series that will be added to X_train
                X_train_series_names_in_ = [
                    col for col in data_to_return_dict.keys()
                    if data_to_return_dict[col] in ['X', 'both']
                ]
        return data_to_return_dict, X_train_series_names_in_
    
    def _create_lags(
        self, 
        y: np.ndarray,
        lags: np.ndarray,
        data_to_return: Optional[str] = 'both'
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:

        X_data = None
        y_data = None
        if data_to_return is not None:

            n_rows = len(y) - self.window_size - (self.steps - 1)

            if data_to_return != 'y':
                # If `data_to_return` is not 'y', it means is 'X' or 'both', X_data is created
                X_data = np.full(
                    shape=(n_rows, len(lags)), fill_value=np.nan, order='F', dtype=float
                )
                for i, lag in enumerate(lags):
                    X_data[:, i] = y[self.window_size - lag : -(lag + self.steps - 1)]

            if data_to_return != 'X':
                # If `data_to_return` is not 'X', it means is 'y' or 'both', y_data is created
                y_data = np.full(
                    shape=(n_rows, self.steps), fill_value=np.nan, order='F', dtype=float
                )
                for step in range(self.steps):
                    y_data[:, step] = y[self.window_size + step : self.window_size + step + n_rows]
        
        return X_data, y_data

    def _create_window_features(
        self, 
        y: pd.Series,
        train_index: pd.Index,
        X_as_pandas: bool = False,
    ) -> Tuple[list, list]:
        len_train_index = len(train_index)
        X_train_window_features = []
        X_train_window_features_names_out_ = []
        for wf in self.window_features:
            X_train_wf = wf.transform_batch(y)
            if not isinstance(X_train_wf, pd.DataFrame):
                raise TypeError(
                    (f"The method `transform_batch` of {type(wf).__name__} "
                     f"must return a pandas DataFrame.")
                )
            X_train_wf = X_train_wf.iloc[-len_train_index:]
            if not len(X_train_wf) == len_train_index:
                raise ValueError(
                    (f"The method `transform_batch` of {type(wf).__name__} "
                     f"must return a DataFrame with the same number of rows as "
                     f"the input time series - (`window_size` + (`steps` - 1)): {len_train_index}.")
                )
            X_train_wf.index = train_index
            
            X_train_wf.columns = [f'{y.name}_{col}' for col in X_train_wf.columns]
            X_train_window_features_names_out_.extend(X_train_wf.columns)
            if not X_as_pandas:
                X_train_wf = X_train_wf.to_numpy()     
            X_train_window_features.append(X_train_wf)

        return X_train_window_features, X_train_window_features_names_out_        

    def _create_train_X_y(
        self,
        series: pd.DataFrame,
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None
    ) -> Tuple[pd.DataFrame, dict, list, list, list, list, list, dict]:

        if not isinstance(series, pd.DataFrame):
            raise TypeError(
                f"`series` must be a pandas DataFrame. Got {type(series)}."
            )

        if len(series) < self.window_size + self.steps:
            raise ValueError(
                f"Minimum length of `series` for training this forecaster is "
                f"{self.window_size + self.steps}. Reduce the number of "
                f"predicted steps, {self.steps}, or the maximum "
                f"window_size, {self.window_size}, if no more data is available.\n"
                f"    Length `series`: {len(series)}.\n"
                f"    Max step : {self.steps}.\n"
                f"    Max window size: {self.window_size}.\n"
                f"    Lags window size: {self.max_lag}.\n"
                f"    Window features window size: {self.max_size_window_features}."
            )

        series_names_in_ = list(series.columns)

        if set(self.levels) - set(series_names_in_):
            raise ValueError(
                f"One of the `series` columns must be named as the `level` of the forecaster.\n"
                f"  Forecaster `level` : {self.level}.\n"
                f"  `series` columns   : {series_names_in_}."
            )
    
        data_to_return_dict, X_train_series_names_in_ = (
            self._create_data_to_return_dict(series_names_in_=series_names_in_)
        )

        series_to_create_autoreg_features_and_y = [
            col for col in series_names_in_
            if col in X_train_series_names_in_ + self.levels
        ]

        fit_transformer = False
        if not self.is_fitted:
            fit_transformer = True
            self.transformer_series_ = initialize_transformer_series(
                forecaster_name = type(self).__name__,
                series_names_in_ = series_to_create_autoreg_features_and_y,
                transformer_series = self.transformer_series
            )
        
        if self.differentiation is None:
            self.differentiator_ = {
                serie: None for serie in series_to_create_autoreg_features_and_y
            }
        else:
            if not self.is_fitted:
                self.differentiator_ = {
                    serie: copy(self.differentiator)
                    for serie in series_to_create_autoreg_features_and_y
                }
        
        exog_names_in_ = None
        exog_dtypes_in_ = None
        categorical_features = False
        if exog is not None:
            check_exog(exog=exog, allow_nan=True)
            exog = input_to_frame(data=exog, input_name='exog')
            
            series_index_no_ws = series.index[self.window_size:]
            len_series = len(series)
            len_series_no_ws = len_series - self.window_size
            len_exog = len(exog)
            if not len_exog == len_series and not len_exog == len_series_no_ws:
                raise ValueError(
                    f"Length of `exog` must be equal to the length of `series` (if "
                    f"index is fully aligned) or length of `seriesy` - `window_size` "
                    f"(if `exog` starts after the first `window_size` values).\n"
                    f"    `exog`                   : ({exog.index[0]} -- {exog.index[-1]})  (n={len_exog})\n"
                    f"    `series`                 : ({series.index[0]} -- {series.index[-1]})  (n={len_series})\n"
                    f"    `series` - `window_size` : ({series_index_no_ws[0]} -- {series_index_no_ws[-1]})  (n={len_series_no_ws})"
                )
            
            exog_names_in_ = exog.columns.to_list()
            if len(set(exog_names_in_) - set(series_names_in_)) != len(exog_names_in_):
                raise ValueError(
                    f"`exog` cannot contain a column named the same as one of "
                    f"the series (column names of series).\n"
                    f"  `series` columns : {series_names_in_}.\n"
                    f"  `exog`   columns : {exog_names_in_}."
                )
            
            # NOTE: Need here for filter_train_X_y_for_step to work without fitting
            self.exog_in_ = True
            exog_dtypes_in_ = get_exog_dtypes(exog=exog)

            exog = transform_dataframe(
                       df                = exog,
                       transformer       = self.transformer_exog,
                       fit               = fit_transformer,
                       inverse_transform = False
                   )
                
            check_exog_dtypes(exog, call_check_exog=True)
            categorical_features = (
                exog.select_dtypes(include=np.number).shape[1] != exog.shape[1]
            )

            # Use .index as series.index is not yet preprocessed with preprocess_y
            if len_exog == len_series:
                if not (exog.index == series.index).all():
                    raise ValueError(
                        "When `exog` has the same length as `series`, the index "
                        "of `exog` must be aligned with the index of `series` "
                        "to ensure the correct alignment of values."
                    )
                # The first `self.window_size` positions have to be removed from 
                # exog since they are not in X_train.
                exog = exog.iloc[self.window_size:, ]
            else:
                if not (exog.index == series_index_no_ws).all():
                    raise ValueError(
                        "When `exog` doesn't contain the first `window_size` "
                        "observations, the index of `exog` must be aligned with "
                        "the index of `series` minus the first `window_size` "
                        "observations to ensure the correct alignment of values."
                    )
        X_train_autoreg = []
        X_train_window_features_names_out_ = [] if self.window_features is not None else None
        X_train_features_names_out_ = []
        y_train = []
        for col in series_to_create_autoreg_features_and_y:
            y = series[col]
            check_y(y=y,series_id=f"Column '{col}'")
            y = transform_series(
                series = y,
                transformer = self.transformer_series_[col],
                fit = fit_transformer,
                inverse_transform = False 
            )
            y_values, y_index = preprocess_y(y=y)

            if self.differentiation is not None:
                if not self.is_fitted:
                    y_values = self.differentiator_[col].fit_transform(y_values)
                else:
                    differentiator = copy(self.differentiator_[col])
                    y_values = differentiator.fit_transform(y_values)
            
            X_train_autoreg_col = []
            train_index = y_index[self.window_size+(self.steps-1):]

            X_train_lags, y_train_values = self._create_lags(
                y=y_values, lags=self.lags_[col], data_to_return=data_to_return_dict.get(col, None)
            )

            if X_train_lags is not None:
                X_train_autoreg_col.append(X_train_lags)
                X_train_features_names_out_.extend(self.lags_names[col])
            
            if col in self.levels:
                y_train.append(y_train_values)

            if self.window_features is not None:
                n_diff = 0 if self.differentiation is None else self.differentiation
                end_wf = None if self.steps == 1 else -(self.steps - 1)
                y_window_features = pd.Series(
                    y_values[n_diff:end_wf], index=y_index[n_diff:end_wf], name=col
                )
                X_train_window_features, X_train_wf_names_out_ = (
                    self._create_window_features(
                        y=y_window_features, X_as_pandas=False, train_index=train_index
                    )
                )
                X_train_autoreg_col.extend(X_train_window_features)
                X_train_window_features_names_out_.extend(X_train_wf_names_out_)
                X_train_features_names_out_.extend(X_train_wf_names_out_)

            if X_train_autoreg_col:
                if len(X_train_autoreg_col) == 1:
                    X_train_autoreg_col = X_train_autoreg_col[0]
                else:
                    X_train_autoreg_col = np.concatenate(X_train_autoreg_col, axis=1)

                X_train_autoreg.append(X_train_autoreg_col)            

        y_train = np.array(y_train)
        print

        X_train = []
        len_train_index = len(train_index)
        if categorical_features:
            if len(X_train_autoreg) == 1:
                X_train_autoreg = X_train_autoreg[0]
            else:
                X_train_autoreg = np.concatenate(X_train_autoreg, axis=1)
            X_train_autoreg = pd.DataFrame(
                                  data    = X_train_autoreg,
                                  columns = X_train_features_names_out_,
                                  index   = train_index
                              )
            X_train.append(X_train_autoreg)
        else:
            X_train.extend(X_train_autoreg)
        
        self.X_train_window_features_names_out_ = X_train_window_features_names_out_

        X_train_exog_names_out_ = None
        if exog is not None:
            X_train_exog_names_out_ = exog.columns.to_list()
            if categorical_features:
                exog_direct, X_train_direct_exog_names_out_ = exog_to_direct(
                    exog=exog, steps=self.steps
                )
                exog_direct.index = train_index
            else:
                exog_direct, X_train_direct_exog_names_out_ = exog_to_direct_numpy(
                    exog=exog, steps=self.steps
                )

            self.X_train_direct_exog_names_out_ = X_train_direct_exog_names_out_

            X_train_features_names_out_.extend(self.X_train_direct_exog_names_out_)
            X_train.append(exog_direct)

        if len(X_train) == 1:
            X_train = X_train[0]
        else:
            if categorical_features:
                X_train = pd.concat(X_train, axis=1)
            else:
                X_train = np.concatenate(X_train, axis=1)
                
        if categorical_features:
            X_train.index = train_index
        else:
            X_train = pd.DataFrame(
                          data    = X_train,
                          index   = train_index,
                          columns = X_train_features_names_out_
                      )

        y_train = {
            step: pd.DataFrame(
                data=y_train[:,:, step-1].T,
                index=y_index[self.window_size+step-1:][:len_train_index],
                columns=self.levels
            )
            for step in range(1, self.steps+1)
        }
    
        return (
            X_train,
            y_train,
            series_names_in_,
            X_train_series_names_in_,
            exog_names_in_,
            X_train_exog_names_out_,
            X_train_features_names_out_,
            exog_dtypes_in_
        )

    def create_train_X_y(
        self,
        series: pd.DataFrame,
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
        suppress_warnings: bool = False
    ) -> Tuple[pd.DataFrame, dict]:
        set_skforecast_warnings(suppress_warnings, action='ignore')

        output = self._create_train_X_y(
                     series = series, 
                     exog   = exog
                 )

        X_train = output[0]
        y_train = output[1]
        
        set_skforecast_warnings(suppress_warnings, action='default')

        return X_train, y_train

    def filter_train_X_y_for_step(
        self,
        step: int,
        X_train: pd.DataFrame,
        y_train: dict,
        remove_suffix: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        if (step < 1) or (step > self.steps):
            raise ValueError(
                (f"Invalid value `step`. For this forecaster, minimum value is 1 "
                 f"and the maximum step is {self.steps}.")
            )

        y_train_step = y_train[step]
        if not self.exog_in_:
            X_train_step = X_train
        else:
            n_lags = len(list(
                chain(*[v for v in self.lags_.values() if v is not None])
            ))
            n_window_features = (
                len(self.X_train_window_features_names_out_) if self.window_features is not None else 0
            )
            idx_columns_autoreg = np.arange(n_lags + n_window_features)
            n_exog = len(self.X_train_direct_exog_names_out_) / self.steps
            idx_columns_exog = (
                np.arange((step - 1) * n_exog, (step) * n_exog) + idx_columns_autoreg[-1] + 1 
            )
            idx_columns = np.concatenate((idx_columns_autoreg, idx_columns_exog))
            X_train_step = X_train.iloc[:, idx_columns]

        if remove_suffix:
            X_train_step.columns = [
                col_name.replace(f"_step_{step}", "")
                for col_name in X_train_step.columns
            ]
            y_train_step.name = y_train_step.name.replace(f"_step_{step}", "")

        return X_train_step, y_train_step

    def create_sample_weights(
        self,
        X_train: pd.DataFrame
    ) -> np.ndarray:
        
        sample_weight = None

        if self.weight_func is not None:
            sample_weight = self.weight_func(X_train.index)

        if sample_weight is not None:
            if np.isnan(sample_weight).any():
                raise ValueError(
                    "The resulting `sample_weight` cannot have NaN values."
                )
            if np.any(sample_weight < 0):
                raise ValueError(
                    "The resulting `sample_weight` cannot have negative values."
                )
            if np.sum(sample_weight) == 0:
                raise ValueError(
                    ("The resulting `sample_weight` cannot be normalized because "
                     "the sum of the weights is zero.")
                )

        return sample_weight        

    def fit(
        self,
        series: pd.DataFrame,
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
        store_last_window: bool = True,
        store_in_sample_residuals: bool = True,
        suppress_warnings: bool = False
    ) -> None:
        
        set_skforecast_warnings(suppress_warnings, action='ignore')

        self.lags_ = None
        self.last_window_ = None
        self.index_type_ = None
        self.index_freq_ = None
        self.training_range_ = None
        self.series_names_in_ = None
        self.exog_in_ = False
        self.exog_names_in_ = None
        self.exog_type_in_ = None
        self.exog_dtypes_in_ = None
        self.X_train_series_names_in_ = None
        self.X_train_window_features_names_out_ = None
        self.X_train_exog_names_out_ = None
        self.X_train_direct_exog_names_out_ = None
        self.X_train_features_names_out_ = None
        self.in_sample_residuals_ = {step: None for step in range(1, self.steps+1)}
        self.is_fitted = False
        self.fit_date = None

        (
            X_train,
            y_train,
            series_names_in_,
            X_train_series_names_in_,
            exog_names_in_,
            X_train_exog_names_out_,
            X_train_features_names_out_,
            exog_dtypes_in_
        ) = self._create_train_X_y(series=series, exog=exog)        

        def fit_forecaster(regressor, X_train, y_train, step, store_in_sample_residuals):
            X_train_step, y_train_step = self.filter_train_X_y_for_step(step, X_train, y_train)
            sample_weight = self.create_sample_weights(X_train=X_train_step)
            if sample_weight is not None:
                regressor.fit(X = X_train_step, y=y_train_step, sample_weight=sample_weight **self.fit_kwargs)
            else:
                regressor.fit(X=X_train_step, y=y_train_step, **self.fit_kwargs)

            if store_in_sample_residuals:
                residuals = (
                    (y_train_step - regressor.predict(X_train_step))
                ).to_numpy()

            else:
                residuals = None
                
            return step, regressor, residuals        

        results_fit = (
            Parallel(n_jobs = self.n_jobs)
            (delayed(fit_forecaster)
            (
                regressor = copy(self.regressor),
                X_train = X_train,
                y_train = y_train,
                step = step,
                store_in_sample_residuals = store_in_sample_residuals
            )
            for step in range(1, self.steps+1))
        )

        self.regressors_ = {step: regressor for step, regressor, _ in results_fit}

        if store_in_sample_residuals:
            self.in_sample_residuals_ = {step: residuals for step, _, residuals in results_fit}

        self.series_names_in_ = series_names_in_
        self.X_train_series_names_in_ = X_train_series_names_in_
        self.X_train_features_names_out_ = X_train_features_names_out_

        self.is_fitted = True
        self.fit_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.training_range_ = preprocess_y(y=series[self.levels[0]], return_values=False)[1][[0, -1]]
        self.index_type_ = type(X_train.index)
        if isinstance(X_train.index, pd.DatetimeIndex):
            self.index_freq_ = X_train.index.freqstr
        else:
            self.index_freq_ = X_train.index.step

        if exog is not None:
            self.exog_in_ = True
            self.exog_names_in_ = exog_names_in_
            self.exog_type_in_ = type(exog)
            self.exog_dtypes_in_ = exog_dtypes_in_
            self.X_train_exog_names_out_ = X_train_exog_names_out_

        if store_last_window:
            self.last_window_ = series.iloc[-self.window_size:, ][
                self.X_train_series_names_in_
            ].copy()
        
        set_skforecast_warnings(suppress_warnings, action='default')

    def _create_predict_inputs(
        self,
        steps: Optional[Union[int, list]] = None,
        last_window: Optional[pd.DataFrame] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
        check_inputs: bool = True
    ) -> Tuple[list, list, list, pd.Index]:

        steps = prepare_steps_direct(
                    steps    = steps,
                    max_step = self.steps
                )

        if last_window is None:
            last_window = self.last_window_
        
        if check_inputs:
            check_predict_input(
                forecaster_name  = type(self).__name__,
                steps            = steps,
                is_fitted        = self.is_fitted,
                exog_in_         = self.exog_in_,
                index_type_      = self.index_type_,
                index_freq_      = self.index_freq_,
                window_size      = self.window_size,
                last_window      = last_window,
                exog             = exog,
                exog_type_in_    = self.exog_type_in_,
                exog_names_in_   = self.exog_names_in_,
                interval         = None,
                max_steps        = self.steps,
                series_names_in_ = self.X_train_series_names_in_
            )

        last_window = last_window.iloc[
            -self.window_size:, last_window.columns.get_indexer(self.X_train_series_names_in_)
        ].copy()
        
        X_autoreg = []
        Xs_col_names = []
        for serie in self.X_train_series_names_in_:
            last_window_serie = transform_numpy(
                                    array             = last_window[serie].to_numpy(),
                                    transformer       = self.transformer_series_[serie],
                                    fit               = False,
                                    inverse_transform = False
                                )
            
            if self.differentiation is not None:
                last_window_serie = self.differentiator_[serie].fit_transform(last_window_serie)

            if self.lags is not None:
                X_lags = last_window_serie[-self.lags_[serie]]
                X_autoreg.append(X_lags)
                Xs_col_names.extend(self.lags_names[serie])

            if self.window_features is not None:
                n_diff = 0 if self.differentiation is None else self.differentiation
                X_window_features = np.concatenate(
                    [
                        wf.transform(last_window_serie[n_diff:]) 
                        for wf in self.window_features
                    ]
                )
                X_autoreg.append(X_window_features)
                Xs_col_names.extend([f"{serie}_{wf}" for wf in self.window_features_names])
            
        X_autoreg = np.concatenate(X_autoreg).reshape(1, -1)
        _, last_window_index = preprocess_last_window(
            last_window=last_window, return_values=False
        )
        if exog is not None:
            exog = input_to_frame(data=exog, input_name='exog')
            exog = exog.loc[:, self.exog_names_in_]
            exog = transform_dataframe(
                       df                = exog,
                       transformer       = self.transformer_exog,
                       fit               = False,
                       inverse_transform = False
                   )
            check_exog_dtypes(exog=exog)
            exog_values, _ = exog_to_direct_numpy(
                                 exog  = exog.to_numpy()[:max(steps)],
                                 steps = max(steps)
                             )
            exog_values = exog_values[0]
            
            n_exog = exog.shape[1]
            Xs = [
                np.concatenate(
                    [
                        X_autoreg, 
                        exog_values[(step - 1) * n_exog : step * n_exog].reshape(1, -1)
                    ],
                    axis=1
                )
                for step in steps
            ]
            Xs_col_names = Xs_col_names + exog.columns.to_list()
        else:
            Xs = [X_autoreg] * len(steps)

        prediction_index = expand_index(
                               index = last_window_index,
                               steps = max(steps)
                           )[np.array(steps) - 1]
        if isinstance(last_window_index, pd.DatetimeIndex) and np.array_equal(
            steps, np.arange(min(steps), max(steps) + 1)
        ):
            prediction_index.freq = last_window_index.freq

        return Xs, Xs_col_names, steps, prediction_index

                        
    def create_predict_X(
        self,
        steps: Optional[Union[int, list]] = None,
        last_window: Optional[pd.DataFrame] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
        suppress_warnings: bool = False
    ) -> pd.DataFrame:
        
        set_skforecast_warnings(suppress_warnings, action='ignore')

        Xs, Xs_col_names, steps, prediction_index = self._create_predict_inputs(
            steps=steps, last_window=last_window, exog=exog
        )

        X_predict = pd.DataFrame(
                        data    = np.concatenate(Xs, axis=0), 
                        columns = Xs_col_names, 
                        index   = prediction_index
                    )

        if self.transformer_series is not None or self.differentiation is not None:
            warnings.warn(
                "The output matrix is in the transformed scale due to the "
                "inclusion of transformations or differentiation in the Forecaster. "
                "As a result, any predictions generated using this matrix will also "
                "be in the transformed scale. Please refer to the documentation "
                "for more details: "
                "https://skforecast.org/latest/user_guides/training-and-prediction-matrices.html",
                DataTransformationWarning
            )

        set_skforecast_warnings(suppress_warnings, action='default')

        return X_predict                            

    def predict(
        self,
        steps: Optional[Union[int, list]] = None,
        last_window: Optional[pd.DataFrame] = None,
        exog: Optional[Union[pd.Series, pd.DataFrame]] = None,
        suppress_warnings: bool = False,
        check_inputs: bool = True,
        levels: Any = None
    ) -> pd.DataFrame:
        
        set_skforecast_warnings(suppress_warnings, action='ignore')

        Xs, _, steps, prediction_index = self._create_predict_inputs(
            steps=steps, last_window=last_window, exog=exog, check_inputs=check_inputs
        )

        regressors = [self.regressors_[step] for step in steps]

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", 
                message="X does not have valid feature names", 
                category=UserWarning
            )
            predictions = np.array([
                regressor.predict(X).ravel()
                for regressor, X in zip(regressors, Xs)
            ])

        for i, level in enumerate(self.levels):
            if self.differentiation is not None:
                predictions[i] = self.differentiator_[
                    level
                ].inverse_transform_next_window(predictions[i])
            predictions[i] = transform_numpy(
                array = predictions[i],
                transformer = self.transformer_series_[level],
                fit = False,
                inverse_transform = True 
            )

        predictions = pd.DataFrame(
            data = predictions,
            columns = self.levels,
            index = prediction_index
        )

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
            steps = prepare_steps_direct(steps=steps, max_step=self.steps)

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

        Xs, _, steps, prediction_index = self._create_predict_inputs(
            steps=steps, last_window=last_window, exog=exog
        )

        regressors = [self.regressors_[step] for step in steps]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", 
                message="X does not have valid feature names", 
                category=UserWarning
            )
            predictions = np.array([
                regressor.predict(X).ravel()
                for regressor, X in zip(regressors, Xs)
            ])

        boot_predictions = {}
        boot_columns = [f"pred_boot_{i}" for i in range(n_boot)]
        rng = np.random.default_rng(seed=random_state)

        for j, level in enumerate(self.levels):
            boot_level = np.tile(predictions[:,j], (n_boot, 1)).T

            for i, step in enumerate(steps):
                sampled_residuals = residuals[step][rng.integers(low=0, high=len(residuals[step]), size=n_boot), j]
                boot_level[i, :] += sampled_residuals
            
            if self.differentiation is not None:
                boot_level = self.differentiator_[level].inverse_transform_next_window(boot_level)
            
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

        preds = self.predict(steps=steps, last_window=last_window, exog=exog, check_inputs=False)

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
        self.regressor.set_params(**params)
        self.regressors_ = {step: clone(self.regressor)
                            for step in range(1, self.steps + 1)}