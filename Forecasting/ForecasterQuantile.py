from typing import Callable

import copy

import sys

import inspect

import warnings

import numpy as np

import pandas as pd

import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

import skforecast
from skforecast.base import ForecasterBase
from skforecast.exceptions import DataTransformationWarning, MissingValuesWarning, IgnoredArgumentWarning
from skforecast.preprocessing import TimeSeriesDifferentiator
from skforecast.utils import (
    align_series_and_exog_multiseries,
    check_exog_dtypes,
    check_select_fit_kwargs,
    check_predict_input,
    check_preprocess_series,
    check_preprocess_exog_multiseries,
    expand_index,
    get_exog_dtypes,
    initialize_lags,
    initialize_differentiator_multiseries,
    initialize_transformer_series,
    initialize_weights,
    initialize_window_features,
    preprocess_last_window,
    prepare_levels_multiseries,
    preprocess_levels_self_last_window_multiseries,
    set_cpu_gpu_device,
    set_skforecast_warnings,
    transform_dataframe,
    transform_numpy
)

class ForecasterMultiSeriesQuantile(ForecasterBase):

    def __init__(
        self,
        quantiles: int | list[int],
        levels: list[str],
        lags: int | list[int] | np.ndarray | range | None = None,
        window_features: object | list[object] | None = None,
        encoding: str | None = 'ordinal',
        transformer_series: object | dict[str, object] | None = None,
        transformer_exog: object | None = None,
        weight_func: Callable | dict[str, Callable] | None = None,
        series_weights: dict[str, float] | None = None,
        differentiation: int | dict[str, int | None] | None = None,
        dropna_from_series: bool = False,
        fit_kwargs: dict[str, object] | None = None,
        binner_kwargs: dict[str, object] | None = None,
        forecaster_id: str | int | None = None,
        regressor: str = 'GradientBoostingRegressor'
    ) -> None:
        self.quantiles                          = quantiles
        self.encoding                           = encoding
        self.encoder                            = None
        self.encoding_mapping_                  = {}
        self.transformer_series                 = transformer_series
        self.transformer_series_                = {l: None for l in levels}
        self.transformer_series_['_unknown_level'] = None
        self.transformer_exog                   = transformer_exog
        self.weight_func                        = weight_func
        self.weight_func_                       = None
        self.source_code_weight_func            = None
        self.series_weights                     = series_weights
        self.series_weights_                    = None
        self.differentiation                    = differentiation
        self.differentiation_max                = None
        self.differentiator                     = None
        self.differentiator_                    = None
        self.dropna_from_series                 = dropna_from_series
        self.last_window_                       = None
        self.index_type_                        = None
        self.index_freq_                        = None
        self.training_range_                    = None
        self.series_names_in_                   = None
        self.exog_in_                           = False
        self.exog_names_in_                     = None
        self.exog_type_in_                      = None
        self.exog_dtypes_in_                    = None 
        self.X_train_series_names_in_           = None
        self.X_train_window_features_names_out_ = None
        self.X_train_exog_names_out_            = None
        self.X_train_features_names_out_        = None
        self.in_sample_residuals_               = None
        self.in_sample_residuals_by_bin_        = None
        self.out_sample_residuals_              = None
        self.out_sample_residuals_by_bin_       = None
        self.creation_date                      = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.is_fitted                          = False
        self.fit_date                           = None
        self.skforecast_version                 = skforecast.__version__
        self.python_version                     = sys.version.split(" ")[0]
        self.forecaster_id                      = forecaster_id
        self._probabilistic_mode                = "binned"

        self.lags, self.lags_names, self.max_lag = initialize_lags(type(self).__name__, lags)
        self.window_features, self.window_features_names, self.max_size_window_features = (
            initialize_window_features(window_features)
        )

        self.regressors = {}
        for q in quantiles:
            if regressor == 'GradientBoostingRegressor':
                self.regressors[q] = GradientBoostingRegressor(loss='quantile', alpha=q)
            

        if self.window_features is None and self.lags is None:
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

        if self.encoding not in ['ordinal', 'ordinal_category', 'onehot', None]:
            raise ValueError(
                f"Argument `encoding` must be one of the following values: 'ordinal', "
                f"'ordinal_category', 'onehot' or None. Got '{self.encoding}'."
            )

        if self.encoding == 'onehot':
            self.encoder = OneHotEncoder(
                               categories    = 'auto',
                               sparse_output = False,
                               drop          = None,
                               dtype         = int
                           ).set_output(transform='pandas')
        else:
            self.encoder = OrdinalEncoder(
                               categories = 'auto',
                               dtype      = int
                           ).set_output(transform='pandas')

        scaling_regressors = tuple(
            member[1]
            for member in inspect.getmembers(sklearn.linear_model, inspect.isclass)
            + inspect.getmembers(sklearn.svm, inspect.isclass)
        )

        if isinstance(self.transformer_series, dict):
            if self.encoding is None:
                raise TypeError(
                    "When `encoding` is None, `transformer_series` must be a single "
                    "transformer (not `dict`) as it is applied to all series."
                )
            if '_unknown_level' not in self.transformer_series.keys():
                raise ValueError(
                    "If `transformer_series` is a `dict`, a transformer must be "
                    "provided to transform series that do not exist during training. "
                    "Add the key '_unknown_level' to `transformer_series`. "
                    "For example: {'_unknown_level': your_transformer}."
                )

        self.weight_func, self.source_code_weight_func, self.series_weights = (
            initialize_weights(
                forecaster_name = type(self).__name__,
                regressor       = self.regressors[quantiles[0]],
                weight_func     = weight_func,
                series_weights  = series_weights,
            )
        )

        if differentiation is not None:
            if isinstance(differentiation, int):
                if differentiation < 1:
                    raise ValueError(
                        f"If `differentiation` is an integer, it must be equal "
                        f"to or greater than 1. Got {differentiation}."
                    )
                self.differentiation = differentiation
                self.differentiation_max = differentiation
                self.window_size += self.differentiation_max
                self.differentiator = TimeSeriesDifferentiator(
                    order=differentiation, window_size=self.window_size
                )
            elif isinstance(differentiation, dict):

                if self.encoding is None:
                    raise TypeError(
                        "When `encoding` is None, `differentiation` must be an "
                        "integer equal to or greater than 1. Same differentiation "
                        "must be applied to all series."
                    )
                if '_unknown_level' not in differentiation.keys():
                    raise ValueError(
                        "If `differentiation` is a `dict`, an order must be provided "
                        "to differentiate series that do not exist during training. "
                        "Add the key '_unknown_level' to `differentiation`. "
                        "For example: {'_unknown_level': 1}."
                    )
                
                differentiation_max = []
                for level, diff in differentiation.items():
                    if diff is not None:
                        if not isinstance(diff, int) or diff < 1:
                            raise ValueError(
                                f"If `differentiation` is a dict, the values must be "
                                f"None or integers equal to or greater than 1. "
                                f"Got {diff} for series '{level}'."
                            )
                        differentiation_max.append(diff)

                if len(differentiation_max) == 0:
                    raise ValueError(
                        "If `differentiation` is a dict, at least one value must be "
                        "different from None. Got all values equal to None. If you "
                        "do not want to differentiate any series, set `differentiation` "
                        "to None."
                    )
                
                self.differentiation = differentiation
                self.differentiation_max = max(differentiation_max)
                self.window_size += self.differentiation_max
                self.differentiator = {
                    level: (
                        TimeSeriesDifferentiator(order=diff, window_size=self.window_size)
                        if diff is not None else None
                    )
                    for level, diff in differentiation.items()
                }
            else:
                raise TypeError(
                    f"When including `differentiation`, this argument must be "
                    f"an integer (equal to or greater than 1) or a dict of "
                    f"integers. Got {type(differentiation)}."
                )

        self.fit_kwargs = check_select_fit_kwargs(
                              regressor  = self.regressors[quantiles[0]],
                              fit_kwargs = fit_kwargs
                          )
        
        self.binner = {}
        self.binner_intervals_ = {}
        self.binner_kwargs = binner_kwargs
        if binner_kwargs is None:
            self.binner_kwargs = {
                'n_bins': 10, 'method': 'linear', 'subsample': 200000,
                'random_state': 789654, 'dtype': np.float64
            }

    def _create_lags(
        self,
        y: np.ndarray,
        X_as_pandas: bool = False,
        train_index: pd.Index | None = None
    ) -> tuple[np.ndarray | pd.DataFrame | None, np.ndarray]:
        """
        Create the lagged values and their target variable from a time series.
        
        Note that the returned matrix `X_data` contains the lag 1 in the first 
        column, the lag 2 in the in the second column and so on.
        
        Parameters
        ----------
        y : numpy ndarray
            Training time series values.
        X_as_pandas : bool, default False
            If `True`, the returned matrix `X_data` is a pandas DataFrame.
        train_index : pandas Index, default None
            Index of the training data. It is used to create the pandas DataFrame
            `X_data` when `X_as_pandas` is `True`.

        Returns
        -------
        X_data : numpy ndarray, pandas DataFrame, None
            Lagged values (predictors).
        y_data : numpy ndarray
            Values of the time series related to each row of `X_data`.
        
        """

        X_data = None
        if self.lags is not None:
            n_rows = len(y) - self.window_size
            X_data = np.full(
                shape=(n_rows, len(self.lags)), fill_value=np.nan, order='F', dtype=float
            )
            for i, lag in enumerate(self.lags):
                X_data[:, i] = y[self.window_size - lag: -lag]

            if X_as_pandas:
                X_data = pd.DataFrame(
                             data    = X_data,
                             columns = self.lags_names,
                             index   = train_index
                         )

        y_data = y[self.window_size:]

        return X_data, y_data


    def _create_window_features(
        self, 
        y: pd.Series,
        train_index: pd.Index,
        X_as_pandas: bool = False,
    ) -> tuple[list[np.ndarray | pd.DataFrame], list[str]]:
        """
        
        Parameters
        ----------
        y : pandas Series
            Training time series.
        train_index : pandas Index
            Index of the training data. It is used to create the pandas DataFrame
            `X_train_window_features` when `X_as_pandas` is `True`.
        X_as_pandas : bool, default False
            If `True`, the returned matrix `X_train_window_features` is a 
            pandas DataFrame.

        Returns
        -------
        X_train_window_features : list
            List of numpy ndarrays or pandas DataFrames with the window features.
        X_train_window_features_names_out_ : list
            Names of the window features.
        
        """

        len_train_index = len(train_index)
        X_train_window_features = []
        X_train_window_features_names_out_ = []
        for wf in self.window_features:
            X_train_wf = wf.transform_batch(y)
            if not isinstance(X_train_wf, pd.DataFrame):
                raise TypeError(
                    f"The method `transform_batch` of {type(wf).__name__} "
                    f"must return a pandas DataFrame."
                )
            X_train_wf = X_train_wf.iloc[-len_train_index:]
            if not len(X_train_wf) == len_train_index:
                raise ValueError(
                    f"The method `transform_batch` of {type(wf).__name__} "
                    f"must return a DataFrame with the same number of rows as "
                    f"the input time series - `window_size`: {len_train_index}."
                )
            if not (X_train_wf.index == train_index).all():
                raise ValueError(
                    f"The method `transform_batch` of {type(wf).__name__} "
                    f"must return a DataFrame with the same index as "
                    f"the input time series - `window_size`."
                )
            
            X_train_window_features_names_out_.extend(X_train_wf.columns)
            if not X_as_pandas:
                X_train_wf = X_train_wf.to_numpy()     
            X_train_window_features.append(X_train_wf)

        return X_train_window_features, X_train_window_features_names_out_

    def _create_train_X_y_single_series(
        self,
        y: pd.Series,
        ignore_exog: bool,
        exog: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.Series]:
        """
        Create training matrices from univariate time series and exogenous
        variables. This method does not transform the exog variables.
        
        Parameters
        ----------
        y : pandas Series
            Training time series.
        ignore_exog : bool
            If `True`, `exog` is ignored.
        exog : pandas DataFrame, default None
            Exogenous variable/s included as predictor/s.

        Returns
        -------
        X_train_lags : pandas DataFrame
            Training values of lags.
            Shape: (len(y) - self.max_lag, len(self.lags))
        X_train_window_features_names_out_ : list
            Names of the window features.
        X_train_exog : pandas DataFrame
            Training values of exogenous variables.
            Shape: (len(y) - self.max_lag, len(exog.columns))
        y_train : pandas Series
            Values (target) of the time series related to each row of `X_train`.
            Shape: (len(y) - self.max_lag, )
        
        """

        series_name = y.name
        if len(y) <= self.window_size:
            raise ValueError(
                f"Length of '{series_name}' must be greater than the maximum window size "
                f"needed by the forecaster.\n"
                f"    Length '{series_name}': {len(y)}.\n"
                f"    Max window size: {self.window_size}.\n"
                f"    Lags window size: {self.max_lag}.\n"
                f"    Window features window size: {self.max_size_window_features}."
            )

        if self.encoding is None:
            fit_transformer = False
            transformer_series = None
        else:
            fit_transformer = False if self.is_fitted else True
            transformer_series = self.transformer_series_[series_name]

        y_values = y.to_numpy()
        y_index = y.index

        y_values = transform_numpy(
                       array             = y_values,
                       transformer       = transformer_series,
                       fit               = fit_transformer,
                       inverse_transform = False
                   )

        if self.differentiator_[series_name] is not None:
            if not self.is_fitted:
                y_values = self.differentiator_[series_name].fit_transform(y_values)
            else:
                differentiator = copy(self.differentiator_[series_name])
                y_values = differentiator.fit_transform(y_values)

        X_train_autoreg = []
        train_index = y_index[self.window_size:]

        X_train_lags, y_train = self._create_lags(
            y=y_values, X_as_pandas=True, train_index=train_index
        )
        if X_train_lags is not None:
            X_train_autoreg.append(X_train_lags)
        
        X_train_window_features_names_out_ = None
        if self.window_features is not None:
            n_diff = 0 if self.differentiation is None else self.differentiation_max
            y_window_features = pd.Series(y_values[n_diff:], index=y_index[n_diff:])
            X_train_window_features, X_train_window_features_names_out_ = (
                self._create_window_features(
                    y=y_window_features, X_as_pandas=True, train_index=train_index
                )
            )
            X_train_autoreg.extend(X_train_window_features)

        if len(X_train_autoreg) == 1:
            X_train_autoreg = X_train_autoreg[0]
        else:
            X_train_autoreg = pd.concat(X_train_autoreg, axis=1)
        
        X_train_autoreg['_level_skforecast'] = series_name

        if ignore_exog:
            X_train_exog = None
        else:
            if exog is not None:
                # The first `self.window_size` positions have to be removed from exog
                # since they are not in X_train_autoreg.
                X_train_exog = exog.iloc[self.window_size:, ]
            else:
                X_train_exog = pd.DataFrame(
                                   data    = np.nan,
                                   columns = ['_dummy_exog_col_to_keep_shape'],
                                   index   = train_index
                               )

        y_train = pd.Series(
                      data  = y_train,
                      index = train_index,
                      name  = 'y'
                  )

        return X_train_autoreg, X_train_window_features_names_out_, X_train_exog, y_train
    
    def _create_train_X_y(
        self,
        series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame],
        exog: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
        store_last_window: bool | list[str] = True,
    ) -> tuple[
        pd.DataFrame,
        pd.Series,
        dict[str, pd.Index],
        list[str],
        list[str],
        list[str],
        list[str],
        list[str],
        dict[str, type],
        dict[str, pd.Series],
    ]:
        series_dict, series_indexes = check_preprocess_series(series=series)
        input_series_is_dict = isinstance(series, dict)
        series_names_in_ = list(series_dict.keys())

        if self.is_fitted and not set(series_names_in_).issubset(set(self.series_names_in_)):
            raise ValueError(
                f"Once the Forecaster has been trained, `series` must contain "
                f"the same series names as those used during training:\n"
                f" Got      : {series_names_in_}\n"
                f" Expected : {self.series_names_in_}"
            )

        exog_dict = {serie: None for serie in series_names_in_}
        exog_names_in_ = None
        X_train_exog_names_out_ = None
        if exog is not None:
            exog_dict, exog_names_in_ = check_preprocess_exog_multiseries(
                                            input_series_is_dict = input_series_is_dict,
                                            series_indexes       = series_indexes,
                                            series_names_in_     = series_names_in_,
                                            exog                 = exog,
                                            exog_dict            = exog_dict
                                        )

            if self.is_fitted:
                if self.exog_names_in_ is None:
                    raise ValueError(
                        "Once the Forecaster has been trained, `exog` must be `None` "
                        "because no exogenous variables were added during training."
                    )
                else:
                    if not set(exog_names_in_) == set(self.exog_names_in_):
                        raise ValueError(
                            f"Once the Forecaster has been trained, `exog` must contain "
                            f"the same exogenous variables as those used during training:\n"
                            f" Got      : {exog_names_in_}\n"
                            f" Expected : {self.exog_names_in_}"
                        )

        if not self.is_fitted:
            self.transformer_series_ = initialize_transformer_series(
                                           forecaster_name    = type(self).__name__,
                                           series_names_in_   = series_names_in_,
                                           encoding           = self.encoding,
                                           transformer_series = self.transformer_series
                                       )
            
            self.differentiator_ = initialize_differentiator_multiseries(
                                       series_names_in_ = series_names_in_,
                                       differentiator   = self.differentiator
                                   )

        series_dict, exog_dict = align_series_and_exog_multiseries(
                                     series_dict          = series_dict,
                                     input_series_is_dict = input_series_is_dict,
                                     exog_dict            = exog_dict
                                 )
        
        # if not self.is_fitted and self.transformer_series_['_unknown_level'] is not None:
        #     self.transformer_series_['_unknown_level'].fit(
        #         np.concatenate(list(series_dict.values())).reshape(-1, 1)
        #     )

        ignore_exog = True if exog is None else False
        input_matrices = [
            [series_dict[k], exog_dict[k], ignore_exog]
             for k in series_dict.keys()
        ]

        X_train_autoreg_buffer = []
        X_train_exog_buffer = []
        y_train_buffer = []
        for matrices in input_matrices:

            (
                X_train_autoreg,
                X_train_window_features_names_out_,
                X_train_exog,
                y_train
            ) = self._create_train_X_y_single_series(
                y           = matrices[0],
                exog        = matrices[1],
                ignore_exog = matrices[2],
            )

            X_train_autoreg_buffer.append(X_train_autoreg)
            X_train_exog_buffer.append(X_train_exog)
            y_train_buffer.append(y_train)

        X_train = pd.concat(X_train_autoreg_buffer, axis=0)
        y_train = pd.concat(y_train_buffer, axis=0)

        if self.is_fitted:
            encoded_values = self.encoder.transform(X_train[['_level_skforecast']])
        else:
            encoded_values = self.encoder.fit_transform(X_train[['_level_skforecast']])
            for i, code in enumerate(self.encoder.categories_[0]):
                self.encoding_mapping_[code] = i

        if self.encoding == 'onehot': 
            X_train = pd.concat([
                          X_train.drop(columns='_level_skforecast'),
                          encoded_values
                      ], axis=1)
            X_train.columns = X_train.columns.str.replace('_level_skforecast_', '')
        else:
            X_train['_level_skforecast'] = encoded_values

        if self.encoding == 'ordinal_category':
            X_train['_level_skforecast'] = (
                X_train['_level_skforecast'].astype('category')
            )

        del encoded_values

        exog_dtypes_in_ = None
        if exog is not None:

            X_train_exog = pd.concat(X_train_exog_buffer, axis=0)
            if '_dummy_exog_col_to_keep_shape' in X_train_exog.columns:
                X_train_exog = (
                    X_train_exog.drop(columns=['_dummy_exog_col_to_keep_shape'])
                )

            exog_names_in_ = X_train_exog.columns.to_list()
            exog_dtypes_in_ = get_exog_dtypes(exog=X_train_exog)

            fit_transformer = False if self.is_fitted else True
            X_train_exog = transform_dataframe(
                               df                = X_train_exog,
                               transformer       = self.transformer_exog,
                               fit               = fit_transformer,
                               inverse_transform = False
                           )

            check_exog_dtypes(X_train_exog, call_check_exog=False)
            if not (X_train_exog.index == X_train.index).all():
                raise ValueError(
                    "Different index for `series` and `exog` after transformation. "
                    "They must be equal to ensure the correct alignment of values."
                )

            X_train_exog_names_out_ = X_train_exog.columns.to_list()
            X_train = pd.concat([X_train, X_train_exog], axis=1)

        if y_train.isna().to_numpy().any():
            mask = y_train.notna().to_numpy()
            y_train = y_train.iloc[mask]
            X_train = X_train.iloc[mask,]
            warnings.warn(
                "NaNs detected in `y_train`. They have been dropped because the "
                "target variable cannot have NaN values. Same rows have been "
                "dropped from `X_train` to maintain alignment. This is caused by "
                "series with interspersed NaNs.",
                MissingValuesWarning
            )

        if self.dropna_from_series:
            if np.any(X_train.isnull().to_numpy()):
                mask = X_train.notna().all(axis=1).to_numpy()
                X_train = X_train.iloc[mask, ]
                y_train = y_train.iloc[mask]
                warnings.warn(
                    "NaNs detected in `X_train`. They have been dropped. If "
                    "you want to keep them, set `forecaster.dropna_from_series = False`. "
                    "Same rows have been removed from `y_train` to maintain alignment. "
                    "This caused by series with interspersed NaNs.",
                    MissingValuesWarning
                )
        else:
            if np.any(X_train.isnull().to_numpy()):
                warnings.warn(
                    "NaNs detected in `X_train`. Some regressors do not allow "
                    "NaN values during training. If you want to drop them, "
                    "set `forecaster.dropna_from_series = True`.",
                    MissingValuesWarning
                )

        if X_train.empty:
            raise ValueError(
                "All samples have been removed due to NaNs. Set "
                "`forecaster.dropna_from_series = False` or review `exog` values."
            )
        
        if self.encoding == 'onehot':
            X_train_series_names_in_ = [
                col for col in series_names_in_ if X_train[col].sum() > 0
            ]
        else:
            unique_levels = X_train['_level_skforecast'].unique()
            X_train_series_names_in_ = [
                k for k, v in self.encoding_mapping_.items()
                if v in unique_levels
            ]

        # The last time window of training data is stored so that lags needed as
        # predictors in the first iteration of `predict()` can be calculated.
        last_window_ = None
        if store_last_window:

            series_to_store = (
                X_train_series_names_in_ if store_last_window is True else store_last_window
            )

            series_not_in_series_dict = set(series_to_store) - set(X_train_series_names_in_)
            if series_not_in_series_dict:
                warnings.warn(
                    f"Series {series_not_in_series_dict} are not present in "
                    f"`series`. No last window is stored for them.",
                    IgnoredArgumentWarning
                )
                series_to_store = [
                    s for s in series_to_store 
                    if s not in series_not_in_series_dict
                ]

            if series_to_store:
                last_window_ = {
                    k: v.iloc[-self.window_size:].copy()
                    for k, v in series_dict.items()
                    if k in series_to_store
                }

        return (
            X_train,
            y_train,
            series_indexes,
            series_names_in_,
            X_train_series_names_in_,
            exog_names_in_,
            X_train_window_features_names_out_,
            X_train_exog_names_out_,
            exog_dtypes_in_,
            last_window_
        )
    

    def create_train_X_y(
        self,
        series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame],
        exog: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
        suppress_warnings: bool = False
    ) -> tuple[pd.DataFrame, pd.Series]:
        set_skforecast_warnings(suppress_warnings, action='ignore')

        output = self._create_train_X_y(
                     series            = series, 
                     exog              = exog, 
                     store_last_window = False
                 )

        X_train = output[0]
        y_train = output[1]

        if self.encoding is None:
            X_train = X_train.drop(columns='_level_skforecast')
        
        set_skforecast_warnings(suppress_warnings, action='default')

        return X_train, y_train

    def _weight_func_all_1(
        self, 
        index: pd.Index
    ) -> np.ndarray:
        weights = np.ones(len(index), dtype=float)

        return weights

    def create_sample_weights(
        self,
        series_names_in_: list,
        X_train: pd.DataFrame
    ) -> np.ndarray:
        weights = None
        weights_samples = None
        series_weights = None

        if self.series_weights is not None:
            # Series not present in series_weights have a weight of 1 in all their samples.
            # Keys in series_weights not present in series are ignored.
            series_not_in_series_weights = (
                set(series_names_in_) - set(self.series_weights.keys())
            )
            if series_not_in_series_weights:
                warnings.warn(
                    f"{series_not_in_series_weights} not present in `series_weights`. "
                    f"A weight of 1 is given to all their samples.",
                    IgnoredArgumentWarning
                )
            self.series_weights_ = {col: 1. for col in series_names_in_}
            self.series_weights_.update(
                {
                    k: v
                    for k, v in self.series_weights.items()
                    if k in self.series_weights_
                }
            )

            if self.encoding == "onehot":
                series_weights = [
                    np.repeat(self.series_weights_[serie], sum(X_train[serie]))
                    for serie in series_names_in_
                ]
            else:
                series_weights = [
                    np.repeat(
                        self.series_weights_[serie],
                        sum(X_train["_level_skforecast"] == self.encoding_mapping_[serie]),
                    )
                    for serie in series_names_in_
                ]

            series_weights = np.concatenate(series_weights)

        if self.weight_func is not None:
            if isinstance(self.weight_func, Callable):
                self.weight_func_ = {col: copy(self.weight_func)
                                     for col in series_names_in_}
            else:
                # Series not present in weight_func have a weight of 1 in all their samples
                series_not_in_weight_func = (
                    set(series_names_in_) - set(self.weight_func.keys())
                )
                if series_not_in_weight_func:
                    warnings.warn(
                        f"{series_not_in_weight_func} not present in `weight_func`. "
                        f"A weight of 1 is given to all their samples.",
                        IgnoredArgumentWarning
                    )
                self.weight_func_ = {
                    col: self._weight_func_all_1 for col in series_names_in_
                }
                self.weight_func_.update(
                    {
                        k: v
                        for k, v in self.weight_func.items()
                        if k in self.weight_func_
                    }
                )

            weights_samples = []
            for key in self.weight_func_.keys():
                if self.encoding == "onehot":
                    idx = X_train.index[X_train[key] == 1.0]
                else:
                    idx = X_train.index[
                        X_train["_level_skforecast"] == self.encoding_mapping_[key]
                    ]
                weights_samples.append(self.weight_func_[key](idx))
            weights_samples = np.concatenate(weights_samples)

        if series_weights is not None:
            weights = series_weights
            if weights_samples is not None:
                weights = weights * weights_samples
        else:
            if weights_samples is not None:
                weights = weights_samples

        if weights is not None:
            if np.isnan(weights).any():
                raise ValueError(
                    "The resulting `weights` cannot have NaN values."
                )
            if np.any(weights < 0):
                raise ValueError(
                    "The resulting `weights` cannot have negative values."
                )
            if np.sum(weights) == 0:
                raise ValueError(
                    "The resulting `weights` cannot be normalized because "
                    "the sum of the weights is zero."
                )

        return weights

    def fit(
        self,
        series: pd.DataFrame | dict[str, pd.Series | pd.DataFrame],
        exog: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
        store_last_window: bool | list[str] = True,
        suppress_warnings: bool = False
    ) -> None:
        set_skforecast_warnings(suppress_warnings, action='ignore')
        # Reset values in case the forecaster has already been fitted.
        self.last_window_                       = None
        self.index_type_                        = None
        self.index_freq_                        = None
        self.training_range_                    = None
        self.series_names_in_                   = None
        self.exog_in_                           = False
        self.exog_names_in_                     = None
        self.exog_type_in_                      = None
        self.exog_dtypes_in_                    = None
        self.X_train_series_names_in_           = None
        self.X_train_window_features_names_out_ = None
        self.X_train_exog_names_out_            = None
        self.X_train_features_names_out_        = None
        self.binner                             = {}
        self.binner_intervals_                  = {}
        self.is_fitted                          = False
        self.fit_date                           = None

        (
            X_train,
            y_train,
            series_indexes,
            series_names_in_,
            X_train_series_names_in_,
            exog_names_in_,
            X_train_window_features_names_out_,
            X_train_exog_names_out_,
            exog_dtypes_in_,
            last_window_
        ) = self._create_train_X_y(
                series=series, exog=exog, store_last_window=store_last_window
            )

        sample_weight = self.create_sample_weights(
                            series_names_in_ = series_names_in_,
                            X_train          = X_train
                        )

        X_train_regressor = (
            X_train
            if self.encoding is not None
            else X_train.drop(columns="_level_skforecast")
        )
        for quantile, regressor in self.regressors.items():
            if sample_weight is not None:
                regressor.fit(
                    X             = X_train_regressor,
                    y             = y_train,
                    sample_weight = sample_weight,
                    **self.fit_kwargs
                )
            else:
                regressor.fit(X=X_train_regressor, y=y_train, **self.fit_kwargs)

        self.series_names_in_ = series_names_in_
        self.X_train_series_names_in_ = X_train_series_names_in_
        self.X_train_window_features_names_out_ = X_train_window_features_names_out_
        self.X_train_features_names_out_ = X_train_regressor.columns.to_list()

        self.is_fitted = True
        self.fit_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.training_range_ = {k: v[[0, -1]] for k, v in series_indexes.items()}
        self.index_type_ = type(series_indexes[series_names_in_[0]])
        if isinstance(series_indexes[series_names_in_[0]], pd.DatetimeIndex):
            self.index_freq_ = series_indexes[series_names_in_[0]].freqstr
        else:
            self.index_freq_ = series_indexes[series_names_in_[0]].step

        if exog is not None:
            self.exog_in_ = True
            self.exog_names_in_ = exog_names_in_
            self.exog_type_in_ = type(exog)
            self.exog_dtypes_in_ = exog_dtypes_in_
            self.X_train_exog_names_out_ = X_train_exog_names_out_

        if store_last_window:
            self.last_window_ = last_window_
            self.last_window_dict = {q: last_window_ for q in self.quantiles}
        
        set_skforecast_warnings(suppress_warnings, action='default')

    def _create_predict_inputs(
        self,
        steps: int,
        levels: str | list[str] | None = None,
        last_window: pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
        check_inputs: bool = True
    ) -> tuple[dict[float, pd.DataFrame], dict[str, np.ndarray] | None, list[str], pd.Index]:

        input_levels_is_None = True if levels is None else False
        levels, input_levels_is_list = prepare_levels_multiseries(
            X_train_series_names_in_=self.X_train_series_names_in_, levels=levels
        )

        if self.is_fitted:
            if last_window is None:
                levels, last_window = preprocess_levels_self_last_window_multiseries(
                                          levels               = levels,
                                          input_levels_is_list = input_levels_is_list,
                                          last_window_         = self.last_window_
                                      )
            else:
                if input_levels_is_None and isinstance(last_window, pd.DataFrame):
                    levels = last_window.columns.to_list()

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
                levels           = levels,
                series_names_in_ = self.series_names_in_,
                encoding         = self.encoding
            )
        
        last_window = last_window.iloc[
            -self.window_size :, last_window.columns.get_indexer(levels)
        ].copy()
        _, last_window_index = preprocess_last_window(
                                   last_window   = last_window,
                                   return_values = False
                               )
        prediction_index = expand_index(
                               index = last_window_index,
                               steps = steps
                           )
        last_window = last_window.to_numpy()

        if exog is not None:
            if isinstance(exog, dict):
                # Empty dataframe to be filled with the exog values of each level
                empty_exog = pd.DataFrame(
                                 data  = {col: pd.Series(dtype=dtype)
                                          for col, dtype in self.exog_dtypes_in_.items()},
                                 index = prediction_index
                             )
            else:
                if isinstance(exog, pd.Series):
                    exog = exog.to_frame()
                
                exog = transform_dataframe(
                           df                = exog,
                           transformer       = self.transformer_exog,
                           fit               = False,
                           inverse_transform = False
                       )
                check_exog_dtypes(exog=exog)
                exog_values = exog.iloc[:steps, :]
        else:
            exog_values = None
        
        # NOTE: This needs to be done to ensure that the last window dtype is float.
        last_window_values = np.full(
            shape=last_window.shape, fill_value=np.nan, order='F', dtype=float
        )
        exog_values_all_levels = []
        for idx_level, level in enumerate(levels):
            last_window_level = last_window[:, idx_level]
            last_window_level = transform_numpy(
                array             = last_window_level,
                transformer       = None,
                fit               = False,
                inverse_transform = False
            )

            if self.differentiation is not None:
                if level not in self.differentiator_.keys():
                    self.differentiator_[level] = copy(self.differentiator_['_unknown_level'])
                if self.differentiator_[level] is not None:
                    last_window_level = (
                        self.differentiator_[level].fit_transform(last_window_level)
                    )

            last_window_values[:, idx_level] = last_window_level

            if isinstance(exog, dict):
                # Fill the empty dataframe with the exog values of each level
                # and transform them if necessary
                exog_values = exog.get(level, None)
                if exog_values is not None:
                    if isinstance(exog_values, pd.Series):
                        exog_values = exog_values.to_frame()

                    exog_values = exog_values.reindex_like(empty_exog)
                else:
                    exog_values = empty_exog.copy()
            
            exog_values_all_levels.append(exog_values)

        last_window = pd.DataFrame(
                          data    = last_window_values,
                          columns = levels,
                          index   = last_window_index
                      )

        if exog is not None:
            exog_values_all_levels = pd.concat(exog_values_all_levels)
            if isinstance(exog, dict):
                exog_values_all_levels = transform_dataframe(
                                             df                = exog_values_all_levels,
                                             transformer       = self.transformer_exog,
                                             fit               = False,
                                             inverse_transform = False
                                         )
                
                check_exog_dtypes(exog=exog_values_all_levels)
            exog_values_all_levels = exog_values_all_levels.to_numpy()
            exog_values_dict = {
                i + 1: exog_values_all_levels[i::steps, :] 
                for i in range(steps)
            }
        else:
            exog_values_dict = None

        last_window_dict = {q: last_window for q in self.regressors}

        return last_window_dict, exog_values_dict, levels, prediction_index
    
    def _recursive_predict(
        self,
        quantile: float,
        steps: int,
        levels: list,
        last_window: pd.DataFrame,
        exog_values_dict: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        original_device = set_cpu_gpu_device(regressor=self.regressors[quantile], device='cpu')

        n_levels = len(levels)
        n_lags = len(self.lags) if self.lags is not None else 0
        n_window_features = (
            len(self.X_train_window_features_names_out_)
            if self.window_features is not None
            else 0
        )
        n_autoreg = n_lags + n_window_features
        n_exog = len(self.X_train_exog_names_out_) if exog_values_dict is not None else 0

        if self.encoding is not None:
            if self.encoding == "onehot":
                levels_encoded = np.zeros(
                    (n_levels, len(self.X_train_series_names_in_)), dtype=float
                )
                for i, level in enumerate(levels):
                    if level in self.X_train_series_names_in_:
                        levels_encoded[i, self.X_train_series_names_in_.index(level)] = 1.
            else:
                levels_encoded = np.array(
                    [self.encoding_mapping_.get(level, None) for level in levels],
                    dtype="float64"
                ).reshape(-1, 1)
            levels_encoded_shape = levels_encoded.shape[1]
        else:
            levels_encoded_shape = 0

        features_shape = n_autoreg + levels_encoded_shape + n_exog
        features = np.full(
            shape=(n_levels, features_shape), fill_value=np.nan, order='F', dtype=float
        )
        if self.encoding is not None:
            features[:, n_autoreg: n_autoreg + levels_encoded_shape] = levels_encoded

        predictions = np.full(
            shape=(steps, n_levels), fill_value=np.nan, order='C', dtype=float
        )
        last_window = np.concatenate((last_window.to_numpy(), predictions), axis=0)

        for i in range(steps):
            
            if self.lags is not None:
                features[:, :n_lags] = last_window[
                    -self.lags - (steps - i), :
                ].transpose()
            if self.window_features is not None:
                features[:, n_lags:n_autoreg] = np.concatenate(
                    [
                        wf.transform(last_window[i:-(steps - i), :]) 
                        for wf in self.window_features
                    ],
                    axis=1
                )
            if exog_values_dict is not None:
                features[:, -n_exog:] = exog_values_dict[i + 1]

            pred = self.regressors[quantile].predict(features)
            
            predictions[i, :] = pred 

            # Update `last_window` values. The first position is discarded and 
            # the new prediction is added at the end.
            last_window[-(steps - i), :] = pred

        set_cpu_gpu_device(regressor=self.regressors[quantile], device=original_device)

        return predictions
    
    def predict(
        self,
        steps: int,
        levels: str | list[str] | None = None,
        last_window: pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | dict[str, pd.Series | pd.DataFrame] | None = None,
        suppress_warnings: bool = False,
        check_inputs: bool = True
    ) -> dict:
        set_skforecast_warnings(suppress_warnings, action='ignore')

        (
            last_window,
            exog_values_dict,
            levels,
            prediction_index
        ) = self._create_predict_inputs(
                steps        = steps,
                levels       = levels,
                last_window  = last_window,
                exog         = exog,
                check_inputs = check_inputs
            )

        prediction_dict = {}

        for q in self.quantiles:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", 
                    message="X does not have valid feature names", 
                    category=UserWarning
                )
                predictions = self._recursive_predict(
                                quantile         = q,
                                steps            = steps,
                                levels           = levels,
                                last_window      = last_window[q],
                                exog_values_dict = exog_values_dict
                            )
            
            for i, level in enumerate(levels):
                if self.differentiation is not None and self.differentiator_[level] is not None:
                    predictions[:, i] = (
                        self
                        .differentiator_[level]
                        .inverse_transform_next_window(predictions[:, i])
                    )

                predictions[:, i] = transform_numpy(
                    array             = predictions[:, i],
                    transformer       = None,
                    fit               = False,
                    inverse_transform = True
                )
            
            n_steps, n_levels = predictions.shape
            predictions = pd.DataFrame(
                {"level": np.tile(levels, n_steps), "pred": predictions.ravel()},
                index = np.repeat(prediction_index, n_levels),
            )

            prediction_dict[q] = predictions
        
        set_skforecast_warnings(suppress_warnings, action='default')

        return prediction_dict    
    

    def set_params(
        self, 
        params: dict[str, object]
    ) -> None:
        self.regressor = clone(self.regressor)
        self.regressor.set_params(**params)