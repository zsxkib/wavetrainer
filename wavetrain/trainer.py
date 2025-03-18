"""The trainer class."""

import datetime
import functools
import json
import logging
import os
import pickle
from typing import Self

import optuna
import pandas as pd
import tqdm
from sklearn.metrics import accuracy_score, f1_score  # type: ignore

from .calibrator.calibrator_router import CalibratorRouter
from .exceptions import WavetrainException
from .fit import Fit
from .model.model import PREDICTION_COLUMN
from .model.model_router import ModelRouter
from .model_type import ModelType, determine_model_type
from .reducer.combined_reducer import CombinedReducer
from .selector.selector import Selector
from .weights.combined_weights import CombinedWeights
from .weights.weights import WEIGHTS_COLUMN
from .windower.windower import Windower

_SAMPLER_FILENAME = "sampler.pkl"
_STUDYDB_FILENAME = "study.db"
_PARAMS_FILENAME = "params.json"
_TRIALS_KEY = "trials"
_WALKFORWARD_TIMEDELTA_KEY = "walkforward_timedelta"
_DAYS_KEY = "days"
_SECONDS_KEY = "seconds"
_TEST_SIZE_KEY = "test_size"
_VALIDATION_SIZE_KEY = "validation_size"
_IDX_USR_ATTR_KEY = "idx"
_DT_COLUMN_KEY = "dt_column"


class Trainer(Fit):
    """A class for training and predicting from an array of data."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-statements,too-many-locals,too-many-branches

    def __init__(
        self,
        folder: str,
        walkforward_timedelta: datetime.timedelta | None = None,
        trials: int | None = None,
        test_size: float | datetime.timedelta | None = None,
        validation_size: float | datetime.timedelta | None = None,
        dt_column: str | None = None,
    ):
        tqdm.tqdm.pandas()

        self._folder = folder
        if not os.path.exists(self._folder):
            os.makedirs(self._folder)

        if trials is None:
            trials = 100
        if test_size is None:
            test_size = 0.15
        if validation_size is None:
            validation_size = 0.15

        params_file = os.path.join(self._folder, _PARAMS_FILENAME)
        if walkforward_timedelta is None:
            if not os.path.exists(params_file):
                raise ValueError(
                    "walkforward_timedelta must be specified on an untrained model."
                )
            with open(params_file, encoding="utf8") as handle:
                params = json.load(handle)
                walkforward_dict = params[_WALKFORWARD_TIMEDELTA_KEY]
                walkforward_timedelta = datetime.timedelta(
                    days=walkforward_dict.get(_DAYS_KEY),
                    seconds=walkforward_dict.get(_SECONDS_KEY),
                )
                trials = params[_TRIALS_KEY]
                test_size_value = params[_TEST_SIZE_KEY]
                if test_size_value is not None:
                    if isinstance(test_size_value, float):
                        test_size = test_size_value
                    else:
                        test_size = datetime.timedelta(
                            days=test_size_value.get(_DAYS_KEY),
                            seconds=test_size_value.get(_SECONDS_KEY),
                        )
                validation_size_value = params[_VALIDATION_SIZE_KEY]
                if validation_size_value is not None:
                    if isinstance(validation_size_value, float):
                        validation_size = validation_size_value
                    else:
                        validation_size = datetime.timedelta(
                            days=validation_size_value.get(_DAYS_KEY),
                            seconds=validation_size_value.get(_SECONDS_KEY),
                        )
                if dt_column is None:
                    dt_column = params[_DT_COLUMN_KEY]
        else:
            with open(params_file, "w", encoding="utf8") as handle:
                validation_size_value = None
                if validation_size is not None:
                    if isinstance(validation_size, float):
                        validation_size_value = validation_size
                    elif isinstance(validation_size, datetime.timedelta):
                        validation_size_value = {
                            _DAYS_KEY: validation_size.days,
                            _SECONDS_KEY: validation_size.seconds,
                        }
                test_size_value = None
                if test_size is not None:
                    if isinstance(test_size, float):
                        test_size_value = test_size
                    elif isinstance(test_size, datetime.timedelta):
                        test_size_value = {
                            _DAYS_KEY: test_size.days,
                            _SECONDS_KEY: test_size.seconds,
                        }
                json.dump(
                    {
                        _TRIALS_KEY: trials,
                        _WALKFORWARD_TIMEDELTA_KEY: {
                            _DAYS_KEY: walkforward_timedelta.days,
                            _SECONDS_KEY: walkforward_timedelta.seconds,
                        },
                        _TEST_SIZE_KEY: test_size_value,
                        _VALIDATION_SIZE_KEY: validation_size_value,
                        _DT_COLUMN_KEY: dt_column,
                    },
                    handle,
                )
        self._walkforward_timedelta = walkforward_timedelta
        self._trials = trials
        self._test_size = test_size
        self._validation_size = validation_size
        self._dt_column = dt_column

    def _provide_study(self, column: str) -> optuna.Study:
        storage_name = f"sqlite:///{self._folder}/{column}/{_STUDYDB_FILENAME}"
        sampler_file = os.path.join(self._folder, column, _SAMPLER_FILENAME)
        restored_sampler = None
        if os.path.exists(sampler_file):
            with open(sampler_file, "rb") as handle:
                restored_sampler = pickle.load(handle)
        return optuna.create_study(
            study_name="wavetrain",
            storage=storage_name,
            load_if_exists=True,
            sampler=restored_sampler,
            direction=optuna.study.StudyDirection.MAXIMIZE,
        )

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
    ) -> Self:
        """Perform a train on the data to fit to the targets."""
        if y is None:
            return self

        dt_index = df.index if self._dt_column is None else df[self._dt_column]

        def _fit_column(y_series: pd.Series):
            column_dir = os.path.join(self._folder, str(y_series.name))
            if not os.path.exists(column_dir):
                os.mkdir(column_dir)

            study = self._provide_study(str(y_series.name))

            def _fit(
                trial: optuna.Trial | optuna.trial.FrozenTrial,
                x: pd.DataFrame,
                y_series: pd.Series,
                save: bool,
                split_idx: datetime.datetime,
            ) -> float:
                trial.set_user_attr(_IDX_USR_ATTR_KEY, split_idx.isoformat())

                train_dt_index = dt_index[: len(x)]
                x_train = x[train_dt_index < split_idx]
                x_test = x[train_dt_index >= split_idx]
                y_train = y_series[train_dt_index < split_idx]
                y_test = y_series[train_dt_index >= split_idx]

                try:
                    # Window the data
                    windower = Windower(self._dt_column)
                    windower.set_options(trial)
                    x_train = windower.fit_transform(x_train)
                    y_train = y_train[-len(x_train) :]
                    if len(y_train.unique()) <= 1:
                        logging.warning("Y train only contains 1 unique datapoint.")
                        return -1.0

                    # Perform common reductions
                    reducer = CombinedReducer()
                    reducer.set_options(trial)
                    x_train = reducer.fit_transform(x_train)
                    x_test = reducer.transform(x_test)

                    # Calculate the row weights
                    weights = CombinedWeights()
                    weights.set_options(trial)
                    w = weights.fit(x_train, y=y_train).transform(y_train.to_frame())[
                        WEIGHTS_COLUMN
                    ]

                    # Create model
                    model = ModelRouter()
                    model.set_options(trial)

                    # Train
                    selector = Selector(model, len(x_train.columns.values))
                    selector.set_options(trial)
                    selector.fit(x_train, y=y_train, w=w)
                    x_train = selector.transform(x_train)
                    x_test = selector.transform(x_test)
                    x_pred = model.fit_transform(x_train, y=y_train)

                    # Calibrate
                    calibrator = CalibratorRouter(model)
                    calibrator.set_options(trial)
                    calibrator.fit(x_pred, y=y_train)

                    if save:
                        folder = os.path.join(
                            self._folder, str(y_series.name), split_idx.isoformat()
                        )
                        if not os.path.exists(folder):
                            os.mkdir(folder)
                        windower.save(folder)
                        reducer.save(folder)
                        weights.save(folder)
                        model.save(folder)
                        selector.save(folder)
                        calibrator.save(folder)

                    y_pred = model.transform(x_test)
                    y_pred = calibrator.transform(y_pred)
                    if determine_model_type(y_series) == ModelType.REGRESSION:
                        return accuracy_score(y_test, y_pred[[PREDICTION_COLUMN]])
                    return f1_score(y_test, y_pred[[PREDICTION_COLUMN]])
                except WavetrainException as exc:
                    logging.warning(str(exc))
                    return -1.0

            start_validation_index = (
                dt_index[-int(len(dt_index) * self._validation_size) - 1]
                if isinstance(self._validation_size, float)
                else dt_index[dt_index >= self._validation_size][0]
            )
            test_df = df[dt_index < start_validation_index]
            test_dt_index = (
                test_df.index if self._dt_column is None else test_df[self._dt_column]
            )
            start_test_index = (
                test_dt_index[-int(len(test_dt_index) * self._test_size)]
                if isinstance(self._test_size, float)
                else test_dt_index[test_dt_index >= self._test_size][0]
            )

            def test_objective(trial: optuna.Trial) -> float:
                return _fit(
                    trial,
                    test_df,
                    y_series[dt_index < start_validation_index],
                    False,
                    start_test_index.to_pydatetime(),
                )

            initial_trials = 0
            if self._trials is not None:
                initial_trials = max(self._trials - len(study.trials), 0)
            if initial_trials > 0:
                study.optimize(
                    test_objective, n_trials=initial_trials, show_progress_bar=True
                )

            train_len = len(df[dt_index < start_test_index])
            test_len = len(df.loc[start_test_index:start_validation_index])

            for count, test_idx in tqdm.tqdm(
                enumerate(df[dt_index >= start_test_index].index)
            ):
                test_dt = test_idx.to_pydatetime()
                found = False
                for trial in study.trials:
                    dt_idx = datetime.datetime.fromisoformat(
                        trial.user_attrs[_IDX_USR_ATTR_KEY]
                    )
                    if dt_idx == test_dt:
                        found = True
                        break
                if found:
                    continue

                test_df = df.iloc[: train_len + count + test_len]
                test_series = y_series.iloc[: train_len + count + test_len]

                if test_idx < start_validation_index:

                    def validate_objctive(
                        trial: optuna.Trial, idx: datetime.datetime, series: pd.Series
                    ) -> float:
                        return _fit(trial, test_df, series, False, idx)

                    study.optimize(
                        functools.partial(
                            validate_objctive, idx=test_idx, series=test_series
                        ),
                        n_trials=1,
                    )

                _fit(study.best_trial, test_df, test_series, True, test_idx)

        if isinstance(y, pd.Series):
            _fit_column(y)
        else:
            for col in y.columns:
                _fit_column(y[col])

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict the expected values of the data."""
        feature_columns = df.columns.values
        dt_index = df.index if self._dt_column is None else df[self._dt_column]

        for column in os.listdir(self._folder):
            column_path = os.path.join(self._folder, column)
            if not os.path.isdir(column_path):
                continue
            dates = []
            for date_str in os.listdir(column_path):
                date_path = os.path.join(column_path, date_str)
                if not os.path.isdir(date_path):
                    continue
                dates.append(datetime.datetime.fromisoformat(date_str))
            bins: list[datetime.datetime] = sorted(
                [dt_index.min().to_pydatetime()]
                + dates
                + [(dt_index.max() + pd.Timedelta(days=1)).to_pydatetime()]
            )

            def assign_bin(timestamp, bins: list[datetime.datetime]) -> int:
                for i in range(len(bins) - 1):
                    if bins[i] <= timestamp < bins[i + 1]:
                        return i
                return len(bins) - 2  # Assign to last bin if at the end

            def perform_predictions(
                group: pd.DataFrame,
                column_path: str,
                column: str,
                dates: list[datetime.datetime],
            ) -> pd.DataFrame:
                filtered_dates = [x for x in dates if x < group.index.max()]
                if not filtered_dates:
                    filtered_dates = [dates[-1]]
                date_str = dates[-1].isoformat()
                folder = os.path.join(column_path, date_str)

                reducer = CombinedReducer()
                reducer.load(folder)

                model = ModelRouter()
                model.load(folder)

                selector = Selector(model, len(df.columns.values))
                selector.load(folder)

                calibrator = CalibratorRouter(model)
                calibrator.load(folder)

                x_pred = reducer.transform(group[feature_columns])
                x_pred = selector.transform(x_pred)
                y_pred = model.transform(x_pred)
                y_pred = calibrator.transform(y_pred)
                for new_column in y_pred.columns.values:
                    group["_".join([column, new_column])] = y_pred[new_column]
                return group

            old_index = dt_index.copy()
            df = df.groupby(
                dt_index.map(functools.partial(assign_bin, bins=bins))
            ).progress_apply(  # type: ignore
                functools.partial(
                    perform_predictions,
                    column_path=column_path,
                    column=column,
                    dates=dates,
                )
            )
            if self._dt_column is None:
                df = df.set_index(old_index)
        return df
