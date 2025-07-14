"""A model that wraps lightgbm."""

# pylint: disable=duplicate-code,too-many-arguments,too-many-positional-arguments,too-many-instance-attributes
import json
import os
from typing import Self

import joblib  # type: ignore
import lightgbm as lgb
import optuna
import pandas as pd
import torch

from ...exceptions import WavetrainException
from ...model_type import ModelType, determine_model_type
from ..model import PREDICTION_COLUMN, PROBABILITY_COLUMN_PREFIX, Model

_BOOSTING_TYPE_KEY = "gbm_boosting_type"
_NUM_LEAVES_KEY = "gbm_num_leaves"
_MIN_CHILD_SAMPLES_KEY = "gbm_min_child_samples"
_MODEL_PARAMS_FILENAME = "model_params.json"
_MODEL_FILENAME = "model.pkl"
_BEST_ITERATION_KEY = "best_iteration"
_EARLY_STOPPING_ROUNDS_KEY = "gbm_early_stopping_rounds"
_ITERATIONS_KEY = "gbm_iterations"


class LightGBMModel(Model):
    """A class that uses lightgbm as a model."""

    _gbm: lgb.LGBMModel | None
    _boosting_type: str | None
    _num_leaves: int | None
    _min_child_samples: int | None
    _model_type: None | ModelType
    _best_iteration: None | int
    _early_stopping_rounds: None | int
    _iterations: None | int

    @classmethod
    def name(cls) -> str:
        return "lightgbm"

    @classmethod
    def supports_x(cls, df: pd.DataFrame) -> bool:
        return True

    def __init__(self) -> None:
        super().__init__()
        self._gbm = None
        self._boosting_type = None
        self._num_leaves = None
        self._min_child_samples = None
        self._model_type = None
        self._best_iteration = None
        self._early_stopping_rounds = None
        self._iterations = None

    @property
    def supports_importances(self) -> bool:
        return True

    def feature_importances(
        self, df: pd.DataFrame | None
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        gbm = self._provide_gbm()
        importances = gbm.feature_importances_
        names = gbm.feature_name_
        total_importances = sum(importances)
        return {
            names[count]: importance / total_importances
            for count, importance in enumerate(importances)
        }, []

    def provide_estimator(self):
        return self._provide_gbm()

    def create_estimator(self):
        return self._create_gbm()

    def reset(self):
        self._gbm = None
        self._best_iteration = None

    def convert_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        self._boosting_type = trial.suggest_categorical(
            _BOOSTING_TYPE_KEY, ["gbdt", "dart", "rf"]
        )
        self._num_leaves = trial.suggest_int(_NUM_LEAVES_KEY, 2, 256)
        self._min_child_samples = trial.suggest_int(_MIN_CHILD_SAMPLES_KEY, 5, 100)
        self._best_iteration = trial.user_attrs.get(_BEST_ITERATION_KEY)
        self._early_stopping_rounds = trial.suggest_int(
            _EARLY_STOPPING_ROUNDS_KEY, 10, 500
        )
        self._iterations = trial.suggest_int(_ITERATIONS_KEY, 100, 10000)

    def load(self, folder: str) -> None:
        with open(
            os.path.join(folder, _MODEL_PARAMS_FILENAME), encoding="utf8"
        ) as handle:
            params = json.load(handle)
            self._boosting_type = params[_BOOSTING_TYPE_KEY]
            self._num_leaves = params[_NUM_LEAVES_KEY]
            self._min_child_samples = params[_MIN_CHILD_SAMPLES_KEY]
            self._best_iteration = params.get(_BEST_ITERATION_KEY)
            self._early_stopping_rounds = params[_EARLY_STOPPING_ROUNDS_KEY]
            self._iterations = params[_ITERATIONS_KEY]
        self._gbm = joblib.load(os.path.join(folder, _MODEL_FILENAME))

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        with open(
            os.path.join(folder, _MODEL_PARAMS_FILENAME), "w", encoding="utf8"
        ) as handle:
            json.dump(
                {
                    _BOOSTING_TYPE_KEY: self._boosting_type,
                    _NUM_LEAVES_KEY: self._num_leaves,
                    _MIN_CHILD_SAMPLES_KEY: self._min_child_samples,
                    _BEST_ITERATION_KEY: self._best_iteration,
                    _EARLY_STOPPING_ROUNDS_KEY: self._early_stopping_rounds,
                    _ITERATIONS_KEY: self._iterations,
                },
                handle,
            )
        gbm = self._provide_gbm()
        joblib.dump(gbm, os.path.join(folder, _MODEL_FILENAME))
        trial.set_user_attr(_BEST_ITERATION_KEY, self._best_iteration)

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        if y is None:
            raise ValueError("y is null.")
        self._model_type = determine_model_type(y)
        gbm = self._provide_gbm()
        early_stopping_rounds = self._early_stopping_rounds
        if early_stopping_rounds is None:
            raise ValueError("early_stopping_rounds is null")

        eval_set = None
        callbacks = []
        if eval_x is not None and eval_y is not None:
            eval_set = [(eval_x, eval_y.to_numpy().flatten())]  # type: ignore
            callbacks = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            ]
        if self._best_iteration is not None:
            eval_set = None
            callbacks = []
        try:
            gbm.fit(
                X=df,
                y=y.to_numpy().flatten(),
                sample_weight=w,
                eval_set=eval_set,  # type: ignore
                callbacks=callbacks,  # type: ignore
            )
        except lgb.basic.LightGBMError as exc:
            raise WavetrainException() from exc
        self._best_iteration = gbm.best_iteration_
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        gbm = self._provide_gbm()
        pred = gbm.predict(df)
        pred_df = pd.DataFrame(
            index=df.index,
            data={
                PREDICTION_COLUMN: pred.flatten(),  # type: ignore
            },
        )
        if self._model_type != ModelType.REGRESSION:
            proba = gbm.predict_proba(df)  # type: ignore
            for i in range(proba.shape[1]):
                pred_df[f"{PROBABILITY_COLUMN_PREFIX}{i}"] = proba[:, i]
        return pred_df

    def _provide_gbm(self) -> lgb.LGBMModel:
        gbm = self._gbm
        if gbm is None:
            gbm = self._create_gbm()
            self._gbm = gbm
        if gbm is None:
            raise ValueError("gbm is null")
        return gbm

    def _create_gbm(self) -> lgb.LGBMModel:
        best_iteration = self._best_iteration
        iterations = best_iteration if best_iteration is not None else self._iterations
        boosting_type = self._boosting_type
        if boosting_type is None:
            raise ValueError("boosting_type is null")
        num_leaves = self._num_leaves
        if num_leaves is None:
            raise ValueError("num_leaves is null")
        min_child_samples = self._min_child_samples
        if min_child_samples is None:
            raise ValueError("min_child_samples is null")

        match self._model_type:
            case ModelType.BINARY:
                return lgb.LGBMClassifier(
                    boosting_type=boosting_type,
                    num_leaves=num_leaves,
                    objective="binary",
                    min_child_samples=min_child_samples,
                    num_iterations=iterations,
                    device="gpu" if torch.cuda.is_available() else None,
                )
            case ModelType.REGRESSION:
                return lgb.LGBMRegressor(
                    boosting_type=boosting_type,
                    num_leaves=num_leaves,
                    min_child_samples=min_child_samples,
                    num_iterations=iterations,
                    device="gpu" if torch.cuda.is_available() else None,
                )
            case ModelType.BINNED_BINARY:
                return lgb.LGBMClassifier(
                    boosting_type=boosting_type,
                    num_leaves=num_leaves,
                    objective="binary",
                    min_child_samples=min_child_samples,
                    num_iterations=iterations,
                    device="gpu" if torch.cuda.is_available() else None,
                )
            case ModelType.MULTI_CLASSIFICATION:
                return lgb.LGBMClassifier(
                    boosting_type=boosting_type,
                    num_leaves=num_leaves,
                    min_child_samples=min_child_samples,
                    num_iterations=iterations,
                    device="gpu" if torch.cuda.is_available() else None,
                )
            case _:
                raise ValueError(f"Unrecognised model type: {self._model_type}")
