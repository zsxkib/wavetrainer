"""A model that wraps xgboost."""
# pylint: disable=duplicate-code,too-many-arguments,too-many-positional-arguments,too-many-instance-attributes

import json
import os
from typing import Any, Self

import optuna
import pandas as pd
import torch
from xgboost import XGBClassifier, XGBRegressor
from xgboost.callback import TrainingCallback

from ...model_type import ModelType, determine_model_type
from ..model import PREDICTION_COLUMN, PROBABILITY_COLUMN_PREFIX, Model
from .early_stopper import XGBoostEarlyStoppingCallback
from .xgboost_logger import XGBoostEpochsLogger

_MODEL_FILENAME = "xgboost_model.json"
_MODEL_PARAMS_FILENAME = "xgboost_model_params.json"
_MODEL_TYPE_KEY = "model_type"
_BEST_ITERATION_KEY = "best_iteration"


def _convert_categoricals(input_df: pd.DataFrame) -> pd.DataFrame:
    output_df = input_df.copy()
    for col in input_df.select_dtypes(include=["category"]).columns:
        output_df[col] = output_df[col].cat.codes
    return output_df


class XGBoostModel(Model):
    """A class that uses XGBoost as a model."""

    _xgboost: XGBRegressor | XGBClassifier | None
    _model_type: None | ModelType
    _booster: str | None
    _lambda: float | None
    _alpha: float | None
    _subsample: float | None
    _colsample_bytree: float | None
    _max_depth: int | None
    _min_child_weight: int | None
    _eta: float | None
    _gamma: float | None
    _grow_policy: str | None
    _sample_type: str | None
    _normalize_type: str | None
    _rate_drop: float | None
    _skip_drop: float | None
    _num_boost_rounds: int | None
    _early_stopping_rounds: int | None
    _best_iteration: int | None

    @classmethod
    def name(cls) -> str:
        return "xgboost"

    @classmethod
    def supports_x(cls, df: pd.DataFrame) -> bool:
        return True

    def __init__(self) -> None:
        super().__init__()
        self._xgboost = None
        self._model_type = None
        self._booster = None
        self._lambda = None
        self._alpha = None
        self._subsample = None
        self._colsample_bytree = None
        self._max_depth = None
        self._min_child_weight = None
        self._eta = None
        self._gamma = None
        self._grow_policy = None
        self._sample_type = None
        self._normalize_type = None
        self._rate_drop = None
        self._skip_drop = None
        self._num_boost_rounds = None
        self._early_stopping_rounds = None
        self._best_iteration = None

    @property
    def estimator(self) -> Any:
        return self._provide_xgboost()

    @property
    def supports_importances(self) -> bool:
        return True

    @property
    def feature_importances(self) -> dict[str, float]:
        bst = self._provide_xgboost()
        return bst.get_score(importance_type="weight")  # type: ignore

    def pre_fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
    ):
        if y is None:
            raise ValueError("y is null.")
        self._model_type = determine_model_type(y)
        return {
            "eval_set": (eval_x, eval_y),
            "sample_weight": w,
        }

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        self._booster = trial.suggest_categorical(
            "booster", ["gbtree", "gblinear", "dart"]
        )
        self._lambda = trial.suggest_float("lambda", 1e-8, 1.0, log=True)
        self._alpha = trial.suggest_float("alpha", 1e-8, 1.0, log=True)
        self._subsample = trial.suggest_float("subsample", 0.2, 1.0)
        self._colsample_bytree = trial.suggest_float("colsample_bytree", 0.2, 1.0)
        if self._booster in ["gbtree", "dart"]:
            self._max_depth = trial.suggest_int("max_depth", 3, 9, step=2)
            self._min_child_weight = trial.suggest_int(
                "min_child_weight", 2, 10, log=True
            )
            self._eta = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            self._gamma = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            self._grow_policy = trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            )
        else:
            self._sample_type = trial.suggest_categorical(
                "sample_type", ["uniform", "weighted"]
            )
            self._normalize_type = trial.suggest_categorical(
                "normalize_type", ["tree", "forest"]
            )
            self._rate_drop = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            self._skip_drop = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        self._num_boost_rounds = trial.suggest_int("num_boost_rounds", 100, 10000)
        self._early_stopping_rounds = trial.suggest_int(
            "early_stopping_rounds", 50, 500
        )
        self._best_iteration = trial.user_attrs.get(_BEST_ITERATION_KEY)

    def load(self, folder: str) -> None:
        with open(
            os.path.join(folder, _MODEL_PARAMS_FILENAME), encoding="utf8"
        ) as handle:
            params = json.load(handle)
            self._model_type = ModelType(params[_MODEL_TYPE_KEY])
            self._best_iteration = params.get(_BEST_ITERATION_KEY)
        bst = self._provide_xgboost()
        bst.load_model(os.path.join(folder, _MODEL_FILENAME))

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        bst = self._provide_xgboost()
        bst.save_model(os.path.join(folder, _MODEL_FILENAME))
        with open(
            os.path.join(folder, _MODEL_PARAMS_FILENAME), "w", encoding="utf8"
        ) as handle:
            json.dump(
                {
                    _MODEL_TYPE_KEY: str(self._model_type),
                    _BEST_ITERATION_KEY: self._best_iteration,
                },
                handle,
            )
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
        xgboost = self._provide_xgboost()
        df = _convert_categoricals(df)
        evals = [(df, y)]
        if eval_x is not None and eval_y is not None and self._best_iteration is None:
            eval_x = _convert_categoricals(eval_x)
            evals.append((eval_x, eval_y))
        xgboost.fit(  # type: ignore
            df,
            y,
            eval_set=evals,
            sample_weight=w,
            verbose=False,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x_df = _convert_categoricals(df)
        xgboost = self._provide_xgboost()
        pred = xgboost.predict(x_df)
        df = pd.DataFrame(
            index=df.index,
            data={
                PREDICTION_COLUMN: pred.flatten(),
            },
        )
        if self._model_type != ModelType.REGRESSION:
            proba = xgboost.predict_proba(x_df)  # type: ignore
            for i in range(proba.shape[1]):
                df[f"{PROBABILITY_COLUMN_PREFIX}{i}"] = proba[:, i]
        return df

    def _provide_xgboost(self) -> XGBClassifier | XGBRegressor:
        xgboost = self._xgboost
        if xgboost is None:
            callbacks: list[TrainingCallback] = [
                XGBoostEpochsLogger(),
            ]
            if self._best_iteration is not None:
                callbacks.append(
                    XGBoostEarlyStoppingCallback(rounds=self._early_stopping_rounds)
                )
            param = {
                "objective": "binary:logistic",
                "tree_method": "gpu_hist" if torch.cuda.is_available() else "exact",
                # defines booster, gblinear for linear functions.
                "booster": self._booster,
                # L2 regularization weight.
                "reg_lambda": self._lambda,
                # L1 regularization weight.
                "alpha": self._alpha,
                # sampling ratio for training data.
                "subsample": self._subsample,
                # sampling according to each tree.
                "colsample_bytree": self._colsample_bytree,
                "n_estimators": self._best_iteration
                if self._best_iteration is not None
                else self._num_boost_rounds,
                "base_score": 0.5,
                "verbosity": 0,
                "verbose": False,
                "callbacks": callbacks,
                "eval_metric": ["logloss", "error"],
            }
            if param["booster"] in ["gbtree", "dart"]:
                # maximum depth of the tree, signifies complexity of the tree.
                param["max_depth"] = self._max_depth
                # minimum child weight, larger the term more conservative the tree.
                param["min_child_weight"] = self._min_child_weight
                param["eta"] = self._eta
                # defines how selective algorithm is.
                param["gamma"] = self._gamma
                param["grow_policy"] = self._grow_policy

            if param["booster"] == "dart":
                param["sample_type"] = self._sample_type
                param["normalize_type"] = self._normalize_type
                param["rate_drop"] = self._rate_drop
                param["skip_drop"] = self._skip_drop
            match self._model_type:
                case ModelType.BINARY:
                    xgboost = XGBClassifier(**param)
                case ModelType.REGRESSION:
                    param["objective"] = "reg:squarederror"
                    param["eval_metric"] = ["rmse", "mae"]
                    xgboost = XGBRegressor(**param)
                case ModelType.BINNED_BINARY:
                    xgboost = XGBClassifier(**param)
                case ModelType.MULTI_CLASSIFICATION:
                    xgboost = XGBClassifier(**param)
            self._xgboost = xgboost
        if xgboost is None:
            raise ValueError("xgboost is null")
        return xgboost
