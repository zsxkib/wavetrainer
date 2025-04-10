"""A model that wraps catboost."""

import json
import logging
import os
from typing import Any, Self

import optuna
import pandas as pd
import torch
from catboost import CatBoost, Pool  # type: ignore

from ..model_type import ModelType, determine_model_type
from .catboost_classifier_wrap import CatBoostClassifierWrapper
from .catboost_kwargs import (CAT_FEATURES_ARG_KEY, EVAL_SET_ARG_KEY,
                              ORIGINAL_X_ARG_KEY)
from .catboost_regressor_wrap import CatBoostRegressorWrapper
from .model import PREDICTION_COLUMN, PROBABILITY_COLUMN_PREFIX, Model

_MODEL_FILENAME = "model.cbm"
_MODEL_PARAMS_FILENAME = "model_params.json"
_ITERATIONS_KEY = "iterations"
_LEARNING_RATE_KEY = "learning_rate"
_DEPTH_KEY = "depth"
_L2_LEAF_REG_KEY = "l2_leaf_reg"
_BOOSTING_TYPE_KEY = "boosting_type"
_MODEL_TYPE_KEY = "model_type"
_EARLY_STOPPING_ROUNDS = "early_stopping_rounds"


class CatboostModel(Model):
    """A class that uses Catboost as a model."""

    # pylint: disable=too-many-positional-arguments,too-many-arguments,too-many-instance-attributes

    _catboost: CatBoost | None
    _iterations: None | int
    _learning_rate: None | float
    _depth: None | int
    _l2_leaf_reg: None | float
    _boosting_type: None | str
    _model_type: None | ModelType
    _early_stopping_rounds: None | int

    @classmethod
    def name(cls) -> str:
        return "catboost"

    def __init__(self) -> None:
        super().__init__()
        self._catboost = None
        self._iterations = None
        self._learning_rate = None
        self._depth = None
        self._l2_leaf_reg = None
        self._boosting_type = None
        self._model_type = None
        self._early_stopping_rounds = None

    @property
    def estimator(self) -> Any:
        return self._provide_catboost()

    def pre_fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ):
        if y is None:
            raise ValueError("y is null.")
        self._model_type = determine_model_type(y)
        return {
            EVAL_SET_ARG_KEY: (eval_x, eval_y),
            CAT_FEATURES_ARG_KEY: df.select_dtypes(include="category").columns.tolist(),
            ORIGINAL_X_ARG_KEY: df,
        }

    def set_options(self, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        self._iterations = trial.suggest_int(_ITERATIONS_KEY, 100, 10000)
        self._learning_rate = trial.suggest_float(_LEARNING_RATE_KEY, 0.001, 0.3)
        self._depth = trial.suggest_int(_DEPTH_KEY, 1, 10)
        self._l2_leaf_reg = trial.suggest_float(_L2_LEAF_REG_KEY, 3.0, 50.0)
        self._boosting_type = trial.suggest_categorical(
            _BOOSTING_TYPE_KEY, ["Ordered", "Plain"]
        )
        self._early_stopping_rounds = trial.suggest_int(_EARLY_STOPPING_ROUNDS, 10, 500)

    def load(self, folder: str) -> None:
        with open(
            os.path.join(folder, _MODEL_PARAMS_FILENAME), encoding="utf8"
        ) as handle:
            params = json.load(handle)
            self._iterations = params[_ITERATIONS_KEY]
            self._learning_rate = params[_LEARNING_RATE_KEY]
            self._depth = params[_DEPTH_KEY]
            self._l2_leaf_reg = params[_L2_LEAF_REG_KEY]
            self._boosting_type = params[_BOOSTING_TYPE_KEY]
            self._model_type = ModelType(params[_MODEL_TYPE_KEY])
            self._early_stopping_rounds = params[_EARLY_STOPPING_ROUNDS]
        catboost = self._provide_catboost()
        catboost.load_model(os.path.join(folder, _MODEL_FILENAME))

    def save(self, folder: str) -> None:
        with open(
            os.path.join(folder, _MODEL_PARAMS_FILENAME), "w", encoding="utf8"
        ) as handle:
            json.dump(
                {
                    _ITERATIONS_KEY: self._iterations,
                    _LEARNING_RATE_KEY: self._learning_rate,
                    _DEPTH_KEY: self._depth,
                    _L2_LEAF_REG_KEY: self._l2_leaf_reg,
                    _BOOSTING_TYPE_KEY: self._boosting_type,
                    _MODEL_TYPE_KEY: str(self._model_type),
                    _EARLY_STOPPING_ROUNDS: self._early_stopping_rounds,
                },
                handle,
            )
        catboost = self._provide_catboost()
        catboost.save_model(os.path.join(folder, _MODEL_FILENAME))

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
        if eval_x is None:
            raise ValueError("eval_x is null.")
        self._model_type = determine_model_type(y)
        catboost = self._provide_catboost()

        train_pool = Pool(
            df,
            label=y,
            weight=w,
            cat_features=df.select_dtypes(include="category").columns.tolist(),
        )
        eval_pool = Pool(
            eval_x,
            label=eval_y,
            cat_features=eval_x.select_dtypes(include="category").columns.tolist(),
        )
        catboost.fit(
            train_pool,
            early_stopping_rounds=self._early_stopping_rounds,
            verbose=False,
            metric_period=100,
            eval_set=eval_pool,
        )
        importances = catboost.get_feature_importance(prettified=True)
        logging.info("Importances:\n%s", importances)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pred_pool = Pool(
            df,
            cat_features=df.select_dtypes(include="category").columns.tolist(),
        )
        catboost = self._provide_catboost()
        pred = catboost.predict(pred_pool)
        df = pd.DataFrame(
            index=df.index,
            data={
                PREDICTION_COLUMN: pred.flatten(),
            },
        )
        if self._model_type != ModelType.REGRESSION:
            proba = catboost.predict_proba(pred_pool)  # type: ignore
            for i in range(proba.shape[1]):
                df[f"{PROBABILITY_COLUMN_PREFIX}{i}"] = proba[:, i]
        return df

    def _provide_catboost(self) -> CatBoost:
        catboost = self._catboost
        if catboost is None:
            match self._model_type:
                case ModelType.BINARY:
                    catboost = CatBoostClassifierWrapper(
                        iterations=self._iterations,
                        learning_rate=self._learning_rate,
                        depth=self._depth,
                        l2_leaf_reg=self._l2_leaf_reg,
                        boosting_type=self._boosting_type,
                        early_stopping_rounds=self._early_stopping_rounds,
                        metric_period=100,
                        task_type="GPU" if torch.cuda.is_available() else "CPU",
                        devices="0" if torch.cuda.is_available() else None,
                    )
                case ModelType.REGRESSION:
                    catboost = CatBoostRegressorWrapper(
                        iterations=self._iterations,
                        learning_rate=self._learning_rate,
                        depth=self._depth,
                        l2_leaf_reg=self._l2_leaf_reg,
                        boosting_type=self._boosting_type,
                        early_stopping_rounds=self._early_stopping_rounds,
                        metric_period=100,
                        task_type="GPU" if torch.cuda.is_available() else "CPU",
                        devices="0" if torch.cuda.is_available() else None,
                    )
                case ModelType.BINNED_BINARY:
                    catboost = CatBoostClassifierWrapper(
                        iterations=self._iterations,
                        learning_rate=self._learning_rate,
                        depth=self._depth,
                        l2_leaf_reg=self._l2_leaf_reg,
                        boosting_type=self._boosting_type,
                        early_stopping_rounds=self._early_stopping_rounds,
                        metric_period=100,
                        task_type="GPU" if torch.cuda.is_available() else "CPU",
                        devices="0" if torch.cuda.is_available() else None,
                    )
                case ModelType.MULTI_CLASSIFICATION:
                    catboost = CatBoostClassifierWrapper(
                        iterations=self._iterations,
                        learning_rate=self._learning_rate,
                        depth=self._depth,
                        l2_leaf_reg=self._l2_leaf_reg,
                        boosting_type=self._boosting_type,
                        early_stopping_rounds=self._early_stopping_rounds,
                        metric_period=100,
                        task_type="GPU" if torch.cuda.is_available() else "CPU",
                        devices="0" if torch.cuda.is_available() else None,
                    )
            self._catboost = catboost
        if catboost is None:
            raise ValueError("catboost is null")
        return catboost
