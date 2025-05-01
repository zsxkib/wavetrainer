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
_BEST_ITERATION_KEY = "best_iteration"


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
    _best_iteration: None | int

    @classmethod
    def name(cls) -> str:
        return "catboost"

    @classmethod
    def supports_x(cls, df: pd.DataFrame) -> bool:
        return True

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
        self._best_iteration = None

    @property
    def estimator(self) -> Any:
        return self._provide_catboost()

    @property
    def supports_importances(self) -> bool:
        return True

    @property
    def feature_importances(self) -> dict[str, float]:
        catboost = self._provide_catboost()
        importances = catboost.get_feature_importance(prettified=True)
        if importances is None:
            raise ValueError("importances is null")
        feature_ids = importances["Feature Id"].to_list()  # type: ignore
        importances = importances["Importances"].to_list()  # type: ignore
        return {feature_ids[x]: importances[x] for x in range(len(feature_ids))}

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
            EVAL_SET_ARG_KEY: (eval_x, eval_y),
            CAT_FEATURES_ARG_KEY: df.select_dtypes(include="category").columns.tolist(),
            ORIGINAL_X_ARG_KEY: df,
            "sample_weight": w,
        }

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        self._iterations = trial.suggest_int(_ITERATIONS_KEY, 100, 10000)
        self._learning_rate = trial.suggest_float(_LEARNING_RATE_KEY, 0.001, 0.3)
        self._depth = trial.suggest_int(_DEPTH_KEY, 1, 10)
        self._l2_leaf_reg = trial.suggest_float(_L2_LEAF_REG_KEY, 3.0, 50.0)
        self._boosting_type = trial.suggest_categorical(
            _BOOSTING_TYPE_KEY, ["Ordered", "Plain"]
        )
        self._early_stopping_rounds = trial.suggest_int(_EARLY_STOPPING_ROUNDS, 10, 500)
        self._best_iteration = trial.user_attrs.get(_BEST_ITERATION_KEY)

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
            self._best_iteration = params.get(_BEST_ITERATION_KEY)
        catboost = self._provide_catboost()
        catboost.load_model(os.path.join(folder, _MODEL_FILENAME))

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
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
                    _BEST_ITERATION_KEY: self._best_iteration,
                },
                handle,
            )
        catboost = self._provide_catboost()
        catboost.save_model(os.path.join(folder, _MODEL_FILENAME))
        trial.user_attrs[_BEST_ITERATION_KEY] = self._best_iteration

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
        catboost = self._provide_catboost()

        train_pool = Pool(
            df,
            label=y,
            weight=w,
            cat_features=df.select_dtypes(include="category").columns.tolist(),
        )
        eval_pool = (
            Pool(
                eval_x,
                label=eval_y,
                cat_features=eval_x.select_dtypes(include="category").columns.tolist(),
            )
            if eval_x is not None and self._best_iteration is not None
            else None
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
        self._best_iteration = catboost.get_best_iteration()
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
            best_iteration = self._best_iteration
            iterations = (
                best_iteration if best_iteration is not None else self._iterations
            )
            match self._model_type:
                case ModelType.BINARY:
                    catboost = CatBoostClassifierWrapper(
                        iterations=iterations,
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
                        iterations=iterations,
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
                        iterations=iterations,
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
                        iterations=iterations,
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
