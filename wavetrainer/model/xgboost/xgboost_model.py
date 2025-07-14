"""A model that wraps xgboost."""
# pylint: disable=duplicate-code,too-many-arguments,too-many-positional-arguments,too-many-instance-attributes,line-too-long

import json
import os
from typing import Self

import jax.numpy as jnp
import numpy as np
import optuna
import pandas as pd
import pytest_is_running
import torch
from jax import grad, hessian, vmap
from xgboost import XGBClassifier, XGBRegressor
from xgboost.callback import TrainingCallback
from xgboost.core import XGBoostError

from ...exceptions import WavetrainException
from ...model_type import ModelType, determine_model_type
from ..model import PREDICTION_COLUMN, PROBABILITY_COLUMN_PREFIX, Model
from .early_stopper import XGBoostEarlyStoppingCallback
from .xgboost_logger import XGBoostEpochsLogger

_MODEL_FILENAME = "xgboost_model.json"
_MODEL_PARAMS_FILENAME = "xgboost_model_params.json"
_MODEL_TYPE_KEY = "model_type"
_BEST_ITERATION_KEY = "best_iteration"
_BOOSTER_KEY = "booster"
_LAMBDA_KEY = "lambda"
_ALPHA_KEY = "alpha"
_SUBSAMPLE_KEY = "subsample"
_COLSAMPLE_BYTREE_KEY = "colsample_bytree"
_MAX_DEPTH_KEY = "max_depth"
_MIN_CHILD_WEIGHT_KEY = "min_child_weight"
_ETA_KEY = "eta"
_GAMMA_KEY = "gamma"
_GROW_POLICY_KEY = "grow_policy"
_SAMPLE_TYPE_KEY = "sample_type"
_NORMALIZE_TYPE_KEY = "normalize_type"
_RATE_DROP_KEY = "rate_drop"
_SKIP_DROP_KEY = "skip_drop"
_NUM_BOOST_ROUNDS_KEY = "num_boost_rounds"
_EARLY_STOPPING_ROUNDS_KEY = "early_stopping_rounds"
_LOSS_FUNCTION_KEY = "xgboost_loss_function"
_DEFAULT_LOSS_FUNCTION = "default"
_FOCALLOSS_LOSS_FUNCTION = "focalloss"
_FOCALLOSS_GAMMA_KEY = "focalloss_gamma"
_FOCALLOSS_ALPHA_KEY = "focalloss_alpha"


def _convert_categoricals(input_df: pd.DataFrame) -> pd.DataFrame:
    output_df = input_df.copy()
    for col in input_df.select_dtypes(include=["category"]).columns:
        output_df[col] = output_df[col].cat.codes
    return output_df.replace([np.inf, -np.inf], np.nan)


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
    _focalloss_alpha: float | None
    _focalloss_gamma: float | None
    _loss_function: str | None

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
        self._loss_function = None
        self._focalloss_gamma = None
        self._focalloss_alpha = None

    @property
    def supports_importances(self) -> bool:
        return True

    def feature_importances(
        self, df: pd.DataFrame | None
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        bst = self._provide_xgboost()
        try:
            score_dict = bst.get_booster().get_score(importance_type="weight")  # type: ignore
            total = sum(score_dict.values())  # type: ignore
            return {
                k: 0.0 if total == 0.0 else v / total  # type: ignore
                for k, v in score_dict.items()  # type: ignore
            }, []  # type: ignore
        except XGBoostError as exc:
            print(str(exc))
            return {}, []

    def provide_estimator(self):
        return self._provide_xgboost()

    def create_estimator(self):
        return self._create_xgboost()

    def reset(self):
        self._xgboost = None
        self._best_iteration = None

    def convert_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return _convert_categoricals(df)

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        self._booster = trial.suggest_categorical(
            _BOOSTER_KEY, ["gbtree", "gblinear", "dart"]
        )
        self._lambda = trial.suggest_float(_LAMBDA_KEY, 1e-8, 1.0, log=True)
        self._alpha = trial.suggest_float(_ALPHA_KEY, 1e-8, 1.0, log=True)
        self._subsample = trial.suggest_float(_SUBSAMPLE_KEY, 0.2, 1.0)
        self._colsample_bytree = trial.suggest_float(_COLSAMPLE_BYTREE_KEY, 0.2, 1.0)
        if self._booster in ["gbtree", "dart"]:
            self._max_depth = trial.suggest_int(
                _MAX_DEPTH_KEY, 3, 4 if pytest_is_running.is_running() else 8
            )
            self._min_child_weight = trial.suggest_int(
                _MIN_CHILD_WEIGHT_KEY, 2, 10, log=True
            )
            self._eta = trial.suggest_float(_ETA_KEY, 1e-8, 1.0, log=True)
            self._gamma = trial.suggest_float(_GAMMA_KEY, 1e-8, 1.0, log=True)
            self._grow_policy = trial.suggest_categorical(
                _GROW_POLICY_KEY, ["depthwise", "lossguide"]
            )
        else:
            self._sample_type = trial.suggest_categorical(
                _SAMPLE_TYPE_KEY, ["uniform", "weighted"]
            )
            self._normalize_type = trial.suggest_categorical(
                _NORMALIZE_TYPE_KEY, ["tree", "forest"]
            )
            self._rate_drop = trial.suggest_float(_RATE_DROP_KEY, 1e-8, 1.0, log=True)
            self._skip_drop = trial.suggest_float(_SKIP_DROP_KEY, 1e-8, 1.0, log=True)
        self._num_boost_rounds = trial.suggest_int(
            _NUM_BOOST_ROUNDS_KEY, 100, 110 if pytest_is_running.is_running() else 10000
        )
        self._early_stopping_rounds = trial.suggest_int(
            _EARLY_STOPPING_ROUNDS_KEY, 50, 500
        )
        self._best_iteration = trial.user_attrs.get(_BEST_ITERATION_KEY)
        loss_function = trial.suggest_categorical(
            _LOSS_FUNCTION_KEY, [_DEFAULT_LOSS_FUNCTION, _FOCALLOSS_LOSS_FUNCTION]
        )
        self._loss_function = loss_function
        if loss_function == _FOCALLOSS_LOSS_FUNCTION:
            self._focalloss_gamma = trial.suggest_float(_FOCALLOSS_GAMMA_KEY, 0.5, 5.0)
            self._focalloss_alpha = trial.suggest_float(
                _FOCALLOSS_ALPHA_KEY, 0.05, 0.95
            )

    def load(self, folder: str) -> None:
        with open(
            os.path.join(folder, _MODEL_PARAMS_FILENAME), encoding="utf8"
        ) as handle:
            params = json.load(handle)
            self._model_type = ModelType(params[_MODEL_TYPE_KEY])
            self._booster = params[_BOOSTER_KEY]
            self._lambda = params[_LAMBDA_KEY]
            self._alpha = params[_ALPHA_KEY]
            self._subsample = params[_SUBSAMPLE_KEY]
            self._colsample_bytree = params[_COLSAMPLE_BYTREE_KEY]
            self._max_depth = params[_MAX_DEPTH_KEY]
            self._min_child_weight = params[_MIN_CHILD_WEIGHT_KEY]
            self._eta = params[_ETA_KEY]
            self._gamma = params[_GAMMA_KEY]
            self._grow_policy = params[_GROW_POLICY_KEY]
            self._sample_type = params[_SAMPLE_TYPE_KEY]
            self._normalize_type = params[_NORMALIZE_TYPE_KEY]
            self._rate_drop = params[_RATE_DROP_KEY]
            self._skip_drop = params[_SKIP_DROP_KEY]
            self._num_boost_rounds = params[_NUM_BOOST_ROUNDS_KEY]
            self._early_stopping_rounds = params[_EARLY_STOPPING_ROUNDS_KEY]
            self._best_iteration = params.get(_BEST_ITERATION_KEY)
            self._loss_function = params.get(_LOSS_FUNCTION_KEY, _DEFAULT_LOSS_FUNCTION)
            self._focalloss_gamma = params.get(_FOCALLOSS_GAMMA_KEY)
            self._focalloss_alpha = params.get(_FOCALLOSS_ALPHA_KEY)
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
                    _BOOSTER_KEY: self._booster,
                    _LAMBDA_KEY: self._lambda,
                    _ALPHA_KEY: self._alpha,
                    _SUBSAMPLE_KEY: self._subsample,
                    _COLSAMPLE_BYTREE_KEY: self._colsample_bytree,
                    _MAX_DEPTH_KEY: self._max_depth,
                    _MIN_CHILD_WEIGHT_KEY: self._min_child_weight,
                    _ETA_KEY: self._eta,
                    _GAMMA_KEY: self._gamma,
                    _GROW_POLICY_KEY: self._grow_policy,
                    _SAMPLE_TYPE_KEY: self._sample_type,
                    _NORMALIZE_TYPE_KEY: self._normalize_type,
                    _RATE_DROP_KEY: self._rate_drop,
                    _SKIP_DROP_KEY: self._skip_drop,
                    _NUM_BOOST_ROUNDS_KEY: self._num_boost_rounds,
                    _EARLY_STOPPING_ROUNDS_KEY: self._early_stopping_rounds,
                    _LOSS_FUNCTION_KEY: self._loss_function,
                    _FOCALLOSS_GAMMA_KEY: self._gamma,
                    _FOCALLOSS_ALPHA_KEY: self._alpha,
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
        xgboost = self._create_xgboost()
        self._xgboost = xgboost
        df = _convert_categoricals(df)
        evals = [(df, y)]
        if eval_x is not None and eval_y is not None and self._best_iteration is None:
            eval_x = _convert_categoricals(eval_x)
            evals = [(eval_x, eval_y), (df, y)]
        if w is not None:
            w = w.fillna(0.0).clip(lower=0.00001)
        xgboost.fit(  # type: ignore
            df,
            y,
            eval_set=evals,
            sample_weight=w,
            verbose=False,
        )
        if eval_x is not None and eval_y is not None and self._best_iteration is None:
            self._best_iteration = xgboost.best_iteration
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x_df = _convert_categoricals(df)
        xgboost = self._provide_xgboost()
        try:
            pred = xgboost.predict(x_df)
        except XGBoostError as exc:
            raise WavetrainException() from exc
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
            xgboost = self._create_xgboost()
            self._xgboost = xgboost
        if xgboost is None:
            raise ValueError("xgboost is null")
        return xgboost

    def _create_xgboost(self) -> XGBClassifier | XGBRegressor:
        callbacks: list[TrainingCallback] = [
            XGBoostEpochsLogger(),
        ]
        best_iteration = self._best_iteration
        if best_iteration is None:
            callbacks.append(
                XGBoostEarlyStoppingCallback(rounds=self._early_stopping_rounds)
            )
        param = {
            "objective": "binary:logistic",
            "tree_method": "hist" if torch.cuda.is_available() else "exact",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
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
            "n_estimators": best_iteration
            if best_iteration is not None
            else self._num_boost_rounds,
            "base_score": 0.5,
            "callbacks": callbacks,
            "eval_metric": ["logloss", "error"],
            "use_label_encoder": False,
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
        if (
            self._loss_function == _FOCALLOSS_LOSS_FUNCTION
            and self._focalloss_alpha is not None
            and self._focalloss_gamma is not None
        ):

            def focal_loss(alpha=0.25, gamma=2.0):
                def fl(x, t):
                    p = 1 / (1 + jnp.exp(-x))
                    pt = t * p + (1 - t) * (1 - p)
                    alpha_t = alpha * t + (1 - alpha) * (1 - t)
                    return (
                        -alpha_t * (1 - pt) ** gamma * jnp.log(jnp.clip(pt, 1e-8, 1.0))
                    )

                fl_grad = grad(fl)
                fl_hess = hessian(fl)
                grad_batch = vmap(fl_grad)
                hess_batch = vmap(fl_hess)

                def custom_loss(y_pred, y_true, sample_weight=None):
                    y_true = jnp.array(y_true)
                    y_pred = jnp.array(y_pred)

                    grad_vals = grad_batch(y_pred, y_true)
                    hess_vals = hess_batch(y_pred, y_true)

                    if sample_weight is not None:
                        sample_weight = jnp.array(sample_weight)
                        grad_vals *= sample_weight
                        hess_vals *= sample_weight

                    # Convert to NumPy arrays for XGBoost compatibility
                    return np.array(grad_vals), np.array(hess_vals)

                return custom_loss

            param["objective"] = focal_loss(
                alpha=self._focalloss_alpha, gamma=self._focalloss_gamma
            )
        print(
            f"Creating xgboost model with max_depth {self._max_depth}, best iteration {best_iteration}, booster: {self._booster}",
        )
        bst = None
        match self._model_type:
            case ModelType.BINARY:
                bst = XGBClassifier(**param)
            case ModelType.REGRESSION:
                param["objective"] = "reg:squarederror"
                param["eval_metric"] = ["rmse", "mae"]
                bst = XGBRegressor(**param)
            case ModelType.BINNED_BINARY:
                bst = XGBClassifier(**param)
            case ModelType.MULTI_CLASSIFICATION:
                bst = XGBClassifier(**param)
            case _:
                raise ValueError(f"Unrecognised model type: {self._model_type}")
        return bst
