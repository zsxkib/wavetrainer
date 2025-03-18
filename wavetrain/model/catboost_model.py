"""A model that wraps catboost."""

import os
from typing import Any, Self

import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool  # type: ignore

from .model import PREDICTION_COLUMN, PROBABILITY_COLUMN_PREFIX, Model

_MODEL_FILENAME = "model.cbm"


class CatboostModel(Model):
    """A class that uses Catboost as a model."""

    @classmethod
    def name(cls) -> str:
        return "catboost"

    def __init__(self) -> None:
        super().__init__()
        self._catboost = CatBoostClassifier()

    @property
    def estimator(self) -> Any:
        return self._catboost

    def set_options(self, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        iterations = trial.suggest_int("iterations", 100, 10000)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.3)
        depth = trial.suggest_int("depth", 1, 12)
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 3.0, 50.0)
        boosting_type = trial.suggest_categorical("boosting_type", ["Ordered", "Plain"])
        self._catboost.set_params(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            boosting_type=boosting_type,
            early_stopping_rounds=100,
        )

    def load(self, folder: str) -> None:
        self._catboost.load_model(os.path.join(folder, _MODEL_FILENAME))

    def save(self, folder: str) -> None:
        self._catboost.save_model(os.path.join(folder, _MODEL_FILENAME))

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
    ) -> Self:
        train_pool = Pool(
            df,
            label=y,
            weight=w,
        )
        self._catboost.fit(
            train_pool,
            early_stopping_rounds=100,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pred_pool = Pool(df)
        pred = self._catboost.predict(pred_pool)
        proba = self._catboost.predict_proba(pred_pool)
        df = pd.DataFrame(
            index=df.index,
            data={
                PREDICTION_COLUMN: pred.flatten(),
            },
        )
        for i in range(proba.shape[1]):
            df[f"{PROBABILITY_COLUMN_PREFIX}{i}"] = proba[:, i]
        return df
