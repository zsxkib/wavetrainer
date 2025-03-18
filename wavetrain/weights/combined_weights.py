"""A weights class that combines different weights methods."""

from typing import Self

import optuna
import pandas as pd

from .class_weights import ClassWeights
from .weights import WEIGHTS_COLUMN, Weights
from .weights_router import WeightsRouter


class CombinedWeights(Weights):
    """A weights class that combines multiple weights."""

    def __init__(self) -> None:
        super().__init__()
        self._weights = [WeightsRouter(), ClassWeights()]

    @classmethod
    def name(cls) -> str:
        return "combined"

    def set_options(self, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        for weights in self._weights:
            weights.set_options(trial)

    def load(self, folder: str) -> None:
        for weights in self._weights:
            weights.load(folder)

    def save(self, folder: str) -> None:
        for weights in self._weights:
            weights.save(folder)

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
    ) -> Self:
        for weights in self._weights:
            weights.fit(df, y=y)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_weights: pd.DataFrame | None = None
        for weights in self._weights:
            if df_weights is None:
                df_weights = weights.transform(df)
            else:
                df_weights[WEIGHTS_COLUMN] *= weights.transform(df)[WEIGHTS_COLUMN]
        if df_weights is None:
            raise ValueError("df_weights is null.")
        df_weights[WEIGHTS_COLUMN] = df_weights[WEIGHTS_COLUMN].clip(lower=0.000001)
        return df_weights
