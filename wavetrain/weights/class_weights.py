"""A weight class that computes the imbalance between classification classes."""

from typing import Any, Self

import numpy as np
import optuna
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight  # type: ignore

from ..model_type import ModelType, determine_model_type
from .weights import WEIGHTS_COLUMN, Weights


class ClassWeights(Weights):
    """Class weight class."""

    _class_weights: dict[Any, float]

    def __init__(self) -> None:
        super().__init__()
        self._class_weights = {}

    @classmethod
    def name(cls) -> str:
        """The name of the weight class."""
        return "class"

    def set_options(self, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        pass

    def load(self, folder: str) -> None:
        pass

    def save(self, folder: str) -> None:
        pass

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
    ) -> Self:
        if not isinstance(y, pd.Series):
            raise ValueError("y is not a series.")

        if determine_model_type(y) == ModelType.REGRESSION:
            self._class_weights = {}
            return self

        arr = df.astype(int).to_numpy().flatten().astype(float)
        unique_vals = np.unique(arr)
        w_arr = compute_class_weight(
            class_weight="balanced", classes=unique_vals, y=arr
        )
        for count, unique_val in enumerate(unique_vals):
            self._class_weights[unique_val] = w_arr[count]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._class_weights:
            return pd.DataFrame(
                data=[1.0 for _ in range(len(df))],
                columns=[WEIGHTS_COLUMN],
                index=df.index,
            )

        arr = df.astype(int).to_numpy().flatten().astype(float)
        for k, v in self._class_weights.items():
            arr[arr == k] = v
        return pd.DataFrame(
            data=arr,
            columns=[WEIGHTS_COLUMN],
            index=df.index,
        )
