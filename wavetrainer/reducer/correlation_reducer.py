"""A reducer that removes correlation features."""

# pylint: disable=too-many-arguments,too-many-positional-arguments,consider-using-enumerate
import json
import os
from typing import Self

import numpy as np
import optuna
import pandas as pd

from .non_categorical_numeric_columns import \
    find_non_categorical_numeric_columns
from .reducer import Reducer

_CORRELATION_REDUCER_FILENAME = "correlation_reducer.json"
_CORRELATION_REDUCER_THRESHOLD = "correlation_reducer_threshold"


def _get_correlated_features_to_drop(
    df: pd.DataFrame, threshold: float = 0.85, random_seed: int = 42
) -> list[str]:
    """
    Identify highly correlated features to drop, keeping one per group.
    NaNs are replaced with a single fixed junk value to allow correlation computation.
    Columns are processed in sorted order to ensure deterministic output.

    Args:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Correlation threshold above which features are considered redundant.
        random_seed (int): Seed used to generate the fixed junk value.

    Returns:
        List[str]: List of column names to drop.
    """
    np.random.seed(random_seed)

    # Select and sort numeric columns
    sorted_cols = sorted(find_non_categorical_numeric_columns(df))
    df_numeric = df[sorted_cols].copy()

    # Generate and apply a fixed junk value for NaNs
    junk_value = np.random.uniform(-1e9, 1e9)
    df_numeric = df_numeric.fillna(junk_value)

    if df_numeric.shape[1] < 2:
        return []

    # Compute absolute correlation matrix
    corr_matrix = np.corrcoef(df_numeric.values, rowvar=False)
    abs_corr = np.abs(corr_matrix)

    # Greedy feature drop based on sorted order
    to_drop = set()
    for i in range(len(sorted_cols)):
        if sorted_cols[i] in to_drop:
            continue
        for j in range(i + 1, len(sorted_cols)):
            if sorted_cols[j] in to_drop:
                continue
            if abs_corr[i, j] > threshold:
                to_drop.add(sorted_cols[j])

    return sorted(to_drop)


class CorrelationReducer(Reducer):
    """A class that removes correlated values from a dataset."""

    _correlation_drop_features: dict[str, bool]

    def __init__(self) -> None:
        self._threshold = 0.0
        self._correlation_drop_features = {}

    @classmethod
    def name(cls) -> str:
        return "correlation"

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        self._threshold = trial.suggest_float(_CORRELATION_REDUCER_THRESHOLD, 0.7, 0.99)

    def load(self, folder: str) -> None:
        with open(
            os.path.join(folder, _CORRELATION_REDUCER_FILENAME), encoding="utf8"
        ) as handle:
            self._correlation_drop_features = json.load(handle)

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        with open(
            os.path.join(folder, _CORRELATION_REDUCER_FILENAME), "w", encoding="utf8"
        ) as handle:
            json.dump(self._correlation_drop_features, handle)

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        drop_features = _get_correlated_features_to_drop(df, threshold=self._threshold)
        self._correlation_drop_features = {x: True for x in drop_features}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(
            columns=list(self._correlation_drop_features.keys()), errors="ignore"
        )
