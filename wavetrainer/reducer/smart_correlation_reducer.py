"""A reducer that removes correlation features via further heuristics."""

# pylint: disable=too-many-arguments,too-many-positional-arguments
from typing import Self

import optuna
import pandas as pd
from feature_engine.selection import SmartCorrelatedSelection

from .base_selector_reducer import BaseSelectorReducer
from .non_categorical_numeric_columns import \
    find_non_categorical_numeric_columns

_SMART_CORRELATION_REDUCER_FILENAME = "smart_correlation_reducer.joblib"
_SMART_CORRELATION_REDUCER_THRESHOLD = "smart_correlation_reducer_threshold"


class SmartCorrelationReducer(BaseSelectorReducer):
    """A class that removes smart correlated values from a dataset."""

    def __init__(self) -> None:
        self._correlation_selector = SmartCorrelatedSelection(missing_values="ignore")
        super().__init__(
            self._correlation_selector,
            _SMART_CORRELATION_REDUCER_FILENAME,
        )

    @classmethod
    def name(cls) -> str:
        return "smart_correlation"

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        self._correlation_selector.threshold = trial.suggest_float(
            _SMART_CORRELATION_REDUCER_THRESHOLD, 0.7, 0.99
        )

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        self._correlation_selector.variables = find_non_categorical_numeric_columns(df)
        if len(self._correlation_selector.variables) <= 1:
            return self
        return super().fit(df, y=y, w=w, eval_x=eval_x, eval_y=eval_y)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(find_non_categorical_numeric_columns(df)) <= 1:
            return df
        return super().transform(df)
