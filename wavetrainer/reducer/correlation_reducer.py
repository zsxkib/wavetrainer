"""A reducer that removes correlation features."""

# pylint: disable=too-many-arguments,too-many-positional-arguments
from typing import Self

import optuna
import pandas as pd
from feature_engine.selection import DropCorrelatedFeatures

from .base_selector_reducer import BaseSelectorReducer
from .non_categorical_numeric_columns import \
    find_non_categorical_numeric_columns

_CORRELATION_REDUCER_FILENAME = "correlation_reducer.joblib"
_CORRELATION_REDUCER_THRESHOLD = "correlation_reducer_threshold"


class CorrelationReducer(BaseSelectorReducer):
    """A class that removes correlated values from a dataset."""

    def __init__(self) -> None:
        self._correlation_selector = DropCorrelatedFeatures(missing_values="ignore")
        super().__init__(
            self._correlation_selector,
            _CORRELATION_REDUCER_FILENAME,
        )

    @classmethod
    def name(cls) -> str:
        return "correlation"

    @classmethod
    def should_raise(cls) -> bool:
        return False

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        self._correlation_selector.threshold = trial.suggest_float(
            _CORRELATION_REDUCER_THRESHOLD, 0.7, 0.99
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
        return super().fit(df, y=y, w=w, eval_x=eval_x, eval_y=eval_y)
