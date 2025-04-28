"""A reducer that removes correlation features."""

import optuna
import pandas as pd
from feature_engine.selection import DropCorrelatedFeatures

from .base_selector_reducer import BaseSelectorReducer

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
            _CORRELATION_REDUCER_THRESHOLD, 0.1, 0.9
        )
