"""A reducer that removes correlation features via further heuristics."""

import optuna
import pandas as pd
from feature_engine.selection import SmartCorrelatedSelection

from .base_selector_reducer import BaseSelectorReducer

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
            _SMART_CORRELATION_REDUCER_THRESHOLD, 0.1, 0.9
        )
