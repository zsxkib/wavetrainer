"""A reducer that removes features by their single performance via further heuristics."""

# pylint: disable=too-many-arguments,too-many-positional-arguments
from typing import Self

import optuna
import pandas as pd
from feature_engine.selection import SelectBySingleFeaturePerformance
from sklearn.ensemble import RandomForestClassifier  # type: ignore

from ..model_type import ModelType, determine_model_type
from .base_selector_reducer import BaseSelectorReducer

_SINGLE_FEATURE_PERFORMANCE_REDUCER_FILENAME = (
    "single_feature_performance_reducer.joblib"
)
_SINGLE_FEATURE_PERFORMANCE_REDUCER_THRESHOLD = (
    "single_feature_performance_reducer_threshold"
)


class SelectBySingleFeaturePerformanceReducer(BaseSelectorReducer):
    """A class that removes smart correlated values from a dataset."""

    def __init__(self) -> None:
        self._singlefeatureperformance_selector = SelectBySingleFeaturePerformance(
            RandomForestClassifier(random_state=42, n_jobs=-1), scoring="accuracy", cv=1
        )
        super().__init__(
            self._singlefeatureperformance_selector,
            _SINGLE_FEATURE_PERFORMANCE_REDUCER_FILENAME,
        )

    @classmethod
    def name(cls) -> str:
        return "single_feature_performance"

    @classmethod
    def should_raise(cls) -> bool:
        return False

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        self._singlefeatureperformance_selector.threshold = trial.suggest_float(
            _SINGLE_FEATURE_PERFORMANCE_REDUCER_THRESHOLD, 0.1, 0.9
        )

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        self._singlefeatureperformance_selector.scoring = (
            "r2" if determine_model_type(y) == ModelType.REGRESSION else "accuracy"  # type: ignore
        )
        return super().fit(df, y=y, w=w, eval_x=eval_x, eval_y=eval_y)
