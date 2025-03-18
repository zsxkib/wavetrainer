"""The selector class."""

import logging
import os
from typing import Self

import joblib  # type: ignore
import optuna
import pandas as pd
from sklearn.feature_selection import RFE  # type: ignore

from ..fit import Fit
from ..model.model import Model
from ..params import Params

_SELECTOR_FILE = "selector.joblib"


class Selector(Params, Fit):
    """The selector class."""

    def __init__(self, model: Model, total_features: int):
        super().__init__()
        self._model = model
        self._feature_ratio = 0.0
        self._steps = 0
        n_features_to_select = max(1, total_features * self._feature_ratio)
        self._selector = RFE(
            model.estimator,
            n_features_to_select=n_features_to_select,
            step=self._steps,
            verbose=1,
        )

    def set_options(self, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        self._feature_ratio = trial.suggest_float("feature_ratio", 0.0, 1.0)
        steps = trial.suggest_int("steps", 1, 16)
        self._steps = steps
        self._selector.step = steps

    def load(self, folder: str) -> None:
        self._selector = joblib.load(os.path.join(folder, _SELECTOR_FILE))

    def save(self, folder: str) -> None:
        joblib.dump(self._selector, os.path.join(folder, _SELECTOR_FILE))

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
    ) -> Self:
        if not isinstance(y, pd.Series):
            raise ValueError("y is not a series.")

        try:
            self._selector.fit(df, y=y, sample_weight=w)
        except ValueError as exc:
            # Catch issues with 1 feature as a reduction target.
            logging.warning(str(exc))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            return df[self._selector.get_feature_names_out()]
        except AttributeError as exc:
            # Catch issues with 1 feature as a reduction target.
            logging.warning(str(exc))
            return df
