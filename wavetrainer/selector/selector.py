"""The selector class."""

import logging
import os
from typing import Self

import joblib  # type: ignore
import optuna
import pandas as pd
import sklearn  # type: ignore
from sklearn.feature_selection import RFE  # type: ignore

from ..fit import Fit
from ..model.model import Model
from ..params import Params

_SELECTOR_FILE = "selector.joblib"


class Selector(Params, Fit):
    """The selector class."""

    # pylint: disable=too-many-positional-arguments,too-many-arguments,consider-using-enumerate

    _selector: RFE | None

    def __init__(self, model: Model):
        super().__init__()
        self._model = model
        self._feature_ratio = 0.0
        self._steps = 0
        self._selector = None

    def set_options(self, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        self._feature_ratio = trial.suggest_float("feature_ratio", 0.0, 1.0)
        self._steps = trial.suggest_int("steps", 1, 10)

    def load(self, folder: str) -> None:
        self._selector = joblib.load(os.path.join(folder, _SELECTOR_FILE))

    def save(self, folder: str) -> None:
        joblib.dump(self._selector, os.path.join(folder, _SELECTOR_FILE))

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        sklearn.set_config(enable_metadata_routing=False)
        model_kwargs = self._model.pre_fit(df, y=y, eval_x=eval_x, eval_y=eval_y)
        if not isinstance(y, pd.Series):
            raise ValueError("y is not a series.")
        if len(df.columns) <= 1:
            return self
        n_features_to_select = max(1, int(len(df.columns) * self._feature_ratio))
        self._selector = RFE(
            self._model.estimator,
            n_features_to_select=n_features_to_select,
            step=max(
                1,
                int((len(df.columns) - n_features_to_select) / self._steps),
            ),
        )
        try:
            self._selector.fit(df, y=y, sample_weight=w, **model_kwargs)
        except ValueError as exc:
            # Catch issues with 1 feature as a reduction target.
            logging.warning(str(exc))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df.columns) <= 1:
            return df
        selector = self._selector
        if selector is None:
            raise ValueError("selector is null.")
        try:
            return df[selector.get_feature_names_out()]
        except AttributeError as exc:
            # Catch issues with 1 feature as a reduction target.
            logging.warning(str(exc))
            return df
