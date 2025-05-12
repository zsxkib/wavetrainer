"""The selector class."""

# pylint: disable=too-many-locals
import functools
import json
import logging
import os
from typing import Self

import optuna
import pandas as pd

from ..fit import Fit
from ..model.model import Model
from ..params import Params

_SELECTOR_FILE = "selector.json"


class Selector(Params, Fit):
    """The selector class."""

    # pylint: disable=too-many-positional-arguments,too-many-arguments,consider-using-enumerate

    _selector: list[str] | None

    def __init__(self, model: Model):
        super().__init__()
        self._model = model
        self._feature_ratio = 0.0
        self._steps = 0
        self._selector = None

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        self._feature_ratio = trial.suggest_float("feature_ratio", 0.0, 1.0)
        self._steps = trial.suggest_int("steps", 1, 10)

    def load(self, folder: str) -> None:
        with open(os.path.join(folder, _SELECTOR_FILE), encoding="utf8") as handle:
            self._selector = json.load(handle)

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        with open(os.path.join(folder, _SELECTOR_FILE), "w", encoding="utf8") as handle:
            json.dump(self._selector, handle)

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        if not self._model.supports_importances:
            return self
        if not isinstance(y, pd.Series):
            raise ValueError("y is not a series.")
        if len(df.columns) <= 1:
            return self
        n_features_to_select = max(1, int(len(df.columns) * self._feature_ratio))
        steps = int((len(df.columns) - n_features_to_select) / self._steps)
        current_features = df.columns.values.tolist()
        self._model.fit(df, y=y, w=w, eval_x=eval_x, eval_y=eval_y)

        def set_current_features():
            nonlocal current_features
            feature_importances = self._model.feature_importances
            if not feature_importances:
                return
            current_features = sorted(
                [
                    x
                    for x in current_features
                    if x in feature_importances and feature_importances[x] > 0.0
                ],
                key=functools.partial(lambda x, f: f[x], f=feature_importances),
                reverse=True,
            )
            if not current_features:
                current_features = [list(feature_importances.keys())[0]]

        for i in range(steps):
            print(
                f"Recursive Feature Elimination Step {i}, current features: {len(current_features)}"
            )
            ratio_diff = 1.0 - self._feature_ratio
            ratio_step = ratio_diff / float(steps)
            current_ratio = 1.0 - (ratio_step * i)
            n_features = max(1, int(len(df.columns) * current_ratio))
            if n_features >= len(current_features):
                continue
            set_current_features()
            print(f"Reduced features to {len(current_features)}")
            df = df[current_features]
            if eval_x is not None:
                eval_x = eval_x[current_features]
            self._model.fit(df, y=y, w=w, eval_x=eval_x, eval_y=eval_y)
        set_current_features()
        self._selector = current_features

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df.columns) <= 1:
            return df
        selector = self._selector
        if selector is None:
            logging.warning("selector is null")
            return df
        return df[self._selector]  # type: ignore
