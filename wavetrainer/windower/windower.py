"""The windower class."""

import datetime
import json
import os
from typing import Self

import optuna
import pandas as pd

from ..fit import Fit
from ..params import Params

_WINDOWER_FILE = "windower.json"
_LOOKBACK_KEY = "lookback"


class Windower(Params, Fit):
    """The windower class."""

    _lookback_ratio: float | None

    def __init__(self, dt_column: str | None):
        super().__init__()
        self._lookback = None
        self._lookback_ratio = None
        self._dt_column = dt_column

    def set_options(self, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        self._lookback_ratio = trial.suggest_float("lookback", 0.1, 1.0)

    def load(self, folder: str) -> None:
        with open(os.path.join(folder, _WINDOWER_FILE), encoding="utf8") as handle:
            params = json.load(handle)
            self._lookback = params[_LOOKBACK_KEY]

    def save(self, folder: str) -> None:
        with open(os.path.join(folder, _WINDOWER_FILE), "w", encoding="utf8") as handle:
            json.dump(
                {
                    _LOOKBACK_KEY: self._lookback,
                },
                handle,
            )

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
    ) -> Self:
        lookback_ratio = self._lookback_ratio
        if lookback_ratio is None:
            raise ValueError("lookback_ratio is null")
        dt_index = df.index if self._dt_column is None else df[self._dt_column]
        start_idx = dt_index[int(len(df) * lookback_ratio)]
        end_idx = dt_index[-1]
        td = end_idx.to_pydatetime() - start_idx.to_pydatetime()
        self._lookback = td.total_seconds()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        lookback = self._lookback
        if lookback is None:
            raise ValueError("lookback is null")
        dt_index = df.index if self._dt_column is None else df[self._dt_column]
        return df[
            dt_index
            >= dt_index[-1].to_pydatetime() - datetime.timedelta(seconds=lookback)
        ]
