"""A reducer that combines all the other reducers."""

import json
import logging
import os
from typing import Self

import optuna
import pandas as pd

from .constant_reducer import ConstantReducer
from .correlation_reducer import CorrelationReducer
from .duplicate_reducer import DuplicateReducer
from .nonnumeric_reducer import NonNumericReducer
from .reducer import Reducer
from .unseen_reducer import UnseenReducer

_COMBINED_REDUCER_FILE = "combined_reducer.json"
_REDUCERS_KEY = "reducers"


class CombinedReducer(Reducer):
    """A reducer that combines a series of reducers."""

    # pylint: disable=too-many-positional-arguments,too-many-arguments

    def __init__(self):
        super().__init__()
        self._reducers = [
            UnseenReducer(),
            ConstantReducer(),
            DuplicateReducer(),
            NonNumericReducer(),
            CorrelationReducer(),
        ]

    @classmethod
    def name(cls) -> str:
        return "combined"

    def set_options(self, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        for reducer in self._reducers:
            reducer.set_options(trial)

    def load(self, folder: str) -> None:
        self._reducers = []
        with open(
            os.path.join(folder, _COMBINED_REDUCER_FILE), encoding="utf8"
        ) as handle:
            params = json.load(handle)
            for reducer_name in params[_REDUCERS_KEY]:
                if reducer_name == ConstantReducer.name():
                    self._reducers.append(ConstantReducer())
                elif reducer_name == DuplicateReducer.name():
                    self._reducers.append(DuplicateReducer())
                elif reducer_name == CorrelationReducer.name():
                    self._reducers.append(CorrelationReducer())
                elif reducer_name == NonNumericReducer.name():
                    self._reducers.append(NonNumericReducer())
                elif reducer_name == UnseenReducer.name():
                    self._reducers.append(UnseenReducer())
        for reducer in self._reducers:
            reducer.load(folder)

    def save(self, folder: str) -> None:
        with open(
            os.path.join(folder, _COMBINED_REDUCER_FILE), "w", encoding="utf8"
        ) as handle:
            json.dump(
                {
                    _REDUCERS_KEY: [x.name() for x in self._reducers],
                },
                handle,
            )
        for reducer in self._reducers:
            reducer.save(folder)

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        for reducer in self._reducers:
            df = reducer.fit_transform(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for reducer in self._reducers:
            try:
                df = reducer.transform(df)
            except ValueError as exc:
                logging.warning("Failed to reduce %s", reducer.name())
                raise exc
        return df
