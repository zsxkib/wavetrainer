"""A reducer that uses a base selector from the feature engine."""

import logging
import os
from typing import Self

import joblib  # type: ignore
import optuna
import pandas as pd
from feature_engine.selection.base_selector import BaseSelector

from ..exceptions import WavetrainException
from .reducer import Reducer


class BaseSelectorReducer(Reducer):
    """A class that uses the base selector from the feature engine."""

    # pylint: disable=too-many-positional-arguments,too-many-arguments

    def __init__(self, base_selector: BaseSelector, file_name: str) -> None:
        super().__init__()
        self._base_selector = base_selector
        self._file_name = file_name

    @classmethod
    def name(cls) -> str:
        raise NotImplementedError("name not implemented in parent class.")

    @classmethod
    def should_raise(cls) -> bool:
        """Whether the class should raise its exception if it encounters it."""
        return True

    def set_options(self, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        pass

    def load(self, folder: str) -> None:
        file_path = os.path.join(folder, self._file_name)
        self._base_selector = joblib.load(file_path)

    def save(self, folder: str) -> None:
        file_path = os.path.join(folder, self._file_name)
        joblib.dump(self._base_selector, file_path)

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        if len(df.columns) <= 1:
            return self
        try:
            self._base_selector.fit(df)  # type: ignore
        except ValueError as exc:
            logging.warning(str(exc))
            if self.should_raise():
                raise WavetrainException() from exc
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df.columns) <= 1:
            return df
        return self._base_selector.transform(df)
