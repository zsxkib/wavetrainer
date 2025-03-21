"""A reducer that removes non-numeric columns."""

from typing import Self

import optuna
import pandas as pd

from .reducer import Reducer


class NonNumericReducer(Reducer):
    """A class that removes non numeric columns from a dataframe."""

    # pylint: disable=too-many-positional-arguments,too-many-arguments

    @classmethod
    def name(cls) -> str:
        return "nonnumeric"

    def set_options(self, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        pass

    def load(self, folder: str) -> None:
        pass

    def save(self, folder: str) -> None:
        pass

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        categorical_cols = df.select_dtypes(include="category").columns.tolist()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        keep_cols = categorical_cols + numeric_cols
        return df[keep_cols]
