"""A reducer that removes unseen columns."""

import json
import os
from typing import Self

import optuna
import pandas as pd

from .reducer import Reducer

_UNSEEN_REDUCER_FILE = "unseen_reducer.json"


class UnseenReducer(Reducer):
    """A class that removes unseen columns from a dataframe."""

    # pylint: disable=too-many-positional-arguments,too-many-arguments
    _seen_features: list[str]

    def __init__(self, insert_null: bool):
        super().__init__()
        self._seen_features = []
        self._insert_null = insert_null

    @classmethod
    def name(cls) -> str:
        return "unseen"

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        pass

    def load(self, folder: str) -> None:
        with open(
            os.path.join(folder, _UNSEEN_REDUCER_FILE), encoding="utf8"
        ) as handle:
            self._seen_features = json.load(handle)

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        with open(
            os.path.join(folder, _UNSEEN_REDUCER_FILE), "w", encoding="utf8"
        ) as handle:
            json.dump(
                self._seen_features,
                handle,
            )

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        self._seen_features = df.columns.values.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = set(df.columns.values.tolist())
        seen_features = set(self._seen_features)
        if self._insert_null:
            # Are we missing features?
            missing_features = seen_features - cols
            for missing_feature in missing_features:
                print(f"Adding back {missing_features} as null")
                df[missing_feature] = None
                df[missing_feature] = df[missing_feature].astype(float)
        return df[self._seen_features]
