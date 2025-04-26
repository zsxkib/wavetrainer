"""A reducer that removes low variance columns."""

import os
from typing import Self

import joblib  # type: ignore
import optuna
import pandas as pd
from sklearn.decomposition import PCA  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from .reducer import Reducer

_PCA_FILE = "pca.joblib"
_PCA_SCALER_FILE = "pca_scaler.joblib"


class PCAReducer(Reducer):
    """A class that removes low variance columns from a dataframe."""

    # pylint: disable=too-many-positional-arguments,too-many-arguments

    def __init__(self):
        super().__init__()
        self._scaler = StandardScaler()
        self._pca = PCA(n_components=300)

    @classmethod
    def name(cls) -> str:
        return "pca"

    def set_options(self, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        pass

    def load(self, folder: str) -> None:
        self._scaler = joblib.load(os.path.join(folder, _PCA_SCALER_FILE))
        self._pca = joblib.load(os.path.join(folder, _PCA_FILE))

    def save(self, folder: str) -> None:
        joblib.dump(self._scaler, os.path.join(folder, _PCA_SCALER_FILE))
        joblib.dump(self._pca, os.path.join(folder, _PCA_FILE))

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        if len(df.columns.values) < self._pca.n_components:  # type: ignore
            return self
        x_scaled = self._scaler.fit_transform(df)
        self._pca.fit(x_scaled)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df.columns.values) < self._pca.n_components:  # type: ignore
            return df
        return self._pca.transform(df)
