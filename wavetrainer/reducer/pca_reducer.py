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

    def __init__(self, max_features: int | None):
        super().__init__()
        self._max_features = max_features
        if max_features is not None:
            self._scaler = StandardScaler()
            self._pca = PCA(n_components=max_features)
        else:
            self._scaler = None
            self._pca = None

    @classmethod
    def name(cls) -> str:
        return "pca"

    def set_options(self, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        pass

    def load(self, folder: str) -> None:
        pca_scaler_file = os.path.join(folder, _PCA_SCALER_FILE)
        pca_file = os.path.join(folder, _PCA_FILE)
        if os.path.exists(pca_scaler_file):
            self._scaler = joblib.load(pca_scaler_file)
        if os.path.exists(pca_file):
            self._pca = joblib.load(pca_file)

    def save(self, folder: str) -> None:
        if self._scaler is not None:
            joblib.dump(self._scaler, os.path.join(folder, _PCA_SCALER_FILE))
        if self._pca is not None:
            joblib.dump(self._pca, os.path.join(folder, _PCA_FILE))

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        pca = self._pca
        scaler = self._scaler
        if pca is None or scaler is None:
            return self
        if len(df.columns.values) < pca.n_components:  # type: ignore
            return self
        x_scaled = scaler.fit_transform(df)
        pca.fit(x_scaled)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._pca is None:
            return df
        if len(df.columns.values) < self._pca.n_components:  # type: ignore
            return df
        return self._pca.transform(df)
