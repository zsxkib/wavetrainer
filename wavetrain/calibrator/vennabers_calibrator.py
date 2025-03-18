"""A calibrator that implements venn abers."""

import os
from typing import Self

import joblib  # type: ignore
import optuna
import pandas as pd
from venn_abers import VennAbers  # type: ignore

from ..model.model import PROBABILITY_COLUMN_PREFIX, Model
from .calibrator import Calibrator

_CALIBRATOR_FILENAME = "vennabers.joblib"


class VennabersCalibrator(Calibrator):
    """A class that uses venn abers as a calibrator."""

    def __init__(self, model: Model):
        super().__init__(model)
        self._vennabers = VennAbers()

    @classmethod
    def name(cls) -> str:
        return "vennabers"

    def set_options(self, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        pass

    def load(self, folder: str) -> None:
        self._vennabers = joblib.load(os.path.join(folder, _CALIBRATOR_FILENAME))

    def save(self, folder: str) -> None:
        joblib.dump(self._vennabers, os.path.join(folder, _CALIBRATOR_FILENAME))

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
    ) -> Self:
        vennabers = self._vennabers
        if vennabers is None:
            raise ValueError("vennabers is null")
        if y is None:
            raise ValueError("y is null")
        prob_columns = [
            x for x in df.columns.values if x.startswith(PROBABILITY_COLUMN_PREFIX)
        ]
        vennabers.fit(df[prob_columns].to_numpy(), y.to_numpy())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        p_prime, _ = self._vennabers.predict_proba(df.to_numpy())
        for i in range(p_prime.shape[1]):
            prob = p_prime[:, i]
            df[f"{PROBABILITY_COLUMN_PREFIX}{i}"] = prob
        return df
