"""A calibrator that implements MAPIE."""

import logging
import os
from typing import Self

import joblib  # type: ignore
import optuna
import pandas as pd
import sklearn  # type: ignore
from mapie.regression import MapieRegressor  # type: ignore

from ..model.model import PROBABILITY_COLUMN_PREFIX, Model
from .calibrator import Calibrator

_CALIBRATOR_FILENAME = "mapie.joblib"


class MAPIECalibrator(Calibrator):
    """A class that uses MAPIE as a calibrator."""

    # pylint: disable=too-many-positional-arguments,too-many-arguments

    def __init__(self, model: Model):
        super().__init__(model)
        self._mapie = MapieRegressor(model.estimator, method="plus")

    @classmethod
    def name(cls) -> str:
        return "mapie"

    def set_options(self, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        pass

    def load(self, folder: str) -> None:
        self._mapie = joblib.load(os.path.join(folder, _CALIBRATOR_FILENAME))

    def save(self, folder: str) -> None:
        joblib.dump(self._mapie, os.path.join(folder, _CALIBRATOR_FILENAME))

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        mapie = self._mapie
        if mapie is None:
            raise ValueError("mapie is null")
        if y is None:
            raise ValueError("y is null")
        if len(df) <= 5:
            return self
        mapie.fit(df.to_numpy(), y.to_numpy())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            alpha = []
            for potential_alpha in [0.05, 0.32]:
                if len(df) > int(1.0 / potential_alpha) + 1:
                    alpha.append(potential_alpha)
            if alpha:
                _, y_pis = self._mapie.predict(df, alpha=alpha)
                for i in range(y_pis.shape[1]):
                    if i >= len(alpha):
                        continue
                    for ii in range(y_pis.shape[2]):
                        alpha_val = alpha[i]
                        values = y_pis[:, i, ii].flatten().tolist()
                        df[f"{PROBABILITY_COLUMN_PREFIX}{alpha_val}_{ii == 1}"] = values
        except sklearn.exceptions.NotFittedError as exc:  # type: ignore
            logging.warning(str(exc))
        return df
