"""A calibrator class that routes to other calibrators."""

import json
import logging
import os
from typing import Self

import optuna
import pandas as pd

from ..model.model import Model
from ..model_type import ModelType, determine_model_type
from .calibrator import Calibrator
from .vennabers_calibrator import VennabersCalibrator

_CALIBRATOR_ROUTER_FILE = "calibrator_router.json"
_CALIBRATOR_KEY = "calibrator"
_CALIBRATORS = {
    VennabersCalibrator.name(): VennabersCalibrator,
}


class CalibratorRouter(Calibrator):
    """A router that routes to a different calibrator class."""

    # pylint: disable=too-many-positional-arguments,too-many-arguments

    _calibrator: Calibrator | None

    def __init__(self, model: Model):
        super().__init__(model)
        self._calibrator = None

    @classmethod
    def name(cls) -> str:
        return "router"

    def predictions_as_x(self, y: pd.Series | pd.DataFrame | None = None) -> bool:
        calibrator = self._calibrator
        if calibrator is not None:
            return calibrator.predictions_as_x(None)
        return True

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        calibrator = self._calibrator
        if calibrator is None:
            return
        calibrator.set_options(trial, df)

    def load(self, folder: str) -> None:
        file_path = os.path.join(folder, _CALIBRATOR_ROUTER_FILE)
        if not os.path.exists(file_path):
            logging.warning("file %s does not exist", file_path)
            return
        with open(file_path, encoding="utf8") as handle:
            params = json.load(handle)
            calibrator = _CALIBRATORS[params[_CALIBRATOR_KEY]](self._model)
        calibrator.load(folder)
        self._calibrator = calibrator

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        calibrator = self._calibrator
        if calibrator is None:
            logging.warning("calibrator is null")
            return
        calibrator.save(folder, trial)
        with open(
            os.path.join(folder, _CALIBRATOR_ROUTER_FILE), "w", encoding="utf8"
        ) as handle:
            json.dump(
                {
                    _CALIBRATOR_KEY: calibrator.name(),
                },
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
        # pylint: disable=no-else-return
        calibrator: Calibrator | None = None
        if y is None:
            raise ValueError("y is null")
        if determine_model_type(y) == ModelType.REGRESSION:
            return self
        else:
            calibrator = VennabersCalibrator(self._model)
        calibrator.fit(df, y=y, w=w)
        self._calibrator = calibrator
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        calibrator = self._calibrator
        if calibrator is None:
            logging.warning("calibrator is null")
            return df
        return calibrator.transform(df)
