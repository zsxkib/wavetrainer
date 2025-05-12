"""The prototype calibrator class."""

import pandas as pd

from ..fit import Fit
from ..model.model import Model
from ..params import Params


class Calibrator(Params, Fit):
    """The prototype calibrator class."""

    def __init__(self, model: Model):
        self._model = model

    @classmethod
    def name(cls) -> str:
        """The name of the calibrator."""
        raise NotImplementedError("name not implemented in parent class.")

    def predictions_as_x(self, y: pd.Series | pd.DataFrame | None = None) -> bool:
        """Whether the calibrator wants predictions as X rather than features."""
        raise NotImplementedError("predictions_as_x not implemented in parent class.")
