"""The prototype calibrator class."""

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
