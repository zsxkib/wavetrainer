"""The prototype model class."""

from typing import Any

from ..fit import Fit
from ..params import Params

PREDICTION_COLUMN = "prediction"
PROBABILITY_COLUMN_PREFIX = "probability_"


class Model(Params, Fit):
    """The prototype model class."""

    @classmethod
    def name(cls) -> str:
        """The name of the model."""
        raise NotImplementedError("name not implemented in parent class.")

    @property
    def estimator(self) -> Any:
        """The estimator backing the model."""
        raise NotImplementedError("estimator not implemented in parent class.")
