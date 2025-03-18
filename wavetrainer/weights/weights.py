"""The prototype weights class."""

from ..fit import Fit
from ..params import Params

WEIGHTS_COLUMN = "weights"


class Weights(Params, Fit):
    """The prototype weights class."""

    @classmethod
    def name(cls) -> str:
        """The name of the weighter."""
        raise NotImplementedError("name not implemented in parent class.")
