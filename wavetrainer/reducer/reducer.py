"""The prototype reducer class."""

from ..fit import Fit
from ..params import Params


class Reducer(Params, Fit):
    """The prototype reducer class."""

    @classmethod
    def name(cls) -> str:
        """The name of the reducer."""
        raise NotImplementedError("name not implemented in parent class.")
