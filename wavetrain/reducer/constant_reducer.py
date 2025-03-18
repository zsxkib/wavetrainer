"""A reducer that removes constant features."""

from feature_engine.selection import DropConstantFeatures

from .base_selector_reducer import BaseSelectorReducer

_CONSTANT_REDUCER_FILENAME = "constant_reducer.joblib"


class ConstantReducer(BaseSelectorReducer):
    """A class that removes constant values from a dataset."""

    def __init__(self) -> None:
        super().__init__(
            DropConstantFeatures(missing_values="ignore"), _CONSTANT_REDUCER_FILENAME
        )

    @classmethod
    def name(cls) -> str:
        return "constant"
