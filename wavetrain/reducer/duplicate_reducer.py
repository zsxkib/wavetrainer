"""A reducer that removes duplicate features."""

from feature_engine.selection import DropDuplicateFeatures

from .base_selector_reducer import BaseSelectorReducer

_DUPLICATE_REDUCER_FILENAME = "duplicate_reducer.joblib"


class DuplicateReducer(BaseSelectorReducer):
    """A class that removes duplicate values from a dataset."""

    def __init__(self) -> None:
        super().__init__(
            DropDuplicateFeatures(missing_values="ignore"), _DUPLICATE_REDUCER_FILENAME
        )

    @classmethod
    def name(cls) -> str:
        return "duplicate"
