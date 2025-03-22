"""A reducer that removes correlation features."""

from feature_engine.selection import DropCorrelatedFeatures

from .base_selector_reducer import BaseSelectorReducer

_CORRELATION_REDUCER_FILENAME = "correlation_reducer.joblib"


class CorrelationReducer(BaseSelectorReducer):
    """A class that removes correlated values from a dataset."""

    def __init__(self) -> None:
        super().__init__(
            DropCorrelatedFeatures(missing_values="ignore"),
            _CORRELATION_REDUCER_FILENAME,
        )

    @classmethod
    def name(cls) -> str:
        return "correlation"

    @classmethod
    def should_raise(cls) -> bool:
        return False
