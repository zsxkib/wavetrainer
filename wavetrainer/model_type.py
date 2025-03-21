"""An enum to define the model type."""

from enum import StrEnum, auto

import pandas as pd


class ModelType(StrEnum):
    """The type of model being run."""

    BINARY = auto()
    REGRESSION = auto()
    BINNED_BINARY = auto()
    MULTI_CLASSIFICATION = auto()


def determine_model_type(y: pd.Series | pd.DataFrame) -> ModelType:
    """Determine the model type from the Y value."""
    if isinstance(y, pd.DataFrame) and len(y.columns.values) > 1:
        return ModelType.BINNED_BINARY

    series = y if isinstance(y, pd.Series) else y[y.columns[0]]

    if series.dtype == float:
        return ModelType.REGRESSION
    if series.dtype == int:
        return ModelType.MULTI_CLASSIFICATION
    return ModelType.BINARY
