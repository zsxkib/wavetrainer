"""An enum to define the model type."""

from enum import Enum

import pandas as pd


class ModelType(Enum):
    """The type of model being run."""

    BINARY = 1
    REGRESSION = 2
    BINNED_BINARY = 3
    MULTI_CLASSIFICATION = 4


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
