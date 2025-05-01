"""The prototype model class."""

# pylint: disable=too-many-arguments,too-many-positional-arguments
from typing import Any

import pandas as pd

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

    @classmethod
    def supports_x(cls, df: pd.DataFrame) -> bool:
        """Whether the model supports the X values."""
        raise NotImplementedError("supports_x not implemented in parent class.")

    @property
    def estimator(self) -> Any:
        """The estimator backing the model."""
        raise NotImplementedError("estimator not implemented in parent class.")

    @property
    def supports_importances(self) -> bool:
        """Whether this model supports feature importances."""
        raise NotImplementedError(
            "supports_importances not implemented in parent class."
        )

    @property
    def feature_importances(self) -> dict[str, float]:
        """The feature importances of this model."""
        raise NotImplementedError(
            "feature_importances not implemented in parent class."
        )

    def pre_fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
    ) -> dict[str, Any]:
        """A call to make sure the model is prepared for the target type."""
        raise NotImplementedError("pre_fit not implemented in parent class.")
