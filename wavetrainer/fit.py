"""A prototype class implementing a fit method."""

from typing import Self

import pandas as pd


class Fit:
    """The prototype fit class."""

    # pylint: disable=too-many-positional-arguments,too-many-arguments

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        """Fit the dataframe."""
        raise NotImplementedError("fit not implemented in parent class.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the dataframe."""
        raise NotImplementedError("transform not implemented in parent class.")

    def fit_transform(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Fit and then trasnfrom the dataframe."""
        return self.fit(df, y=y, w=w, eval_x=eval_x, eval_y=eval_y).transform(df)
