"""A prototype class implementing a fit method."""

from typing import Self

import pandas as pd


class Fit:
    """The prototype fit class."""

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
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
    ) -> pd.DataFrame:
        """Fit and then trasnfrom the dataframe."""
        return self.fit(df, y=y).transform(df)
