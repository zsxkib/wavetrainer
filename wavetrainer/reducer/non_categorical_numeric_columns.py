"""A helper function for retrieving numeric columns without categoricals."""

import numpy as np
import pandas as pd


def find_non_categorical_numeric_columns(df: pd.DataFrame) -> list[str]:
    """
    Finds numeric columns in a Pandas DataFrame that are not categorical.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        list: A list of column names that are numeric and not categorical.
    """
    numeric_cols = set(df.select_dtypes(include=np.number).columns.tolist())
    categorical_cols = set(df.select_dtypes(include="category").columns.tolist())
    return list(numeric_cols.difference(categorical_cols))
