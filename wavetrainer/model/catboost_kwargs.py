"""A list of constant catboost kwargs."""

from typing import Any

import numpy as np
from catboost import Pool  # type: ignore

ORIGINAL_X = "original_x"
EVAL_SET = "eval_set"


def handle_fit_kwargs(*args, **kwargs) -> dict[str, Any]:
    """Handles keyword args coming into a catboost fit method."""
    if ORIGINAL_X in kwargs:
        df = kwargs[ORIGINAL_X]
        eval_x, eval_y = kwargs[EVAL_SET]
        fit_x = args[0]
        fix_x_cp = fit_x.copy()

        # Stupid code to ensure eval is feature equivalent to train data
        included_columns = []
        for i in range(fix_x_cp.shape[1]):
            arr_col_values = fix_x_cp[:, i]
            for col in df.columns:
                df_col_values = df[col].values
                if np.allclose(df_col_values, arr_col_values, equal_nan=True):
                    included_columns.append(col)
                    df = df.drop(col, axis=1)
                    break

        eval_x = eval_x[included_columns]
        kwargs[EVAL_SET] = Pool(eval_x, label=eval_y)

        del kwargs[ORIGINAL_X]
    return kwargs
