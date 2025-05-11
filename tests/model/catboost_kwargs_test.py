"""Tests for the catboost kwargs handler class."""
import unittest

import pandas as pd

from wavetrainer.model.catboost.catboost_kwargs import handle_fit_kwargs


class TestCatboostKwargs(unittest.TestCase):

    def test_handle_fit_kwargs(self):
        x_train = pd.DataFrame(data={
            "thing": [0.0, 1.0, 2.0, 3.0, 4.0],
        })
        x_train["thing"] = x_train["thing"].astype('category')
        y_train = pd.Series(data=[1.0, 2.0, 3.0, 4.0])
        x_test = pd.DataFrame(data={
            "thing": [0.0, 1.0, 2.0, 3.0, 4.0],
        })
        x_test["thing"] = x_test["thing"].astype('category')
        y_test = pd.Series(data=[1.0, 2.0, 3.0, 4.0])
        args, _ = handle_fit_kwargs(
            x_train,
            y_train,
            eval_set=(x_test, y_test),
            cat_features=x_train.select_dtypes(include="category").columns.tolist(),
        )
        assert len(args) == 2
