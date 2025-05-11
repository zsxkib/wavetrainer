"""A wrapper for catboost regressor to handle some edge cases."""

from catboost import CatBoostRegressor  # type: ignore

from .catboost_kwargs import handle_fit_kwargs


class CatBoostRegressorWrapper(CatBoostRegressor):
    """A wrapper for the catboost regressor."""

    def fit(self, *args, **kwargs):
        args, kwargs = handle_fit_kwargs(*args, **kwargs)
        return super().fit(*args, **kwargs)
