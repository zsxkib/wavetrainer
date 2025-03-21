"""A wrapper for catboost classifier to handle some edge cases."""

# pylint: disable=duplicate-code

from catboost import CatBoostClassifier  # type: ignore

from .catboost_kwargs import handle_fit_kwargs


class CatBoostClassifierWrapper(CatBoostClassifier):
    """A wrapper for the catboost classifier."""

    def fit(self, *args, **kwargs):
        args, kwargs = handle_fit_kwargs(*args, **kwargs)
        return super().fit(*args, **kwargs)
