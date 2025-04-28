"""A weights class that routes to other weights."""

import json
import os
from typing import Self

import optuna
import pandas as pd

from .exponential_weights import ExponentialWeights
from .linear_weights import LinearWeights
from .noop_weights import NoopWeights
from .sigmoid_weights import SigmoidWeights
from .weights import Weights

_WEIGHTS_ROUTER_FILE = "weights_router.json"
_WEIGHTS_KEY = "weights"
_WEIGHTS = {
    ExponentialWeights.name(): ExponentialWeights,
    LinearWeights.name(): LinearWeights,
    NoopWeights.name(): NoopWeights,
    SigmoidWeights.name(): SigmoidWeights,
}


class WeightsRouter(Weights):
    """A router that routes to a different weights class."""

    # pylint: disable=too-many-positional-arguments,too-many-arguments

    _weights: Weights | None

    def __init__(self) -> None:
        super().__init__()
        self._weights = None

    @classmethod
    def name(cls) -> str:
        return "router"

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        self._weights = _WEIGHTS[
            trial.suggest_categorical("weights", list(_WEIGHTS.keys()))
        ]()

    def load(self, folder: str) -> None:
        weights = self._weights
        if weights is None:
            raise ValueError("weights is null")
        with open(
            os.path.join(folder, _WEIGHTS_ROUTER_FILE), encoding="utf8"
        ) as handle:
            params = json.load(handle)
            weights = _WEIGHTS[params[_WEIGHTS_KEY]]()
        self._weights = weights

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        weights = self._weights
        if weights is None:
            raise ValueError("weights is null")
        weights.save(folder, trial)
        with open(
            os.path.join(folder, _WEIGHTS_ROUTER_FILE), "w", encoding="utf8"
        ) as handle:
            json.dump(
                {
                    _WEIGHTS_KEY: weights.name(),
                },
                handle,
            )

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
    ) -> Self:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        weights = self._weights
        if weights is None:
            raise ValueError("weights is null")
        return weights.transform(df)
