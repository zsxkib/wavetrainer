"""A model class that routes to other models."""

import json
import os
from typing import Any, Self

import optuna
import pandas as pd

from .catboost_model import CatboostModel
from .model import Model
from .tabpfn_model import TabPFNModel

_MODEL_ROUTER_FILE = "model_router.json"
_MODEL_KEY = "model"
_MODELS = {
    CatboostModel.name(): CatboostModel,
    TabPFNModel.name(): TabPFNModel,
}


class ModelRouter(Model):
    """A router that routes to a different weights class."""

    # pylint: disable=too-many-positional-arguments,too-many-arguments

    _model: Model | None

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    @classmethod
    def name(cls) -> str:
        return "router"

    @classmethod
    def supports_x(cls, df: pd.DataFrame) -> bool:
        return True

    @property
    def estimator(self) -> Any:
        model = self._model
        if model is None:
            raise ValueError("model is null")
        return model.estimator

    @property
    def supports_importances(self) -> bool:
        model = self._model
        if model is None:
            raise ValueError("model is null")
        return model.supports_importances

    @property
    def feature_importances(self) -> dict[str, float]:
        model = self._model
        if model is None:
            raise ValueError("model is null")
        return model.feature_importances

    def pre_fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None,
        eval_x: pd.DataFrame | None = None,
        eval_y: pd.Series | pd.DataFrame | None = None,
        w: pd.Series | None = None,
    ) -> dict[str, Any]:
        model = self._model
        if model is None:
            raise ValueError("model is null")
        return model.pre_fit(df, y=y, eval_x=eval_x, eval_y=eval_y, w=w)

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        model = _MODELS[
            trial.suggest_categorical(
                "model", [k for k, v in _MODELS.items() if v.supports_x(df)]
            )
        ]()
        model.set_options(trial, df)
        self._model = model

    def load(self, folder: str) -> None:
        with open(os.path.join(folder, _MODEL_ROUTER_FILE), encoding="utf8") as handle:
            params = json.load(handle)
            model = _MODELS[params[_MODEL_KEY]]()
        model.load(folder)
        self._model = model

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        model = self._model
        if model is None:
            raise ValueError("model is null")
        model.save(folder, trial)
        with open(
            os.path.join(folder, _MODEL_ROUTER_FILE), "w", encoding="utf8"
        ) as handle:
            json.dump(
                {
                    _MODEL_KEY: model.name(),
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
        model = self._model
        if model is None:
            raise ValueError("model is null")
        model.fit(df, y=y, w=w, eval_x=eval_x, eval_y=eval_y)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        model = self._model
        if model is None:
            raise ValueError("model is null")
        return model.transform(df)
