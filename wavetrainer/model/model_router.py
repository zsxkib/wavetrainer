"""A model class that routes to other models."""

import json
import os
from typing import Self

import optuna
import pandas as pd

from .catboost.catboost_model import CatboostModel
from .model import Model
from .tabpfn.tabpfn_model import TabPFNModel
from .xgboost.xgboost_model import XGBoostModel

_MODEL_ROUTER_FILE = "model_router.json"
_MODEL_KEY = "model"
_MODELS = {
    CatboostModel.name(): CatboostModel,
    TabPFNModel.name(): TabPFNModel,
    XGBoostModel.name(): XGBoostModel,
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

    def provide_estimator(self):
        model = self._model
        if model is None:
            raise ValueError("model is null")
        return model.provide_estimator()

    def create_estimator(self):
        model = self._model
        if model is None:
            raise ValueError("model is null")
        return model.create_estimator()

    def reset(self):
        model = self._model
        if model is None:
            raise ValueError("model is null")
        model.reset()

    def convert_df(self, df: pd.DataFrame) -> pd.DataFrame:
        model = self._model
        if model is None:
            raise ValueError("model is null")
        return model.convert_df(df)

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        model_name = trial.suggest_categorical(
            "model", [k for k, v in _MODELS.items() if v.supports_x(df)]
        )
        print(f"Using {model_name} model")
        model = _MODELS[model_name]()
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
