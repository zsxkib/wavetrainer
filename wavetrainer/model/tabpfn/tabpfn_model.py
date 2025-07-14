"""A model that wraps tabpfn."""
# pylint: disable=duplicate-code,too-many-arguments,too-many-positional-arguments

import json
import logging
import os
import pickle
from typing import Self

import optuna
import pandas as pd
import pytest_is_running
import torch
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (  # type: ignore
    AutoTabPFNClassifier, AutoTabPFNRegressor)

from ...exceptions import WavetrainException
from ...model_type import ModelType, determine_model_type
from ..model import PREDICTION_COLUMN, PROBABILITY_COLUMN_PREFIX, Model

_MODEL_FILENAME = "model.pkl"
_MODEL_PARAMS_FILENAME = "model_params.json"
_MODEL_TYPE_KEY = "model_type"


class TabPFNModel(Model):
    """A class that uses TabPFN as a model."""

    _tabpfn: AutoTabPFNClassifier | AutoTabPFNRegressor | None
    _model_type: None | ModelType

    @classmethod
    def name(cls) -> str:
        return "tabpfn"

    @classmethod
    def supports_x(cls, df: pd.DataFrame) -> bool:
        return len(df.columns.values) < 500

    def __init__(self) -> None:
        super().__init__()
        self._tabpfn = None
        self._model_type = None

    @property
    def supports_importances(self) -> bool:
        return False

    def feature_importances(
        self, df: pd.DataFrame | None
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        return {}, []

    def provide_estimator(self):
        return self._provide_tabpfn()

    def create_estimator(self):
        return self._create_tabpfn()

    def reset(self):
        pass

    def convert_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def set_options(
        self, trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
    ) -> None:
        pass

    def load(self, folder: str) -> None:
        with open(os.path.join(folder, _MODEL_FILENAME), "rb") as f:
            self._tabpfn = pickle.load(f)
        with open(
            os.path.join(folder, _MODEL_PARAMS_FILENAME), encoding="utf8"
        ) as handle:
            params = json.load(handle)
            self._model_type = ModelType(params[_MODEL_TYPE_KEY])

    def save(self, folder: str, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        with open(os.path.join(folder, _MODEL_FILENAME), "wb") as f:
            pickle.dump(self._tabpfn, f)
        with open(
            os.path.join(folder, _MODEL_PARAMS_FILENAME), "w", encoding="utf8"
        ) as handle:
            json.dump(
                {
                    _MODEL_TYPE_KEY: str(self._model_type),
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
        if y is None:
            raise ValueError("y is null.")
        self._model_type = determine_model_type(y)
        tabpfn = self._provide_tabpfn()
        try:
            tabpfn.fit(df, y)
        except (ValueError, RuntimeError) as exc:
            logging.warning(str(exc))
            raise WavetrainException() from exc
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        tabpfn = self._provide_tabpfn()
        if tabpfn is None:
            raise ValueError("tabpfn is null")
        pred = tabpfn.predict(df)
        new_df = pd.DataFrame(
            index=df.index,
            data={
                PREDICTION_COLUMN: pred.flatten(),
            },
        )
        if isinstance(tabpfn, AutoTabPFNClassifier):
            proba = tabpfn.predict_proba(df)
            for i in range(proba.shape[1]):
                new_df[f"{PROBABILITY_COLUMN_PREFIX}{i}"] = proba[:, i]
        return new_df

    def _provide_tabpfn(self) -> AutoTabPFNClassifier | AutoTabPFNRegressor:
        tabpfn = self._tabpfn
        if tabpfn is None:
            tabpfn = self._create_tabpfn()
            self._tabpfn = tabpfn
        if tabpfn is None:
            raise ValueError("tabpfn is null")
        return tabpfn

    def _create_tabpfn(self) -> AutoTabPFNClassifier | AutoTabPFNRegressor:
        max_time = 1 if pytest_is_running.is_running() else 120
        match self._model_type:
            case ModelType.BINARY:
                return AutoTabPFNClassifier(
                    max_time=max_time,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
            case ModelType.REGRESSION:
                return AutoTabPFNRegressor(
                    max_time=max_time,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
            case ModelType.BINNED_BINARY:
                return AutoTabPFNClassifier(
                    max_time=max_time,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
            case ModelType.MULTI_CLASSIFICATION:
                return AutoTabPFNClassifier(
                    max_time=max_time,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
            case _:
                raise ValueError(f"Unrecognised model type: {self._model_type}")
