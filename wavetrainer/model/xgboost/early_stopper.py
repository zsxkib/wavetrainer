"""A callback function for early stopping."""

from typing import Any

from xgboost.callback import EarlyStopping, TrainingCallback


class XGBoostEarlyStoppingCallback(EarlyStopping):
    """A callback for early stopping in XGBoost models."""

    def after_iteration(
        self, model: Any, epoch: int, evals_log: TrainingCallback.EvalsLog
    ) -> bool:
        if len(evals_log.keys()) < 1:
            return False
        filtered_evals_log = {"validation": evals_log["validation_0"]}
        return super().after_iteration(model, epoch, filtered_evals_log)
