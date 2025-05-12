"""An XGBoost callback class for logging epochs."""

from typing import Any

from xgboost.callback import TrainingCallback


class XGBoostEpochsLogger(TrainingCallback):
    """Log the epochs in XGBoost."""

    def after_iteration(
        self, model: Any, epoch: int, evals_log: TrainingCallback.EvalsLog
    ) -> bool:
        if epoch % 100 != 0:
            return False
        log_items = []
        for dataset, metrics in evals_log.items():
            if dataset == "validation_0":
                dataset = "validation"
            elif dataset == "validation_1":
                dataset = "train"
            for metric_name, values in metrics.items():
                current_val = values[-1]
                log_items.append(f"{dataset}-{metric_name}: {current_val:.5f}")

        print(f"XGBoost: [{epoch}] " + " | ".join(log_items))
        return False
