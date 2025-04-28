"""A function for creating a new trainer."""

# pylint: disable=too-many-arguments,too-many-positional-arguments

import datetime

from .trainer import Trainer


def create(
    folder: str,
    walkforward_timedelta: datetime.timedelta = datetime.timedelta(days=1.0),
    test_size: float | datetime.timedelta | None = None,
    validation_size: float | datetime.timedelta | None = None,
    dt_column: str | None = None,
    max_train_timeout: datetime.timedelta | None = None,
    cutoff_dt: datetime.datetime | None = None,
) -> Trainer:
    """Create a trainer."""
    return Trainer(
        folder,
        walkforward_timedelta,
        test_size=test_size,
        validation_size=validation_size,
        dt_column=dt_column,
        max_train_timeout=max_train_timeout,
        cutoff_dt=cutoff_dt,
    )
