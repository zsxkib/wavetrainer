"""The function for loading the trainer state from disk."""

from .trainer import Trainer


def load(folder: str) -> Trainer:
    """Loads the trainer from the folder."""
    raise NotImplementedError("load isn't implemented.")
