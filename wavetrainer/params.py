"""A class for loading/saving parameters."""

import optuna


class Params:
    """The params prototype class."""

    def set_options(self, trial: optuna.Trial | optuna.trial.FrozenTrial) -> None:
        """Set the options used in the object."""
        raise NotImplementedError("set_options not implemented in parent class.")

    def load(self, folder: str) -> None:
        """Loads the objects from a folder."""
        raise NotImplementedError("load not implemented in parent class.")

    def save(self, folder: str) -> None:
        """Saves the objects into a folder."""
        raise NotImplementedError("save not implemented in parent class.")
