"""The single place that says where project data lives.

Every other file asks this module for a path, so moving the data means changing
one environment variable and no code.

    Local:  nothing to set; data lives in <repo>/data
    Colab:  os.environ["CYCLONE_DATA_DIR"] = "/content/drive/MyDrive/cyclone-project/data"
"""

import os
from pathlib import Path

ENV_VAR = "CYCLONE_DATA_DIR"

# <repo>/src/cyclone/config.py -> parents[2] is <repo>.
# A plain "./data" would be relative to wherever Python was started. Jupyter
# starts in notebooks/, so that would point at notebooks/data, which doesn't exist.
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def data_dir() -> Path:
    """Return the data directory: the environment variable if set, else <repo>/data.

    This is a function, not a constant, so the variable is read each time it's
    called. A Colab notebook can then set it in a cell after importing cyclone.
    """
    override = os.environ.get(ENV_VAR)
    return Path(override).expanduser() if override else _DEFAULT_DATA_DIR


def ibtracs_path(basin: str) -> Path:
    """Return the raw IBTrACS CSV path for a basin, e.g. ``"NI"`` or ``"WP"``."""
    return data_dir() / "raw" / f"ibtracs.{basin}.list.v04r01.csv"


def ni_clean_path() -> Path:
    """Return the path of the cleaned North Indian Ocean table."""
    return data_dir() / "processed" / "ni_clean.parquet"


def xgboost_dir() -> Path:
    """Return the folder holding the fitted XGBoost models."""
    return data_dir() / "models" / "xgboost"
