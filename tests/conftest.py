"""Made-up storms shared by the cross-validation and XGBoost tests.

One storm per season, so every block has storms and there are later seasons to keep out.
They differ at random, so a change in the data shows up in a fitted model. pytest finds
this file by itself.
"""

import numpy as np
import pandas as pd
import pytest

from cyclone import cv, lstm_model
from cyclone.features import build_features
from cyclone.folds import FOLD_SEASONS
from cyclone.samples import make_init_times, make_truth
from cyclone.splits import VAL_LAST_SEASON

SIX_HOURS = pd.Timedelta(hours=6)


def make_storm(season: int, rng: np.random.Generator) -> pd.DataFrame:
    """Return one made-up storm in the cleaned table's format."""
    steps = np.arange(24)
    start = pd.Timestamp(f"{season}-05-01")
    return pd.DataFrame(
        {
            "SID": f"STORM{season}",
            "SEASON": season,
            "ISO_TIME": start + steps * SIX_HOURS,
            "split": "train",
            "USA_LAT": 8.0 + rng.uniform(0.1, 0.5) * steps + rng.normal(0, 0.1, len(steps)),
            "USA_LON": 95.0 - rng.uniform(0.1, 0.6) * steps + rng.normal(0, 0.1, len(steps)),
            "USA_WIND": 30.0 + rng.uniform(0.5, 3.0) * steps,
        }
    )


@pytest.fixture(scope="session", autouse=True)
def quick_lstm_training():
    """Keep each LSTM fit to a few epochs, since these tests check rules, not forecasts."""
    patch = pytest.MonkeyPatch()
    patch.setattr(lstm_model, "MAX_EPOCHS", 3)
    yield
    patch.undo()


@pytest.fixture(scope="session")
def table() -> pd.DataFrame:
    """Return one storm per development season, plus storms from two later seasons."""
    rng = np.random.default_rng(0)
    seasons = range(FOLD_SEASONS[0][0], VAL_LAST_SEASON + 3)
    return pd.concat([make_storm(season, rng) for season in seasons], ignore_index=True)


@pytest.fixture(scope="session")
def rows(table) -> pd.DataFrame:
    init_times = make_init_times(table)
    return cv.model_rows(init_times, build_features(init_times), table)


@pytest.fixture(scope="session")
def truth(table) -> pd.DataFrame:
    return make_truth(table)
