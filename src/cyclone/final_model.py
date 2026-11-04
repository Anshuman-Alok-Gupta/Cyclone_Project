"""The final consensus: CLIPER and 5-seed XGBoost, refitted on all of 1980-2017.

Cross-validation's models each saw four blocks, and their job was to measure a model.
Once it's measured, using all five blocks gives the final one more storms to learn from.

These models can't be scored on 1980-2017, since they've seen every one of those storms,
so the cross-validated score stands in for them. Nothing in this file scores anything.
The test split is scored once, at the end of Project 1, under the plan in CLAUDE.md.

It sits in its own file because ``consensus.py`` reads only the scoring format and knows
nothing about any particular model, and ``xgb_model.py`` knows nothing about CLIPER's fit.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression

from cyclone import config, xgb_model
from cyclone.baselines import CLIPER_NAME, fit_cliper_on, predict_cliper
from cyclone.consensus import consensus
from cyclone.folds import development

CLIPER_FILE = "cliper.pkl"


@dataclass(frozen=True)
class FinalConsensus:
    """The two members of the final consensus, both fitted on all of 1980-2017."""

    cliper: dict[int, LinearRegression]
    xgboost: dict[int, dict[tuple[int, str], xgb.XGBRegressor]]


def fit_final_consensus(
    rows: pd.DataFrame,
    truth: pd.DataFrame,
    settings: dict,
    device: str,
) -> FinalConsensus:
    """Refit both members of the consensus on every development season.

    The filters are inside here and in ``xgb_model.fit_final``, so passing the full table
    can't reach test.

    Args:
        rows: a table with a "fold" column, such as the one from ``cv.model_rows``.
        truth: the table from ``samples.make_truth``.
        settings: the XGBoost settings chosen by cross-validation.
        device: "cpu" or "cuda", from ``xgb_model.pick_device``.

    Returns:
        Both members, to pass to ``final_consensus_forecast``.
    """
    return FinalConsensus(
        cliper=fit_cliper_on(development(rows), truth),
        xgboost=xgb_model.fit_final(rows, truth, settings, device),
    )


def final_consensus_forecast(final: FinalConsensus, rows: pd.DataFrame) -> pd.DataFrame:
    """Return the average of CLIPER's position and the 5-seed XGBoost position.

    Args:
        final: the result of ``fit_final_consensus`` or ``load_final_consensus``.
        rows: the start times to forecast, with what both members read.

    Returns:
        Forecasts in the format ``scoring.score`` takes, at each lead time.
    """
    forecasts = pd.concat(
        [predict_cliper(final.cliper, rows), xgb_model.predict_seeds(final.xgboost, rows)],
        ignore_index=True,
    )
    return consensus(forecasts, members=(CLIPER_NAME, xgb_model.MODEL_NAME))


def save_final_consensus(
    final: FinalConsensus,
    search_table: pd.DataFrame,
    directory: str | Path | None = None,
) -> None:
    """Write both members, so test day uses exactly these models.

    Args:
        final: the result of ``fit_final_consensus``.
        search_table: the table from ``xgb_model.search_cv``, as a record of what was tried.
        directory: where to write. Leave it out to use ``config.xgboost_final_dir``.
    """
    directory = Path(directory) if directory else config.xgboost_final_dir()
    xgb_model.save_final(final.xgboost, search_table, directory)
    with open(directory / CLIPER_FILE, "wb") as file:
        pickle.dump(final.cliper, file)


def load_final_consensus(directory: str | Path | None = None) -> FinalConsensus:
    """Read both members back. Pickle runs code as it loads, so read only your own file.

    Args:
        directory: where they were written. Leave it out to use ``config.xgboost_final_dir``.

    Returns:
        Both members, to pass to ``final_consensus_forecast``.
    """
    directory = Path(directory) if directory else config.xgboost_final_dir()
    with open(directory / CLIPER_FILE, "rb") as file:
        cliper = pickle.load(file)
    return FinalConsensus(cliper=cliper, xgboost=xgb_model.load_final(directory))
