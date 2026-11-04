"""The final models: CLIPER, 5-seed XGBoost and 5-seed xgb_residual, on all of 1980-2017.

Cross-validation's models each saw four blocks, and their job was to measure a model.
Once it's measured, using all five blocks gives the final one more storms to learn from.

These models can't be scored on 1980-2017, since they've seen every one of those storms,
so the cross-validated score stands in for them. Nothing in this file scores anything.
The test split is scored once, at the end of Project 1, under the plan in CLAUDE.md.

It sits in its own file because the final models depend on each other: consensus averages
CLIPER and XGBoost, and xgb_residual corrects CLIPER. ``consensus.py`` reads only the
scoring format and knows nothing about any model, and ``xgb_model.py`` knows nothing about
how CLIPER is fitted here.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression

from cyclone import config, xgb_model
from cyclone.baselines import CLIPER_NAME, fit_cliper_on, persistence, predict_cliper
from cyclone.consensus import consensus
from cyclone.folds import development

CLIPER_FILE = "cliper.pkl"

# XGBoost's seed folders sit directly in the final folder, and xgb_residual's in here.
RESIDUAL_DIR = "residual"


@dataclass(frozen=True)
class FinalModels:
    """Every fitted final model. All were fitted on all of 1980-2017.

    Persistence has nothing to fit, and consensus is worked out from the forecasts.
    """

    cliper: dict[int, LinearRegression]
    xgboost: dict[int, dict[tuple[int, str], xgb.XGBRegressor]]
    xgb_residual: dict[int, dict[tuple[int, str], xgb.XGBRegressor]]


def fit_final_models(
    rows: pd.DataFrame,
    truth: pd.DataFrame,
    xgboost_settings: dict,
    residual_settings: dict,
    device: str,
) -> FinalModels:
    """Refit every model on every development season.

    One CLIPER is used throughout. xgb_residual learns to correct the forecasts of the
    same CLIPER that sits beside it at test time, as it did in each fold. Its CLIPER
    forecasts here are on storms CLIPER was fitted on, which is also the same as each fold.

    The filter is at the top, so passing the full table can't reach test.

    Args:
        rows: a table with a "fold" column, such as the one from ``cv.model_rows``.
        truth: the table from ``samples.make_truth``.
        xgboost_settings: the settings chosen for plain XGBoost.
        residual_settings: the settings chosen for xgb_residual.
        device: "cpu" or "cuda", from ``xgb_model.pick_device``.

    Returns:
        The models, to pass to ``final_forecasts``.
    """
    learning = development(rows)
    cliper = fit_cliper_on(learning, truth)
    return FinalModels(
        cliper=cliper,
        xgboost=xgb_model.fit_final(learning, truth, xgboost_settings, device),
        xgb_residual=xgb_model.fit_final(
            learning, truth, residual_settings, device, base=predict_cliper(cliper, learning)
        ),
    )


def final_forecasts(final: FinalModels, rows: pd.DataFrame) -> pd.DataFrame:
    """Return all five models' forecasts for the rows given.

    Args:
        final: the result of ``fit_final_models`` or ``load_final_models``.
        rows: the start times to forecast, with what every model reads.

    Returns:
        Forecasts in the format ``scoring.score`` takes, at each lead time.
    """
    cliper = predict_cliper(final.cliper, rows)
    members = pd.concat(
        [
            persistence(rows),
            cliper,
            xgb_model.predict_seeds(final.xgboost, rows),
            xgb_model.predict_seeds(final.xgb_residual, rows, xgb_model.RESIDUAL_NAME, base=cliper),
        ],
        ignore_index=True,
    )
    averaged = consensus(members, members=(CLIPER_NAME, xgb_model.MODEL_NAME))
    return pd.concat([members, averaged], ignore_index=True)


def save_final_models(
    final: FinalModels,
    xgboost_search: pd.DataFrame,
    residual_search: pd.DataFrame,
    directory: str | Path | None = None,
) -> None:
    """Write every model, so test day uses exactly these.

    Args:
        final: the result of ``fit_final_models``.
        xgboost_search: the table from ``xgb_model.search_cv`` for plain XGBoost, as a
            record of what was tried.
        residual_search: the same for xgb_residual.
        directory: where to write. Leave it out to use ``config.xgboost_final_dir``.
    """
    directory = Path(directory) if directory else config.xgboost_final_dir()
    xgb_model.save_final(final.xgboost, xgboost_search, directory)
    xgb_model.save_final(final.xgb_residual, residual_search, directory / RESIDUAL_DIR)
    with open(directory / CLIPER_FILE, "wb") as file:
        pickle.dump(final.cliper, file)


def load_final_models(directory: str | Path | None = None) -> FinalModels:
    """Read every model back. Pickle runs code as it loads, so read only your own file.

    Args:
        directory: where they were written. Leave it out to use ``config.xgboost_final_dir``.

    Returns:
        The models, to pass to ``final_forecasts``.
    """
    directory = Path(directory) if directory else config.xgboost_final_dir()
    with open(directory / CLIPER_FILE, "rb") as file:
        cliper = pickle.load(file)
    return FinalModels(
        cliper=cliper,
        xgboost=xgb_model.load_final(directory),
        xgb_residual=xgb_model.load_final(directory / RESIDUAL_DIR),
    )
