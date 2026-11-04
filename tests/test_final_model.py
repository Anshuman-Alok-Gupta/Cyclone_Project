"""Tests for the final consensus.

The final models can't be checked by scoring them, since they've seen every development
storm and test stays locked. So these check what they're built from: every development
storm, nothing later, and the same forecasts after a reload.

The data is the set of made-up storms in conftest.py, whose later seasons stand in for
storms outside the blocks.
"""

import pandas as pd
import pandas.testing as tm
import pytest

from cyclone import final_model, xgb_model
from cyclone.baselines import fit_cliper_on, predict_cliper
from cyclone.consensus import CONSENSUS_NAME
from cyclone.features import build_features
from cyclone.folds import NOT_IN_A_FOLD, add_fold
from cyclone.samples import LEAD_HOURS, make_init_times
from cyclone.scoring import FORECAST_COLUMNS

SETTINGS = {
    "max_depth": 2,
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "subsample": 0.6,
    "n_estimators": 5,
}


@pytest.fixture(scope="module")
def final(rows, truth) -> final_model.FinalConsensus:
    return final_model.fit_final_consensus(rows, truth, SETTINGS, "cpu")


@pytest.fixture(scope="module")
def later(table, rows) -> pd.DataFrame:
    """Return start times from the storms outside the blocks, in the same columns as rows."""
    init_times = make_init_times(table)
    extra = build_features(init_times).drop(columns=["split", "lat", "lon", "wind"])
    labelled = add_fold(init_times.merge(extra, on=["SID", "init_time"]), table)
    outside = labelled[labelled["fold"] == NOT_IN_A_FOLD]
    assert len(outside) > 0
    return outside[rows.columns]


def test_the_consensus_is_the_average_of_cliper_and_the_five_seed_average(final, rows):
    forecasts = final_model.final_consensus_forecast(final, rows)
    assert (forecasts["model"] == CONSENSUS_NAME).all()
    assert forecasts.columns.tolist() == FORECAST_COLUMNS

    cliper = predict_cliper(final.cliper, rows)
    xgboost = xgb_model.predict_seeds(final.xgboost, rows)
    keys = ["SID", "init_time", "lead_h"]
    expected = cliper.merge(xgboost, on=keys, suffixes=("_cliper", "_xgboost"), validate="1:1")
    expected = expected.merge(forecasts, on=keys, validate="1:1")

    assert len(expected) == len(rows) * len(LEAD_HOURS)
    assert expected["lat"].to_numpy() == pytest.approx(
        ((expected["lat_cliper"] + expected["lat_xgboost"]) / 2).to_numpy()
    )
    assert expected["lon"].to_numpy() == pytest.approx(
        ((expected["lon_cliper"] + expected["lon_xgboost"]) / 2).to_numpy()
    )


def test_xgboost_member_holds_all_five_seeds(final):
    assert tuple(final.xgboost) == xgb_model.SEEDS


def test_both_members_learn_from_every_block_and_nothing_later(final, rows, truth, later):
    # Later storms with wildly wrong positions: if they were used, the models would move.
    corrupted = truth.copy()
    corrupted.loc[corrupted["SID"].isin(later["SID"]), ["lat", "lon"]] += 40.0
    full = pd.concat([rows, later], ignore_index=True)

    reached = final_model.fit_final_consensus(full, corrupted, SETTINGS, "cpu")
    tm.assert_frame_equal(
        final_model.final_consensus_forecast(final, rows),
        final_model.final_consensus_forecast(reached, rows),
    )

    every_block = fit_cliper_on(rows, truth)
    tm.assert_frame_equal(
        predict_cliper(final.cliper, rows), predict_cliper(every_block, rows)
    )


def test_a_change_in_any_block_changes_the_models(final, rows, truth):
    last_block = rows["fold"].max()
    storms = rows.loc[rows["fold"] == last_block, "SID"].unique()
    changed = truth.copy()
    changed.loc[changed["SID"].isin(storms), ["lat", "lon"]] += 3.0

    moved = final_model.fit_final_consensus(rows, changed, SETTINGS, "cpu")
    before = final_model.final_consensus_forecast(final, rows)
    after = final_model.final_consensus_forecast(moved, rows)
    assert not before["lat"].equals(after["lat"])


def test_final_consensus_reloads_with_identical_forecasts(final, rows, tmp_path):
    final_model.save_final_consensus(final, pd.DataFrame([SETTINGS]), tmp_path)
    reloaded = final_model.load_final_consensus(tmp_path)
    tm.assert_frame_equal(
        final_model.final_consensus_forecast(final, rows),
        final_model.final_consensus_forecast(reloaded, rows),
    )


def test_it_can_forecast_storms_it_never_saw(final, later):
    forecasts = final_model.final_consensus_forecast(final, later)
    assert len(forecasts) == len(later) * len(LEAD_HOURS)
    assert forecasts[["lat", "lon"]].notna().all().all()
