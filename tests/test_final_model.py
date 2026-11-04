"""Tests for the final models.

The final models can't be checked by scoring them, since they've seen every development
storm and test stays locked. So these check what they're built from: every development
storm, nothing later, and the same forecasts after a reload. They also check that the
final models are the models that were measured, fitted on more storms.

The data is the set of made-up storms in conftest.py, whose later seasons stand in for
storms outside the blocks.
"""

import pandas as pd
import pandas.testing as tm
import pytest

from cyclone import final_model, xgb_model
from cyclone.baselines import CLIPER_NAME, PERSISTENCE_NAME, fit_cliper_on, predict_cliper
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
# Different from SETTINGS, so that a model given the other's settings fails a test.
RESIDUAL_SETTINGS = {**SETTINGS, "max_depth": 3, "n_estimators": 8}

ALL_MODELS = {
    PERSISTENCE_NAME,
    CLIPER_NAME,
    xgb_model.MODEL_NAME,
    xgb_model.RESIDUAL_NAME,
    CONSENSUS_NAME,
}


def one_model(forecasts: pd.DataFrame, name: str) -> pd.DataFrame:
    return forecasts[forecasts["model"] == name].reset_index(drop=True)


@pytest.fixture(scope="module")
def final(rows, truth) -> final_model.FinalModels:
    return final_model.fit_final_models(rows, truth, SETTINGS, RESIDUAL_SETTINGS, "cpu")


@pytest.fixture(scope="module")
def later(table, rows) -> pd.DataFrame:
    """Return start times from the storms outside the blocks, in the same columns as rows."""
    init_times = make_init_times(table)
    extra = build_features(init_times).drop(columns=["split", "lat", "lon", "wind"])
    labelled = add_fold(init_times.merge(extra, on=["SID", "init_time"]), table)
    outside = labelled[labelled["fold"] == NOT_IN_A_FOLD]
    assert len(outside) > 0
    return outside[rows.columns]


def test_there_are_five_models_at_every_lead_for_every_row(final, rows):
    forecasts = final_model.final_forecasts(final, rows)
    assert set(forecasts["model"]) == ALL_MODELS
    assert forecasts.columns.tolist() == FORECAST_COLUMNS
    assert len(forecasts) == len(ALL_MODELS) * len(rows) * len(LEAD_HOURS)


def test_the_consensus_is_the_average_of_cliper_and_the_five_seed_average(final, rows):
    forecasts = final_model.final_forecasts(final, rows)
    keys = ["SID", "init_time", "lead_h"]
    expected = (
        one_model(forecasts, CLIPER_NAME)
        .merge(
            one_model(forecasts, xgb_model.MODEL_NAME),
            on=keys,
            suffixes=("_cliper", "_xgboost"),
            validate="1:1",
        )
        .merge(one_model(forecasts, CONSENSUS_NAME), on=keys, validate="1:1")
    )
    assert len(expected) == len(rows) * len(LEAD_HOURS)
    assert expected["lat"].to_numpy() == pytest.approx(
        ((expected["lat_cliper"] + expected["lat_xgboost"]) / 2).to_numpy()
    )
    assert expected["lon"].to_numpy() == pytest.approx(
        ((expected["lon_cliper"] + expected["lon_xgboost"]) / 2).to_numpy()
    )


def test_every_xgboost_model_holds_all_five_seeds(final):
    assert tuple(final.xgboost) == xgb_model.SEEDS
    assert tuple(final.xgb_residual) == xgb_model.SEEDS


def test_final_xgboost_is_the_model_that_was_measured(final, rows, truth):
    # The forecaster used under cross-validation, given every block to learn from. If
    # these matched only loosely, the cross-validated score wouldn't stand in for this one.
    measured = xgb_model.xgboost_forecaster(SETTINGS, "cpu", xgb_model.SEEDS)(rows, truth, rows)
    tm.assert_frame_equal(
        one_model(final_model.final_forecasts(final, rows), xgb_model.MODEL_NAME), measured
    )


def test_final_residual_is_the_model_that_was_measured(final, rows, truth):
    measured = xgb_model.residual_forecaster(RESIDUAL_SETTINGS, "cpu", xgb_model.SEEDS)(
        rows, truth, rows
    )
    tm.assert_frame_equal(
        one_model(final_model.final_forecasts(final, rows), xgb_model.RESIDUAL_NAME), measured
    )


def test_every_model_learns_from_every_block_and_nothing_later(final, rows, truth, later):
    # Later storms with wildly wrong positions: if they were used, the models would move.
    corrupted = truth.copy()
    corrupted.loc[corrupted["SID"].isin(later["SID"]), ["lat", "lon"]] += 40.0
    full = pd.concat([rows, later], ignore_index=True)

    reached = final_model.fit_final_models(full, corrupted, SETTINGS, RESIDUAL_SETTINGS, "cpu")
    tm.assert_frame_equal(
        final_model.final_forecasts(final, rows),
        final_model.final_forecasts(reached, rows),
    )

    every_block = fit_cliper_on(rows, truth)
    tm.assert_frame_equal(predict_cliper(final.cliper, rows), predict_cliper(every_block, rows))


def test_a_change_in_any_block_changes_every_fitted_model(final, rows, truth):
    last_block = rows["fold"].max()
    storms = rows.loc[rows["fold"] == last_block, "SID"].unique()
    changed = truth.copy()
    changed.loc[changed["SID"].isin(storms), ["lat", "lon"]] += 3.0

    moved = final_model.fit_final_models(rows, changed, SETTINGS, RESIDUAL_SETTINGS, "cpu")
    before = final_model.final_forecasts(final, rows)
    after = final_model.final_forecasts(moved, rows)
    for name in (CLIPER_NAME, xgb_model.MODEL_NAME, xgb_model.RESIDUAL_NAME, CONSENSUS_NAME):
        assert not one_model(before, name)["lat"].equals(one_model(after, name)["lat"])


def test_final_models_reload_with_identical_forecasts(final, rows, tmp_path):
    table = pd.DataFrame([SETTINGS])
    final_model.save_final_models(final, table, table, tmp_path)
    reloaded = final_model.load_final_models(tmp_path)
    tm.assert_frame_equal(
        final_model.final_forecasts(final, rows),
        final_model.final_forecasts(reloaded, rows),
    )


def test_a_missing_residual_seed_stops_the_load(final, tmp_path):
    table = pd.DataFrame([SETTINGS])
    final_model.save_final_models(final, table, table, tmp_path)
    path = tmp_path / final_model.RESIDUAL_DIR / "seed46" / "lead24_dlat.json"
    path.unlink()
    with pytest.raises(Exception, match="lead24_dlat|seed46"):
        final_model.load_final_models(tmp_path)


def test_the_models_can_forecast_storms_they_never_saw(final, later):
    forecasts = final_model.final_forecasts(final, later)
    assert len(forecasts) == len(ALL_MODELS) * len(later) * len(LEAD_HOURS)
    assert forecasts[["lat", "lon"]].notna().all().all()
