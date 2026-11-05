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

from cyclone import final_model, lstm_model, xgb_model
from cyclone.baselines import CLIPER_NAME, PERSISTENCE_NAME, fit_cliper_on, predict_cliper
from cyclone.consensus import CONSENSUS_LSTM_NAME, CONSENSUS_NAME
from cyclone.features import build_features
from cyclone.folds import NOT_IN_A_FOLD, add_fold
from cyclone.samples import LEAD_HOURS, make_init_times
from cyclone.scoring import FORECAST_COLUMNS
from cyclone.sequences import build_sequences

SETTINGS = {
    "max_depth": 2,
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "subsample": 0.6,
    "n_estimators": 5,
}
# Different from SETTINGS, so that a model given the other's settings fails a test.
RESIDUAL_SETTINGS = {**SETTINGS, "max_depth": 3, "n_estimators": 8}
LSTM_SETTINGS = {
    "hidden_size": 4,
    "num_layers": 1,
    "dropout": 0.0,
    "learning_rate": 0.01,
    "weight_decay": 0.0,
}

ALL_MODELS = {
    PERSISTENCE_NAME,
    CLIPER_NAME,
    xgb_model.MODEL_NAME,
    xgb_model.RESIDUAL_NAME,
    lstm_model.MODEL_NAME,
    CONSENSUS_NAME,
    CONSENSUS_LSTM_NAME,
}


def one_model(forecasts: pd.DataFrame, name: str) -> pd.DataFrame:
    return forecasts[forecasts["model"] == name].reset_index(drop=True)


@pytest.fixture(scope="module")
def sequences(rows, table):
    return build_sequences(rows, table)


def fit(rows, truth, sequences) -> final_model.FinalModels:
    return final_model.fit_final_models(
        rows, truth, sequences, SETTINGS, RESIDUAL_SETTINGS, LSTM_SETTINGS, "cpu", "cpu"
    )


@pytest.fixture(scope="module")
def final(rows, truth, sequences) -> final_model.FinalModels:
    return fit(rows, truth, sequences)


@pytest.fixture(scope="module")
def later(table, rows) -> pd.DataFrame:
    """Return start times from the storms outside the blocks, in the same columns as rows."""
    init_times = make_init_times(table)
    extra = build_features(init_times).drop(columns=["split", "lat", "lon", "wind"])
    labelled = add_fold(init_times.merge(extra, on=["SID", "init_time"]), table)
    outside = labelled[labelled["fold"] == NOT_IN_A_FOLD]
    assert len(outside) > 0
    return outside[rows.columns]


def test_there_are_seven_models_at_every_lead_for_every_row(final, rows, sequences):
    forecasts = final_model.final_forecasts(final, rows, sequences)
    assert set(forecasts["model"]) == ALL_MODELS
    assert forecasts.columns.tolist() == FORECAST_COLUMNS
    assert len(forecasts) == len(ALL_MODELS) * len(rows) * len(LEAD_HOURS)


def test_the_consensus_is_the_average_of_cliper_and_the_five_seed_average(final, rows, sequences):
    forecasts = final_model.final_forecasts(final, rows, sequences)
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


def test_consensus_lstm_is_the_average_of_cliper_xgboost_and_lstm(final, rows, sequences):
    forecasts = final_model.final_forecasts(final, rows, sequences)
    members = [
        one_model(forecasts, name)
        for name in (CLIPER_NAME, xgb_model.MODEL_NAME, lstm_model.MODEL_NAME)
    ]
    averaged = one_model(forecasts, CONSENSUS_LSTM_NAME).sort_values(["SID", "init_time", "lead_h"])
    members = [member.sort_values(["SID", "init_time", "lead_h"]) for member in members]
    assert len(averaged) == len(rows) * len(LEAD_HOURS)
    for column in ("lat", "lon"):
        expected = sum(member[column].to_numpy() for member in members) / 3
        assert averaged[column].to_numpy() == pytest.approx(expected)


def test_every_seeded_model_holds_all_five_seeds(final):
    assert tuple(final.xgboost) == xgb_model.SEEDS
    assert tuple(final.xgb_residual) == xgb_model.SEEDS
    assert tuple(final.lstm) == xgb_model.SEEDS


def test_final_xgboost_is_the_model_that_was_measured(final, rows, truth, sequences):
    # The forecaster used under cross-validation, given every block to learn from. If
    # these matched only loosely, the cross-validated score wouldn't stand in for this one.
    measured = xgb_model.xgboost_forecaster(SETTINGS, "cpu", xgb_model.SEEDS)(rows, truth, rows)
    tm.assert_frame_equal(
        one_model(final_model.final_forecasts(final, rows, sequences), xgb_model.MODEL_NAME),
        measured,
    )


def test_final_residual_is_the_model_that_was_measured(final, rows, truth, sequences):
    measured = xgb_model.residual_forecaster(RESIDUAL_SETTINGS, "cpu", xgb_model.SEEDS)(
        rows, truth, rows
    )
    tm.assert_frame_equal(
        one_model(final_model.final_forecasts(final, rows, sequences), xgb_model.RESIDUAL_NAME),
        measured,
    )


def test_final_lstm_is_the_model_that_was_measured(final, rows, truth, sequences):
    measured = lstm_model.lstm_forecaster(
        LSTM_SETTINGS, "cpu", sequences, seeds=lstm_model.SEEDS
    )(rows, truth, rows)
    tm.assert_frame_equal(
        one_model(final_model.final_forecasts(final, rows, sequences), lstm_model.MODEL_NAME),
        measured,
    )


def test_every_model_learns_from_every_block_and_nothing_later(
    final, rows, truth, table, sequences, later
):
    # Later storms with wildly wrong positions: if they were used, the models would move.
    corrupted = truth.copy()
    corrupted.loc[corrupted["SID"].isin(later["SID"]), ["lat", "lon"]] += 40.0
    full = pd.concat([rows, later], ignore_index=True)

    reached = fit(full, corrupted, build_sequences(full, table))
    tm.assert_frame_equal(
        final_model.final_forecasts(final, rows, sequences),
        final_model.final_forecasts(reached, rows, sequences),
    )

    every_block = fit_cliper_on(rows, truth)
    tm.assert_frame_equal(predict_cliper(final.cliper, rows), predict_cliper(every_block, rows))


def test_a_change_in_any_block_changes_every_fitted_model(final, rows, truth, sequences):
    last_block = rows["fold"].max()
    storms = rows.loc[rows["fold"] == last_block, "SID"].unique()
    changed = truth.copy()
    changed.loc[changed["SID"].isin(storms), ["lat", "lon"]] += 3.0

    moved = fit(rows, changed, sequences)
    before = final_model.final_forecasts(final, rows, sequences)
    after = final_model.final_forecasts(moved, rows, sequences)
    for name in ALL_MODELS - {PERSISTENCE_NAME}:
        assert not one_model(before, name)["lat"].equals(one_model(after, name)["lat"])


def save(final, directory) -> None:
    search = pd.DataFrame([SETTINGS])
    final_model.save_final_models(final, search, search, pd.DataFrame([LSTM_SETTINGS]), directory)


def test_final_models_reload_with_identical_forecasts(final, rows, sequences, tmp_path):
    save(final, tmp_path)
    reloaded = final_model.load_final_models(tmp_path)
    tm.assert_frame_equal(
        final_model.final_forecasts(final, rows, sequences),
        final_model.final_forecasts(reloaded, rows, sequences),
    )


def test_a_missing_lstm_seed_stops_the_load(final, tmp_path):
    save(final, tmp_path)
    (tmp_path / final_model.LSTM_DIR / "seed46.pt").unlink()
    with pytest.raises(FileNotFoundError):
        final_model.load_final_models(tmp_path)


def test_a_missing_residual_seed_stops_the_load(final, tmp_path):
    save(final, tmp_path)
    path = tmp_path / final_model.RESIDUAL_DIR / "seed46" / "lead24_dlat.json"
    path.unlink()
    with pytest.raises(Exception, match="lead24_dlat|seed46"):
        final_model.load_final_models(tmp_path)


def test_the_lstms_checkpoints_are_reused_when_asked(rows, truth, sequences, tmp_path, monkeypatch):
    def fit_with_checkpoints(resume):
        return final_model.fit_final_models(
            rows,
            truth,
            sequences,
            SETTINGS,
            RESIDUAL_SETTINGS,
            LSTM_SETTINGS,
            "cpu",
            "cpu",
            lstm_checkpoints=tmp_path,
            resume=resume,
        )

    first = fit_with_checkpoints(resume=False)
    assert len(list(tmp_path.glob("seed*.pt"))) == len(lstm_model.SEEDS)

    def refuse(*args, **kwargs):
        raise AssertionError("The saved LSTMs should have been used.")

    monkeypatch.setattr(lstm_model, "fit_lstm", refuse)
    resumed = fit_with_checkpoints(resume=True)
    tm.assert_frame_equal(
        lstm_model.predict_seeds(first.lstm, rows, sequences),
        lstm_model.predict_seeds(resumed.lstm, rows, sequences),
    )


def test_the_models_can_forecast_storms_they_never_saw(final, later, table):
    forecasts = final_model.final_forecasts(final, later, build_sequences(later, table))
    assert len(forecasts) == len(ALL_MODELS) * len(later) * len(LEAD_HOURS)
    assert forecasts[["lat", "lon"]].notna().all().all()
