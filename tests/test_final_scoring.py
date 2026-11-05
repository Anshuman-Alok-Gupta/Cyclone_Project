"""Tests for the code that scores the final models.

This code runs once on the test split, so it has to be proved beforehand. Everything here
runs on the made-up storms in conftest.py. The scoring function is only ever called on
development seasons, or refused. Its guard is tested alone, so that confirming a test
season never has to score one.
"""

import pandas as pd
import pandas.testing as tm
import pytest

from cyclone import cv, final_model, final_scoring, lstm_model, xgb_model
from cyclone.baselines import CLIPER_NAME, PERSISTENCE_NAME
from cyclone.consensus import CONSENSUS_LSTM_NAME, CONSENSUS_NAME
from cyclone.folds import FOLD_NAMES, FOLD_SEASONS, NOT_IN_A_FOLD
from cyclone.samples import LEAD_HOURS, make_truth
from cyclone.scoring import score, summarize
from cyclone.sequences import build_sequences

SETTINGS = {
    "max_depth": 2,
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "subsample": 0.6,
    "n_estimators": 5,
}
LSTM_SETTINGS = {
    "hidden_size": 4,
    "num_layers": 1,
    "dropout": 0.0,
    "learning_rate": 0.01,
    "weight_decay": 0.0,
}

DEVELOPMENT_SEASONS = range(FOLD_SEASONS[0][0], FOLD_SEASONS[-1][1] + 1)
LAST_BLOCK = range(FOLD_SEASONS[-1][0], FOLD_SEASONS[-1][1] + 1)
ALL_MODELS = {
    PERSISTENCE_NAME,
    CLIPER_NAME,
    xgb_model.MODEL_NAME,
    xgb_model.RESIDUAL_NAME,
    lstm_model.MODEL_NAME,
    CONSENSUS_NAME,
    CONSENSUS_LSTM_NAME,
}


@pytest.fixture(scope="module")
def saved(rows, truth, table, tmp_path_factory):
    """Fit the final models and save them, as notebook 05 does."""
    directory = tmp_path_factory.mktemp("final")
    final = final_model.fit_final_models(
        rows, truth, build_sequences(rows, table), SETTINGS, SETTINGS, LSTM_SETTINGS, "cpu", "cpu"
    )
    search = pd.DataFrame([SETTINGS])
    final_model.save_final_models(final, search, search, pd.DataFrame([LSTM_SETTINGS]), directory)
    return directory


def test_the_test_seasons_are_2018_to_2024():
    assert final_scoring.TEST_SEASONS == frozenset(range(2018, 2025))


def test_rows_match_the_cross_validation_rows_on_the_development_seasons(table, rows):
    built = final_scoring.rows_for_seasons(table, DEVELOPMENT_SEASONS)
    tm.assert_frame_equal(built, rows)


def test_rows_for_one_block_are_that_blocks_cross_validation_rows(table, rows):
    built = final_scoring.rows_for_seasons(table, LAST_BLOCK)
    expected = rows[rows["fold"] == FOLD_NAMES[-1]].reset_index(drop=True)
    tm.assert_frame_equal(built, expected)


def test_rows_leave_out_the_seasons_not_asked_for(table):
    built = final_scoring.rows_for_seasons(table, LAST_BLOCK)
    asked = table[table["SEASON"].isin(LAST_BLOCK)]["SID"]
    assert set(built["SID"]) == set(asked)


def test_rows_outside_the_blocks_are_labelled_as_outside(table, rows):
    # Made-up storms after 2017, so that a real test season isn't the first to reach this.
    later = sorted(set(table["SEASON"]) - set(DEVELOPMENT_SEASONS))
    built = final_scoring.rows_for_seasons(table, later)
    assert len(built) > 0
    assert (built["fold"] == NOT_IN_A_FOLD).all()
    assert built.columns.tolist() == rows.columns.tolist()


@pytest.mark.parametrize("seasons", [[], [1979], [2017, 2025]])
def test_rows_refuse_seasons_that_are_not_in_the_table(table, seasons):
    with pytest.raises(ValueError, match="not in the cleaned table"):
        final_scoring.rows_for_seasons(table, seasons)


def test_scoring_gives_what_scoring_the_saved_models_directly_gives(table, rows, truth, saved):
    scored = final_scoring.score_final_models(table, DEVELOPMENT_SEASONS, saved)

    final = final_model.load_final_models(saved)
    forecasts = final_model.final_forecasts(final, rows, build_sequences(rows, table))
    expected = score(forecasts, make_truth(table))
    tm.assert_frame_equal(scored, expected)
    assert set(scored["model"]) == ALL_MODELS


def test_scoring_covers_every_model_at_every_lead_for_one_block(table, saved):
    scored = final_scoring.score_final_models(table, LAST_BLOCK, saved)
    assert set(scored["model"]) == ALL_MODELS
    assert set(scored["lead_h"]) == set(LEAD_HOURS)
    last_block_storms = set(table[table["SEASON"].isin(LAST_BLOCK)]["SID"])
    assert set(scored["SID"]) <= last_block_storms


def test_the_scores_go_straight_into_the_reporting_functions(table, saved):
    scored = final_scoring.score_final_models(table, LAST_BLOCK, saved)
    assert set(summarize(scored)["model"]) == ALL_MODELS
    assert set(cv.bootstrap_skill(scored, n_resamples=20)["model"]) == ALL_MODELS - {CLIPER_NAME}


def test_scoring_uses_the_saved_models_and_fits_nothing(table, saved, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("Scoring must not fit anything.")

    monkeypatch.setattr(xgb_model, "fit_fixed", refuse)
    monkeypatch.setattr(final_model, "fit_cliper_on", refuse)
    monkeypatch.setattr(lstm_model, "fit_lstm", refuse)
    monkeypatch.setattr(lstm_model, "fit_scaler", refuse)
    final_scoring.score_final_models(table, LAST_BLOCK, saved)


def test_the_lstm_reads_only_the_seasons_being_scored_up_to_each_start_time(table, saved):
    # Storms from other seasons have no way to reach the LSTM or anything else.
    corrupted = table.copy()
    other_seasons = ~corrupted["SEASON"].isin(LAST_BLOCK)
    corrupted.loc[other_seasons, ["USA_LAT", "USA_LON"]] += 30.0
    tm.assert_frame_equal(
        final_scoring.score_final_models(table, LAST_BLOCK, saved),
        final_scoring.score_final_models(corrupted, LAST_BLOCK, saved),
    )


def test_scoring_stops_if_the_models_are_missing(table, tmp_path):
    with pytest.raises(FileNotFoundError):
        final_scoring.score_final_models(table, LAST_BLOCK, tmp_path)


def test_the_last_development_season_needs_no_confirmation(table, saved):
    scored = final_scoring.score_final_models(table, [2017], saved)
    assert set(scored["model"]) == ALL_MODELS


def test_confirming_changes_nothing_about_the_scores(table, saved):
    tm.assert_frame_equal(
        final_scoring.score_final_models(table, LAST_BLOCK, saved),
        final_scoring.score_final_models(table, LAST_BLOCK, saved, confirm_test=True),
    )


@pytest.mark.parametrize("season", sorted(final_scoring.TEST_SEASONS))
def test_every_test_season_is_refused_without_confirmation(table, saved, season):
    with pytest.raises(ValueError, match="test split"):
        final_scoring.score_final_models(table, [season], saved)


def test_one_test_season_among_development_seasons_is_refused(table, saved):
    with pytest.raises(ValueError, match=r"\[2018\]"):
        final_scoring.score_final_models(table, [*LAST_BLOCK, 2018], saved)


def test_a_season_range_reaching_into_test_is_refused(table, saved):
    with pytest.raises(ValueError, match="test split"):
        final_scoring.score_final_models(table, range(2017, 2025), saved)


@pytest.mark.parametrize("not_a_confirmation", [False, None, 1, "yes", "True", [True]])
def test_only_true_itself_confirms(table, saved, not_a_confirmation):
    with pytest.raises(ValueError, match="test split"):
        final_scoring.score_final_models(table, [2018], saved, confirm_test=not_a_confirmation)


def test_confirmation_must_be_named_in_the_call(table, saved):
    with pytest.raises(TypeError):
        final_scoring.score_final_models(table, [2018], saved, True)


def test_the_guard_runs_before_anything_is_read(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("Nothing may be read before the guard has run.")

    monkeypatch.setattr(final_scoring, "rows_for_seasons", refuse)
    monkeypatch.setattr(final_model, "load_final_models", refuse)
    # No table at all: a refusal can't depend on looking at the data.
    with pytest.raises(ValueError, match="test split"):
        final_scoring.score_final_models(None, [2018])


def test_the_guard_lets_a_confirmed_test_season_through():
    # The guard alone, so that this path is proved without scoring a test season.
    assert final_scoring.check_seasons([2018], confirm_test=True) == {2018}


def test_the_guard_lets_development_seasons_through_unconfirmed():
    assert final_scoring.check_seasons(LAST_BLOCK) == frozenset(LAST_BLOCK)


def test_the_guard_refuses_an_empty_list():
    with pytest.raises(ValueError, match="No seasons"):
        final_scoring.check_seasons([])
