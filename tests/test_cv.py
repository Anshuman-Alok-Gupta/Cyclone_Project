"""Tests for the cross-validation runner and its uncertainty measures.

The leakage tests matter most. Cross-validation is only worth doing if each block is
forecast by a model that never saw it. A leak here wouldn't crash: it would quietly
improve the scores of every model, which is the very thing the method exists to stop.

The data is the set of made-up storms in conftest.py, one per season, so these run
anywhere and need no download.
"""

import pandas as pd
import pandas.testing as tm
import pytest

from cyclone import cv, xgb_model
from cyclone.baselines import cliper_forecast, persistence_forecast
from cyclone.features import build_features
from cyclone.folds import FOLD_NAMES, NOT_IN_A_FOLD, add_fold
from cyclone.samples import LEAD_HOURS, make_init_times, make_truth, with_truth_at
from cyclone.scoring import score, skill, summarize
from cyclone.splits import VAL_LAST_SEASON

HELD_OUT = FOLD_NAMES[2]

# Small, so that the XGBoost leakage tests stay quick.
TINY_SETTINGS = {
    "max_depth": 2,
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "subsample": 1.0,
    "n_estimators": 5,
}


def corrupt_block(truth: pd.DataFrame, rows: pd.DataFrame, fold: str) -> pd.DataFrame:
    """Return truth with every position of one block's storms moved a long way."""
    storms = rows[rows["fold"] == fold]["SID"].unique()
    changed = truth.copy()
    in_block = changed["SID"].isin(storms)
    changed.loc[in_block, "lat"] += 7.0
    changed.loc[in_block, "lon"] -= 11.0
    return changed


def block_forecasts(forecasts: pd.DataFrame, fold: str) -> pd.DataFrame:
    return forecasts[forecasts["fold"] == fold].reset_index(drop=True)


def test_a_model_never_receives_the_block_it_forecasts(rows, truth):
    seen = []

    def spy(learning, learning_truth, held_out):
        block = set(held_out["SID"])
        assert block.isdisjoint(learning["SID"])
        assert block.isdisjoint(learning_truth["SID"])
        assert held_out["fold"].nunique() == 1
        seen.append(held_out["fold"].iloc[0])
        return persistence_forecast(learning, learning_truth, held_out)

    cv.out_of_fold(rows, truth, spy)
    assert seen == list(FOLD_NAMES)


def test_a_model_learns_from_all_four_other_blocks(rows, truth):
    def spy(learning, learning_truth, held_out):
        expected = set(FOLD_NAMES) - set(held_out["fold"])
        assert set(learning["fold"]) == expected
        assert set(learning_truth["SID"]) == set(learning["SID"])
        return persistence_forecast(learning, learning_truth, held_out)

    cv.out_of_fold(rows, truth, spy)


def test_every_start_time_is_forecast_once_at_every_lead(rows, truth):
    forecasts = cv.out_of_fold(rows, truth, cliper_forecast)
    assert len(forecasts) == len(rows) * len(LEAD_HOURS)
    assert not forecasts.duplicated(["SID", "init_time", "lead_h"]).any()
    assert set(forecasts["init_time"]) == set(rows["init_time"])


def test_each_forecast_is_labelled_with_its_storms_block(rows, truth):
    forecasts = cv.out_of_fold(rows, truth, cliper_forecast)
    expected = rows.drop_duplicates("SID").set_index("SID")["fold"]
    assert forecasts["fold"].eq(forecasts["SID"].map(expected)).all()


def test_model_rows_drops_storms_after_the_development_seasons(rows, table):
    assert (table["SEASON"] > VAL_LAST_SEASON).any()
    last = table[table["SEASON"] == VAL_LAST_SEASON]["SID"].iloc[0]
    assert rows["SID"].max() == last


def test_out_of_fold_ignores_rows_outside_the_blocks(table, rows, truth):
    # Even a caller who adds later storms back in can't reach them.
    init_times = make_init_times(table)
    later = add_fold(build_features(init_times), table)
    later = later[later["fold"] == NOT_IN_A_FOLD]
    assert len(later) > 0
    full = pd.concat([rows, later], ignore_index=True)
    tm.assert_frame_equal(
        cv.out_of_fold(rows, truth, cliper_forecast),
        cv.out_of_fold(full, truth, cliper_forecast),
    )


def test_model_rows_refuses_tables_that_do_not_line_up(table):
    init_times = make_init_times(table)
    features = build_features(init_times)
    with pytest.raises(ValueError, match="same rows"):
        cv.model_rows(init_times, features.iloc[1:], table)
    # Same rows in a different order: each storm would get another storm's inputs.
    with pytest.raises(ValueError, match="same rows"):
        cv.model_rows(init_times, features.iloc[::-1].reset_index(drop=True), table)


def test_model_rows_keeps_every_column_a_model_reads(rows):
    assert {"lat_m12", "lon_m24", "dlat_6h", "dist_to_land_km", "fold"} <= set(rows.columns)
    assert rows.columns.is_unique


@pytest.mark.parametrize("fold", FOLD_NAMES)
def test_changing_a_blocks_truth_leaves_its_cliper_forecasts_unchanged(rows, truth, fold):
    before = cv.out_of_fold(rows, truth, cliper_forecast)
    after = cv.out_of_fold(rows, corrupt_block(truth, rows, fold), cliper_forecast)
    tm.assert_frame_equal(block_forecasts(before, fold), block_forecasts(after, fold))


@pytest.mark.parametrize(
    "make_forecaster", [xgb_model.xgboost_forecaster, xgb_model.residual_forecaster]
)
def test_changing_a_blocks_truth_leaves_its_xgboost_forecasts_unchanged(
    rows, truth, make_forecaster
):
    forecaster = make_forecaster(TINY_SETTINGS, "cpu")
    before = cv.out_of_fold(rows, truth, forecaster)
    after = cv.out_of_fold(rows, corrupt_block(truth, rows, HELD_OUT), forecaster)

    tm.assert_frame_equal(block_forecasts(before, HELD_OUT), block_forecasts(after, HELD_OUT))
    for fold in set(FOLD_NAMES) - {HELD_OUT}:
        assert not block_forecasts(before, fold)["lat"].equals(block_forecasts(after, fold)["lat"])


def test_xgb_residuals_cliper_is_fitted_on_the_learning_blocks_only(rows, truth, monkeypatch):
    fitted_on = []
    real_fit = xgb_model.fit_cliper_on

    def spy(learning, learning_truth):
        fitted_on.append((set(learning["fold"]), set(learning_truth["SID"])))
        return real_fit(learning, learning_truth)

    monkeypatch.setattr(xgb_model, "fit_cliper_on", spy)
    cv.out_of_fold(rows, truth, xgb_model.residual_forecaster(TINY_SETTINGS, "cpu"))

    assert len(fitted_on) == len(FOLD_NAMES)
    for fold, (folds_seen, storms_seen) in zip(FOLD_NAMES, fitted_on):
        assert folds_seen == set(FOLD_NAMES) - {fold}
        assert storms_seen.isdisjoint(rows[rows["fold"] == fold]["SID"])


def test_changing_a_blocks_truth_does_change_the_other_blocks(rows, truth):
    # The other side of the leakage test: a model that ignored truth altogether would
    # pass it too. This block is learned from by the other four, so they must move.
    before = cv.out_of_fold(rows, truth, cliper_forecast)
    after = cv.out_of_fold(rows, corrupt_block(truth, rows, HELD_OUT), cliper_forecast)
    for fold in set(FOLD_NAMES) - {HELD_OUT}:
        assert not block_forecasts(before, fold)["lat"].equals(block_forecasts(after, fold)["lat"])


def test_a_leaky_runner_would_fail_the_leakage_test(rows, truth):
    # Proof the test can catch a leak: this forecaster reads the held-out block's truth.
    def leaky(learning, learning_truth, held_out):
        everything = pd.concat([learning, held_out], ignore_index=True)
        return cliper_forecast(everything, full_truth, held_out)

    full_truth = truth
    before = cv.out_of_fold(rows, truth, leaky)
    full_truth = corrupt_block(truth, rows, HELD_OUT)
    after = cv.out_of_fold(rows, truth, leaky)
    assert not block_forecasts(before, HELD_OUT)["lat"].equals(block_forecasts(after, HELD_OUT)["lat"])


def make_scored(rows: list[tuple]) -> pd.DataFrame:
    """Return a scored table from (model, SID, hour, lead_h, error_km, fold) rows."""
    table = pd.DataFrame(rows, columns=["model", "SID", "hour", "lead_h", "error_km", "fold"])
    table["init_time"] = pd.Timestamp("2014-10-09") + pd.to_timedelta(table.pop("hour"), unit="h")
    return table


def two_models(errors: dict[str, tuple[float, float]], lead_h: int = 24) -> pd.DataFrame:
    """Return one case per storm for CLIPER and a model, from {SID: (cliper, model)}."""
    rows = []
    for sid, (cliper, model) in errors.items():
        rows.append(("cliper", sid, 0, lead_h, cliper, "a"))
        rows.append(("model", sid, 0, lead_h, model, "a"))
    return make_scored(rows)


def test_bootstrap_skill_is_the_same_number_scoring_gives(rows, truth):
    forecasts = pd.concat(
        [
            cv.out_of_fold(rows, truth, cliper_forecast),
            cv.out_of_fold(rows, truth, persistence_forecast),
        ],
        ignore_index=True,
    )
    scored = score(forecasts, truth)
    expected = skill(summarize(scored), reference="cliper")["persistence"]
    result = cv.bootstrap_skill(scored, n_resamples=200).set_index("lead_h")["skill"]
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_bootstrap_interval_is_narrow_when_every_storm_agrees():
    errors = {f"S{number}": (100.0, 90.0) for number in range(10)}
    result = cv.bootstrap_skill(two_models(errors), n_resamples=500).iloc[0]
    assert result["skill"] == pytest.approx(10.0)
    assert result["low"] == pytest.approx(10.0)
    assert result["high"] == pytest.approx(10.0)


def test_bootstrap_interval_contains_the_skill_and_crosses_zero_for_a_split_vote():
    errors = {f"S{number}": (100.0, 60.0 if number % 2 else 140.0) for number in range(40)}
    result = cv.bootstrap_skill(two_models(errors), n_resamples=2000).iloc[0]
    assert result["skill"] == pytest.approx(0.0)
    assert result["low"] < 0 < result["high"]


def test_bootstrap_is_by_storm_so_more_cases_from_a_storm_add_no_certainty():
    # Each storm's errors repeated at 20 start times. Rows from one storm aren't new
    # evidence, so the interval must not shrink. A bootstrap of rows would shrink it.
    errors = {f"S{number}": (100.0, 80.0 + number) for number in range(30)}
    one_case = two_models(errors)
    many = []
    for sid, (cliper, model) in errors.items():
        for hour in range(0, 120, 6):
            many.append(("cliper", sid, hour, 24, cliper, "a"))
            many.append(("model", sid, hour, 24, model, "a"))
    many_cases = make_scored(many)

    first = cv.bootstrap_skill(one_case, n_resamples=2000).iloc[0]
    second = cv.bootstrap_skill(many_cases, n_resamples=2000).iloc[0]
    assert second["low"] == pytest.approx(first["low"])
    assert second["high"] == pytest.approx(first["high"])
    assert second["cases"] == 20 * first["cases"]
    assert second["storms"] == first["storms"]


def test_bootstrap_repeats_exactly_with_the_same_seed():
    errors = {f"S{number}": (100.0, 80.0 + number) for number in range(30)}
    scored = two_models(errors)
    tm.assert_frame_equal(
        cv.bootstrap_skill(scored, n_resamples=100),
        cv.bootstrap_skill(scored, n_resamples=100),
    )


def test_win_rate_is_the_share_of_cases_a_model_is_closer():
    errors = {"A": (100.0, 90.0), "B": (100.0, 90.0), "C": (100.0, 110.0), "D": (100.0, 100.0)}
    result = cv.win_rate(two_models(errors))
    assert result.loc[24, "model"] == pytest.approx(50.0)


def test_skill_by_fold_reports_each_block_separately():
    scored = make_scored(
        [
            ("cliper", "A", 0, 24, 100.0, "first"),
            ("model", "A", 0, 24, 80.0, "first"),
            ("cliper", "B", 0, 24, 100.0, "second"),
            ("model", "B", 0, 24, 110.0, "second"),
        ]
    )
    result = cv.skill_by_fold(scored)
    assert result.loc[(24, "first"), "model"] == pytest.approx(20.0)
    assert result.loc[(24, "second"), "model"] == pytest.approx(-10.0)


def test_selection_score_is_the_mean_of_each_leads_skill(rows, truth):
    forecasts = pd.concat(
        [
            cv.out_of_fold(rows, truth, persistence_forecast),
            cv.out_of_fold(rows, truth, cliper_forecast),
        ],
        ignore_index=True,
    )
    scored = score(forecasts, truth, report=False)
    table = cv.selection_scores(scored, ["persistence"], n_resamples=50)
    by_lead = cv.bootstrap_skill(scored, n_resamples=50)
    assert table["score"].item() == pytest.approx(by_lead["skill"].mean())
    assert table["low"].item() <= table["score"].item() <= table["high"].item()


def scores_table(errors: dict[str, dict[int, float]], storms: int = 4) -> pd.DataFrame:
    """Build a scored table in which every storm has the same errors."""
    return make_scored(
        [
            (model, f"S{storm}", 0, lead_h, km, "a")
            for model, by_lead in errors.items()
            for lead_h, km in by_lead.items()
            for storm in range(storms)
        ]
    )


def test_selection_ranks_by_score_and_ignores_the_interval():
    scored = scores_table(
        {
            "cliper": {24: 100.0, 72: 100.0},
            "strong_at_one_lead": {24: 70.0, 72: 100.0},
            "steady": {24: 90.0, 72: 95.0},
        }
    )
    table = cv.selection_scores(scored, ["steady", "strong_at_one_lead"], n_resamples=50)
    assert table["model"].tolist() == ["strong_at_one_lead", "steady"]
    assert table["score"].tolist() == pytest.approx([15.0, 7.5])


def test_every_lead_counts_equally_however_large_its_errors():
    # 10% better at +72 h counts the same as 10% better at +24 h, though it saves ten
    # times as many km, so these two tie. Ranking on km would pick the second.
    scored = scores_table(
        {
            "cliper": {24: 100.0, 72: 1000.0},
            "better_at_24": {24: 90.0, 72: 1000.0},
            "better_at_72": {24: 100.0, 72: 900.0},
        }
    )
    table = cv.selection_scores(scored, ["better_at_24", "better_at_72"], n_resamples=50)
    assert table["score"].tolist() == pytest.approx([5.0, 5.0])


def test_an_exact_tie_goes_to_the_simpler_model():
    errors = {"cliper": {24: 100.0}, "simple": {24: 90.0}, "complex": {24: 90.0}}
    table = cv.selection_scores(scores_table(errors), ["simple", "complex"], n_resamples=50)
    assert table["model"].tolist() == ["simple", "complex"]

    table = cv.selection_scores(scores_table(errors), ["complex", "simple"], n_resamples=50)
    assert table["model"].tolist() == ["complex", "simple"]


def test_selection_interval_redraws_a_storm_at_all_its_leads_together():
    # On each storm, the win at one lead exactly cancels the loss at the other, so every
    # redraw scores 0 when a storm moves as a whole. Redrawing each lead separately would
    # break the pairing and give an interval around 0 instead.
    scored = make_scored(
        [
            ("cliper", "A", 0, 24, 100.0, "a"),
            ("cliper", "A", 0, 48, 100.0, "a"),
            ("cliper", "B", 0, 24, 100.0, "a"),
            ("cliper", "B", 0, 48, 100.0, "a"),
            ("model", "A", 0, 24, 80.0, "a"),
            ("model", "A", 0, 48, 120.0, "a"),
            ("model", "B", 0, 24, 120.0, "a"),
            ("model", "B", 0, 48, 80.0, "a"),
        ]
    )
    table = cv.selection_scores(scored, ["model"], n_resamples=500)
    assert table["low"].item() == pytest.approx(0.0, abs=1e-9)
    assert table["high"].item() == pytest.approx(0.0, abs=1e-9)


def test_selection_interval_repeats_exactly_with_the_same_seed():
    scored = scores_table({"cliper": {24: 100.0}, "model": {24: 90.0}})
    first = cv.selection_scores(scored, ["model"], n_resamples=100)
    tm.assert_frame_equal(first, cv.selection_scores(scored, ["model"], n_resamples=100))


def test_storm_gaps_are_sorted_with_the_models_best_storm_first():
    scored = make_scored(
        [
            ("cliper", "A", 0, 72, 100.0, "a"),
            ("model", "A", 0, 72, 150.0, "a"),
            ("cliper", "B", 0, 72, 100.0, "a"),
            ("model", "B", 0, 72, 60.0, "a"),
            ("cliper", "B", 6, 72, 200.0, "a"),
            ("model", "B", 6, 72, 140.0, "a"),
        ]
    )
    result = cv.storm_gaps(scored, "model")
    assert result.index.tolist() == ["B", "A"]
    assert result.loc["B", "gap_km"] == pytest.approx(-50.0)
    assert result.loc["B", "cases"] == 2


def test_comparisons_use_identical_cases_only():
    # Storm B has no CLIPER forecast, so it can't count for either model.
    scored = make_scored(
        [
            ("cliper", "A", 0, 24, 100.0, "a"),
            ("model", "A", 0, 24, 80.0, "a"),
            ("model", "B", 0, 24, 900.0, "a"),
        ]
    )
    assert cv.bootstrap_skill(scored, n_resamples=50).iloc[0]["skill"] == pytest.approx(20.0)
    assert cv.win_rate(scored).loc[24, "model"] == pytest.approx(100.0)
    assert len(cv.storm_gaps(scored, "model", lead_h=24)) == 1


def test_real_cases_from_with_truth_at_match_the_forecasts_scored(rows, truth):
    # Every case that has truth is scored, so none is lost between the two tables.
    forecasts = cv.out_of_fold(rows, truth, cliper_forecast)
    scored = score(forecasts, truth)
    for lead_h in LEAD_HOURS:
        expected = len(with_truth_at(rows, truth, lead_h))
        at_lead = scored[scored["lead_h"] == lead_h]
        assert at_lead["error_km"].notna().sum() == expected


def test_learning_error_is_scored_on_the_storms_a_model_learned_from(rows, truth):
    seen = []

    def perfect(learning, learning_truth, forecast_rows):
        seen.append((learning, learning_truth, forecast_rows))
        forecasts = []
        for lead_h in LEAD_HOURS:
            cases = with_truth_at(forecast_rows, learning_truth, lead_h)
            forecasts.append(
                pd.DataFrame(
                    {
                        "model": "perfect",
                        "SID": cases["SID"],
                        "init_time": cases["init_time"],
                        "lead_h": lead_h,
                        "lat": cases["truth_lat"],
                        "lon": cases["truth_lon"],
                    }
                )
            )
        return pd.concat(forecasts, ignore_index=True)

    errors = cv.learning_error(rows, truth, perfect)

    assert errors.index.tolist() == list(LEAD_HOURS)
    assert (errors == 0).all()
    assert len(seen) == len(FOLD_NAMES)
    for fold, (learning, learning_truth, forecast_rows) in zip(FOLD_NAMES, seen):
        assert fold not in set(learning["fold"])
        assert forecast_rows is learning
        assert set(learning_truth["SID"]) == set(learning["SID"])
