"""Tests for the LSTM's input.

The first group holds the rules that make a forecast honest: nothing after t0, and a gap
ends the history. The second covers blanks, since one blank reaching an LSTM would turn
everything after it blank. The third is about scaling, which must be fitted on the learning
rows only. The fold-by-fold version of that, through a real model, is in test_lstm_model.py.

The data is the set of made-up storms in conftest.py.
"""

import numpy as np
import pandas as pd
import pytest

from cyclone import checks, sequences
from cyclone.checks import check_sequences
from cyclone.features import build_features
from cyclone.samples import make_init_times, make_truth
from cyclone.sequences import (
    FIXED_COLUMNS,
    MAX_HISTORY_H,
    MIN_STEPS,
    STEP_COLUMNS,
    Sequences,
    build_sequences,
    fit_scaler,
    scale,
    select,
)

SIX_HOURS = pd.Timedelta(hours=6)
LONGEST = MAX_HISTORY_H // 6 + 1
DLAT, DLON, LAT, LON, WIND, FLAG = range(len(STEP_COLUMNS))


def rows_of(storms: pd.DataFrame) -> pd.DataFrame:
    """Return the start times and fixed inputs for a table of made-up storms."""
    init_times = make_init_times(storms)
    columns = ["SID", "init_time", "lat", "lon", *FIXED_COLUMNS]
    return init_times.merge(build_features(init_times)[["SID", "init_time", *FIXED_COLUMNS]])[columns]


def one_storm(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["SID"] == table["SID"].iloc[0]].reset_index(drop=True)


def without_row(storm: pd.DataFrame, number: int) -> pd.DataFrame:
    """Return the storm with one fix removed, as cleaning removes an interpolated one."""
    return storm.drop(index=number).reset_index(drop=True)


def real_steps(built: Sequences, number: int) -> np.ndarray:
    return built.steps[number, : built.lengths[number]]


def test_there_is_one_sequence_per_start_time_in_the_same_order(rows, table):
    built = build_sequences(rows, table)
    assert built.keys.equals(rows[["SID", "init_time"]].reset_index(drop=True))
    assert built.steps.shape == (len(rows), LONGEST, len(STEP_COLUMNS))
    assert built.fixed.shape == (len(rows), len(FIXED_COLUMNS))


def test_every_sequence_ends_at_t0_and_steps_6_h_at_a_time(table):
    storm = one_storm(table)
    rows = rows_of(storm)
    built = build_sequences(rows, storm)
    number = len(rows) - 1

    steps = real_steps(built, number)
    expected = storm[storm["ISO_TIME"] <= rows["init_time"].iloc[number]].tail(LONGEST)
    assert steps[:, LAT].tolist() == expected["USA_LAT"].tolist()
    assert steps[:, LON].tolist() == expected["USA_LON"].tolist()
    assert steps[:, WIND].tolist() == expected["USA_WIND"].tolist()


def test_a_sequence_reaches_back_72_h_at_most_and_is_at_least_5_steps(table):
    storm = one_storm(table)
    built = build_sequences(rows_of(storm), storm)
    assert built.lengths[0] == MIN_STEPS
    assert built.lengths.tolist() == sorted(built.lengths)
    assert built.lengths.max() == LONGEST
    assert MAX_HISTORY_H // 6 + 1 == 13


def test_the_change_is_each_fix_minus_the_one_6_h_before(table):
    storm = one_storm(table)
    built = build_sequences(rows_of(storm), storm)
    steps = real_steps(built, 3)
    np.testing.assert_allclose(steps[1:, DLAT], np.diff(steps[:, LAT]))
    np.testing.assert_allclose(steps[1:, DLON], np.diff(steps[:, LON]))


def test_changing_fixes_after_t0_leaves_a_storms_sequences_unchanged(table):
    storm = one_storm(table)
    rows = rows_of(storm)
    middle = rows["init_time"].iloc[len(rows) // 2]

    moved = storm.copy()
    later = moved["ISO_TIME"] > middle
    moved.loc[later, ["USA_LAT", "USA_LON"]] += 30.0
    moved.loc[later, "USA_WIND"] = 999.0

    before = build_sequences(rows, storm)
    after = build_sequences(rows, moved)
    upto = rows["init_time"] <= middle
    np.testing.assert_array_equal(before.steps[upto], after.steps[upto])
    assert not np.array_equal(before.steps[~upto], after.steps[~upto], equal_nan=True)


def test_changing_a_fix_before_t0_does_change_the_sequence(table):
    # A partner to the test above: inputs that ignored the track would pass that one too.
    storm = one_storm(table)
    rows = rows_of(storm)
    moved = storm.copy()
    moved.loc[0, "USA_LAT"] += 1.0
    assert not np.array_equal(
        build_sequences(rows, storm).steps[0],
        build_sequences(rows, moved).steps[0],
        equal_nan=True,
    )


def test_a_gap_starts_a_new_sequence(table):
    storm = without_row(one_storm(table), 5)
    rows = rows_of(storm)
    built = build_sequences(rows, storm)

    # The fix 30 h before this start time isn't 6 h from the next one, so the walk stops.
    just_after_gap = storm["ISO_TIME"].iloc[5] - 2 * SIX_HOURS + 6 * SIX_HOURS
    number = rows.index[rows["init_time"] == just_after_gap][0]
    steps = real_steps(built, number)

    assert built.lengths[number] == MIN_STEPS
    assert steps[0, LAT] == storm.loc[storm["ISO_TIME"] == just_after_gap - 4 * SIX_HOURS, "USA_LAT"].iloc[0]


def test_nothing_from_before_a_gap_is_read(table):
    storm = without_row(one_storm(table), 5)
    rows = rows_of(storm)
    moved = storm.copy()
    moved.loc[moved.index < 5, ["USA_LAT", "USA_LON", "USA_WIND"]] = 999.0

    after_gap = rows["init_time"] >= storm["ISO_TIME"].iloc[5]
    np.testing.assert_array_equal(
        build_sequences(rows, storm).steps[after_gap],
        build_sequences(rows, moved).steps[after_gap],
    )


def test_every_start_time_after_a_gap_reads_only_fixes_after_it(table):
    storm = without_row(one_storm(table), 5)
    rows = rows_of(storm)
    built = build_sequences(rows, storm)
    gap_end = storm["ISO_TIME"].iloc[5]
    for number, init_time in enumerate(rows["init_time"]):
        if init_time >= gap_end:
            assert built.lengths[number] == min((init_time - gap_end) // SIX_HOURS + 1, LONGEST)


def test_the_first_change_is_blank_and_the_rest_are_not(table):
    storm = one_storm(table)
    built = build_sequences(rows_of(storm), storm)
    steps = real_steps(built, len(built) - 1)
    assert np.isnan(steps[0, [DLAT, DLON]]).all()
    assert not np.isnan(steps[1:, [DLAT, DLON]]).any()


def test_a_limit_on_history_leaves_no_trace_of_earlier_fixes(table):
    # The first change is blank even though a fix exists before it. If it weren't, the
    # 24 h model would be reading a position from 30 h ago through that change.
    storm = one_storm(table)
    rows = rows_of(storm)
    built = build_sequences(rows, storm, history_h=24)
    assert built.steps.shape[1] == MIN_STEPS
    assert (built.lengths == MIN_STEPS).all()
    assert np.isnan(built.steps[:, 0, [DLAT, DLON]]).all()

    moved = storm.copy()
    moved.loc[0, ["USA_LAT", "USA_LON", "USA_WIND"]] = 999.0
    last = len(rows) - 1
    np.testing.assert_array_equal(
        built.steps[last],
        build_sequences(rows, moved, history_h=24).steps[last],
    )


def test_the_padding_is_blank_after_a_short_sequence(table):
    storm = one_storm(table)
    built = build_sequences(rows_of(storm), storm)
    assert np.isnan(built.steps[0, MIN_STEPS:]).all()


def test_a_missing_wind_is_flagged_and_a_known_wind_is_not(table):
    storm = one_storm(table)
    gappy = storm.copy()
    gappy.loc[3, "USA_WIND"] = np.nan
    rows = rows_of(gappy)
    steps = real_steps(build_sequences(rows, gappy), 0)

    missing = np.isnan(steps[:, WIND])
    assert missing.tolist() == [False, False, False, True, False]
    assert steps[:, FLAG].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]


def test_scaling_leaves_no_blanks_for_the_network(table):
    storm = one_storm(table)
    gappy = storm.copy()
    gappy.loc[3, "USA_WIND"] = np.nan
    built = build_sequences(rows_of(gappy), gappy)
    steps, fixed = scale(built, fit_scaler(built))
    assert not np.isnan(steps).any()
    assert not np.isnan(fixed).any()


def test_a_missing_wind_is_filled_with_the_average_of_the_known_winds(table):
    storm = one_storm(table)
    gappy = storm.copy()
    gappy.loc[3, "USA_WIND"] = np.nan
    built = build_sequences(rows_of(gappy), gappy)
    scaler = fit_scaler(built)
    steps, _ = scale(built, scaler)

    assert scaler.step_mean[WIND] == pytest.approx(np.nanmean(built.steps[:, :, WIND]))
    assert steps[0, 3, WIND] == 0.0
    assert steps[0, 3, FLAG] == 1.0


def test_scaled_inputs_have_mean_0_and_spread_1_on_the_rows_fitted(rows, table):
    built = build_sequences(rows, table)
    scaler = fit_scaler(built)
    by_hand = (built.steps - scaler.step_mean) / scaler.step_std
    for column in (DLAT, DLON, LAT, LON, WIND):
        assert np.nanmean(by_hand[:, :, column]) == pytest.approx(0.0, abs=1e-9)
        assert np.nanstd(by_hand[:, :, column]) == pytest.approx(1.0)


def test_the_flag_is_not_scaled(rows, table):
    built = build_sequences(rows, table)
    scaler = fit_scaler(built)
    assert scaler.step_mean[FLAG] == 0.0
    assert scaler.step_std[FLAG] == 1.0


def test_a_scaler_is_not_changed_by_rows_it_was_not_fitted_on(rows, table):
    built = build_sequences(rows, table)
    learning_part = rows["fold"] != rows["fold"].iloc[0]
    learning = select(built, rows[learning_part])
    held_out = rows[~learning_part].index

    corrupted = built.steps.copy()
    corrupted[held_out] += 1000.0
    changed = Sequences(built.keys, corrupted, built.lengths, built.fixed)

    first = fit_scaler(learning)
    second = fit_scaler(select(changed, rows[learning_part]))
    np.testing.assert_array_equal(first.step_mean, second.step_mean)
    np.testing.assert_array_equal(first.step_std, second.step_std)
    # And the partner: a scaler fitted on everything would have moved.
    assert not np.array_equal(first.step_mean, fit_scaler(changed).step_mean)


def test_a_constant_input_does_not_divide_by_zero(table):
    storm = one_storm(table)
    flat = storm.assign(USA_WIND=50.0)
    built = build_sequences(rows_of(flat), flat)
    steps, _ = scale(built, fit_scaler(built))
    assert not np.isnan(steps).any()
    assert np.isfinite(steps).all()


def test_select_matches_by_key_not_by_position(rows, table):
    built = build_sequences(rows, table)
    backwards = rows.iloc[::-1]
    chosen = select(built, backwards)
    assert chosen.keys.equals(backwards[["SID", "init_time"]].reset_index(drop=True))
    np.testing.assert_array_equal(chosen.steps[0], built.steps[-1])
    np.testing.assert_array_equal(chosen.fixed[0], built.fixed[-1])


def test_select_refuses_a_row_with_no_sequence(rows, table):
    built = build_sequences(rows.iloc[:5], table)
    with pytest.raises(ValueError, match="no sequence"):
        select(built, rows.iloc[:6])


def test_a_start_time_that_is_not_a_fix_is_refused(table):
    storm = one_storm(table)
    rows = rows_of(storm).assign(init_time=lambda frame: frame["init_time"] + pd.Timedelta(hours=1))
    with pytest.raises(ValueError, match="no fix"):
        build_sequences(rows, storm)


def test_a_repeated_timestamp_is_refused(table):
    storm = one_storm(table)
    rows = rows_of(storm)
    with pytest.raises(ValueError, match="repeats"):
        build_sequences(rows, pd.concat([storm, storm.iloc[[0]]]))


def test_a_start_time_with_too_little_history_is_refused(table):
    storm = without_row(one_storm(table), 5)
    rows = rows_of(storm)
    too_short = rows.assign(init_time=storm["ISO_TIME"].iloc[5 + 1])
    with pytest.raises(ValueError, match="fewer than the minimum"):
        build_sequences(too_short.iloc[[0]], storm)


def test_the_check_refuses_sequences_for_different_start_times(rows, table, monkeypatch):
    monkeypatch.setattr(checks, "check_folds", lambda *args: None)
    built = build_sequences(rows, table)
    check_sequences(built, rows, table, make_truth(table))
    with pytest.raises(AssertionError, match="same start times"):
        check_sequences(built, rows.iloc[::-1], table, make_truth(table))


def test_the_check_refuses_a_sequence_that_ends_somewhere_else(rows, table, monkeypatch):
    monkeypatch.setattr(checks, "check_folds", lambda *args: None)
    built = build_sequences(rows, table)
    with pytest.raises(AssertionError, match="doesn't end"):
        check_sequences(built, rows.assign(lat=rows["lat"] + 1.0), table, make_truth(table))


def test_the_check_runs_the_fold_counts_on_the_sequences_own_keys(rows, table, monkeypatch):
    seen = []
    monkeypatch.setattr(checks, "check_folds", lambda df, keys, truth: seen.append(keys))
    check_sequences(build_sequences(rows, table), rows, table, make_truth(table))
    assert len(seen[0]) == len(rows)


def test_sequences_module_never_reads_truth():
    # It's the reason a future position can't reach an input from this file.
    assert "with_truth_at" not in vars(sequences)
    assert "make_truth" not in vars(sequences)
