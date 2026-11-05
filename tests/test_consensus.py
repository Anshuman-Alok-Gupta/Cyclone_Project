"""Tests for the average of several models' forecasts."""

import pandas as pd
import pytest

from cyclone.consensus import CONSENSUS_LSTM_NAME, CONSENSUS_NAME, consensus
from cyclone.scoring import FORECAST_COLUMNS

T0 = pd.Timestamp("2014-10-09 00:00")
BOTH = ("cliper", "xgboost")


def make_forecasts(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=FORECAST_COLUMNS)


def test_a_consensus_can_carry_its_own_name_and_any_number_of_members():
    forecasts = make_forecasts(
        [
            ("cliper", "S1", T0, 24, 10.0, 80.0),
            ("xgboost", "S1", T0, 24, 13.0, 83.0),
            ("lstm", "S1", T0, 24, 16.0, 86.0),
        ]
    )
    result = consensus(forecasts, ("cliper", "xgboost", "lstm"), CONSENSUS_LSTM_NAME)
    assert result["model"].tolist() == [CONSENSUS_LSTM_NAME]
    assert result["lat"].iloc[0] == pytest.approx(13.0)
    assert result["lon"].iloc[0] == pytest.approx(83.0)


def test_it_is_the_mean_latitude_and_mean_longitude_of_each_case():
    forecasts = make_forecasts(
        [
            ("cliper", "S1", T0, 24, 10.0, 80.0),
            ("xgboost", "S1", T0, 24, 14.0, 90.0),
            ("cliper", "S1", T0, 48, 20.0, 70.0),
            ("xgboost", "S1", T0, 48, 21.0, 72.0),
        ]
    )
    result = consensus(forecasts, BOTH).set_index("lead_h")
    assert result.loc[24, "lat"] == pytest.approx(12.0)
    assert result.loc[24, "lon"] == pytest.approx(85.0)
    assert result.loc[48, "lat"] == pytest.approx(20.5)
    assert result.loc[48, "lon"] == pytest.approx(71.0)


def test_it_returns_the_scoring_format_under_its_own_name():
    forecasts = make_forecasts(
        [("cliper", "S1", T0, 24, 10.0, 80.0), ("xgboost", "S1", T0, 24, 14.0, 90.0)]
    )
    result = consensus(forecasts, BOTH)
    assert result.columns.tolist() == FORECAST_COLUMNS
    assert (result["model"] == CONSENSUS_NAME).all()


def test_other_models_in_the_table_are_ignored():
    forecasts = make_forecasts(
        [
            ("cliper", "S1", T0, 24, 10.0, 80.0),
            ("xgboost", "S1", T0, 24, 14.0, 90.0),
            ("persistence", "S1", T0, 24, 50.0, 10.0),
        ]
    )
    assert consensus(forecasts, BOTH)["lat"].iloc[0] == pytest.approx(12.0)


def test_the_fold_column_is_kept():
    forecasts = make_forecasts(
        [("cliper", "S1", T0, 24, 10.0, 80.0), ("xgboost", "S1", T0, 24, 14.0, 90.0)]
    ).assign(fold="1980-1987")
    assert consensus(forecasts, BOTH)["fold"].tolist() == ["1980-1987"]


def test_a_case_missing_a_member_stops_the_run():
    forecasts = make_forecasts(
        [
            ("cliper", "S1", T0, 24, 10.0, 80.0),
            ("xgboost", "S1", T0, 24, 14.0, 90.0),
            ("cliper", "S2", T0, 24, 11.0, 81.0),
        ]
    )
    with pytest.raises(ValueError, match="1 cases"):
        consensus(forecasts, BOTH)


def test_a_member_missing_altogether_stops_the_run():
    forecasts = make_forecasts([("cliper", "S1", T0, 24, 10.0, 80.0)])
    with pytest.raises(ValueError, match="xgboost"):
        consensus(forecasts, BOTH)
