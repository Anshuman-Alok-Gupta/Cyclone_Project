"""Tests for the scoring harness.

Every score in the project passes through scoring.py, so a bug there would silently
change every result. The distance tests use answers known from geometry: one degree
of arc on a sphere of radius 6371 km is 6371 * pi / 180 = 111.19 km.
"""

import ast
from pathlib import Path

import pandas as pd
import pytest

from cyclone import scoring
from cyclone.scoring import great_circle_km, score, summarize

KM_PER_DEGREE = 111.19
T0 = pd.Timestamp("2014-10-09 00:00")


def make_forecasts(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["model", "SID", "init_time", "lead_h", "lat", "lon"])


def make_truth(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["SID", "valid_time", "lat", "lon"])


def test_identical_points_are_zero_km():
    assert great_circle_km(15.0, 85.0, 15.0, 85.0) == 0.0


def test_one_degree_of_latitude():
    assert great_circle_km(10.0, 85.0, 11.0, 85.0) == pytest.approx(KM_PER_DEGREE, abs=0.01)


def test_one_degree_of_longitude_at_the_equator():
    assert great_circle_km(0.0, 85.0, 0.0, 86.0) == pytest.approx(KM_PER_DEGREE, abs=0.01)


def test_one_degree_of_longitude_shrinks_away_from_the_equator():
    # At 60N a degree of longitude is half its length at the equator.
    assert great_circle_km(60.0, 85.0, 60.0, 86.0) == pytest.approx(KM_PER_DEGREE / 2, abs=0.02)


def test_forecast_is_matched_at_init_time_plus_lead_time():
    forecasts = make_forecasts([("m", "A", T0, 24, 11.0, 85.0)])
    truth = make_truth(
        [
            ("A", T0, 50.0, 50.0),  # the start time: must not be used
            ("A", T0 + pd.Timedelta(hours=24), 10.0, 85.0),
        ]
    )
    scored = score(forecasts, truth)
    assert scored["error_km"].iloc[0] == pytest.approx(KM_PER_DEGREE, abs=0.01)


def test_truth_from_another_storm_is_not_used():
    forecasts = make_forecasts([("m", "A", T0, 24, 11.0, 85.0)])
    truth = make_truth([("B", T0 + pd.Timedelta(hours=24), 11.0, 85.0)])
    assert score(forecasts, truth)["error_km"].isna().all()


def test_forecast_with_no_truth_is_reported_not_scored(capsys):
    forecasts = make_forecasts(
        [
            ("m", "A", T0, 24, 11.0, 85.0),
            ("m", "A", T0, 48, 12.0, 85.0),  # the storm has no fix at +48 h
        ]
    )
    truth = make_truth([("A", T0 + pd.Timedelta(hours=24), 10.0, 85.0)])

    scored = score(forecasts, truth)

    # Kept in the table, with no error, and reported.
    assert len(scored) == 2
    assert scored.loc[scored["lead_h"] == 48, "error_km"].isna().all()
    assert "1 had no matching truth" in capsys.readouterr().out
    # Not counted in the averages.
    summary = summarize(scored)
    assert summary["lead_h"].tolist() == [24]
    assert summary["count"].tolist() == [1]


def test_summarize_keeps_only_cases_every_model_has():
    truth = make_truth(
        [
            ("A", T0 + pd.Timedelta(hours=24), 10.0, 85.0),
            ("B", T0 + pd.Timedelta(hours=24), 10.0, 85.0),
        ]
    )
    forecasts = make_forecasts(
        [
            ("both", "A", T0, 24, 11.0, 85.0),  # 1 degree off
            ("both", "B", T0, 24, 13.0, 85.0),  # 3 degrees off: a case only "both" has
            ("one", "A", T0, 24, 11.0, 85.0),
        ]
    )
    scored = score(forecasts, truth)

    fair = summarize(scored, homogeneous=True).set_index("model")
    assert fair["count"].to_dict() == {"both": 1, "one": 1}
    assert fair.loc["both", "mean_km"] == pytest.approx(fair.loc["one", "mean_km"])

    every_case = summarize(scored, homogeneous=False).set_index("model")
    assert every_case["count"].to_dict() == {"both": 2, "one": 1}
    assert every_case.loc["both", "mean_km"] == pytest.approx(2 * KM_PER_DEGREE, abs=0.01)


def test_repeated_forecast_is_rejected():
    forecast = ("m", "A", T0, 24, 11.0, 85.0)
    truth = make_truth([("A", T0 + pd.Timedelta(hours=24), 10.0, 85.0)])
    with pytest.raises(ValueError, match="repeat"):
        score(make_forecasts([forecast, forecast]), truth)


def test_forecast_with_no_position_is_rejected():
    truth = make_truth([("A", T0 + pd.Timedelta(hours=24), 10.0, 85.0)])
    with pytest.raises(ValueError, match="no position"):
        score(make_forecasts([("m", "A", T0, 24, float("nan"), 85.0)]), truth)


def test_times_can_differ_in_precision_or_arrive_as_text():
    # AI-model output won't necessarily store times the way the cleaned table does.
    truth = make_truth([("A", T0 + pd.Timedelta(hours=24), 10.0, 85.0)])
    truth["valid_time"] = truth["valid_time"].astype("datetime64[us]")
    forecasts = make_forecasts([("m", "A", str(T0), 24, 11.0, 85.0)])
    assert score(forecasts, truth)["error_km"].notna().all()


def test_scoring_imports_nothing_else_from_the_package():
    # Project 3 reuses this file on its own, so this rule is worth enforcing.
    tree = ast.parse(Path(scoring.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported == {"numpy", "pandas"}
