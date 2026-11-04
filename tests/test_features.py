"""Tests for the model inputs.

The leakage tests matter most. A model that can see the future scores brilliantly and
is useless, and nothing else in the pipeline would notice. So they rewrite a storm's
future and rebuild through make_init_times as well as build_features, because a leak
in either one would count.
"""

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest

from cyclone.features import FEATURE_COLUMNS, build_features, distance_to_land_km
from cyclone.samples import make_init_times

START = pd.Timestamp("2014-10-07 00:00")
T0 = pd.Timestamp("2014-10-10 00:00")
SIX_HOURS = pd.Timedelta(hours=6)


def make_storm(n_fixes: int = 30, sid: str = "TEST") -> pd.DataFrame:
    """Return a storm in the cleaned table's format, curving north-west across the Bay."""
    steps = np.arange(n_fixes)
    return pd.DataFrame(
        {
            "SID": sid,
            "ISO_TIME": START + steps * SIX_HOURS,
            "split": "train",
            "USA_LAT": 9.0 + 0.3 * steps + 0.02 * steps**2,
            "USA_LON": 92.0 - 0.5 * steps,
            "USA_WIND": 30.0 + 3.0 * steps,
        }
    )


def features_of(track: pd.DataFrame) -> pd.DataFrame:
    return build_features(make_init_times(track)).set_index("init_time")


def test_changing_the_future_leaves_features_up_to_t0_unchanged():
    storm = make_storm()
    after_t0 = storm["ISO_TIME"] > T0
    changed = storm.copy()
    changed.loc[after_t0, "USA_LAT"] += 20.0
    changed.loc[after_t0, "USA_LON"] -= 30.0
    changed.loc[after_t0, "USA_WIND"] = 999.0

    before, after = features_of(storm), features_of(changed)

    tm.assert_frame_equal(before.loc[:T0], after.loc[:T0])
    # Proof that the change reached the pipeline. Without this, the check above would
    # also pass if the rewritten rows had never been used at all.
    first_changed = T0 + SIX_HOURS
    assert (before.loc[first_changed:, "lat"] != after.loc[first_changed:, "lat"]).all()


def test_deleting_the_future_leaves_features_up_to_t0_unchanged():
    storm = make_storm()
    before = features_of(storm).loc[:T0]
    after = features_of(storm[storm["ISO_TIME"] <= T0])
    tm.assert_frame_equal(before, after)


@pytest.mark.parametrize("hours_before", [0, 6, 12, 24])
def test_changing_a_past_fix_does_change_the_features_at_t0(hours_before):
    # The other side of the leakage test: inputs that ignored the track entirely would
    # pass it too. Every fix that goes into the features must have an effect.
    storm = make_storm()
    moved = storm["ISO_TIME"] == T0 - pd.Timedelta(hours=hours_before)
    changed = storm.copy()
    changed.loc[moved, "USA_LAT"] += 1.0

    before = features_of(storm).loc[T0, FEATURE_COLUMNS]
    after = features_of(changed).loc[T0, FEATURE_COLUMNS]

    assert not before.equals(after)


def test_a_gap_in_the_history_removes_the_start_time():
    # Rows aren't always 6 h apart, so history is matched by timestamp, not by position.
    storm = make_storm()
    without_one_fix = storm[storm["ISO_TIME"] != T0 - pd.Timedelta(hours=12)]
    assert T0 not in features_of(without_one_fix).index


def test_motion_of_a_storm_moving_steadily_north():
    # 1 degree of latitude every 6 h is 111.19 km, so 18.53 km/h, heading due north.
    storm = make_storm(n_fixes=5)
    storm["USA_LAT"] = [10.0, 11.0, 12.0, 13.0, 14.0]
    storm["USA_LON"] = 88.0

    row = features_of(storm).iloc[0]

    assert (row["dlat_6h"], row["dlat_12h"], row["dlat_24h"]) == (1.0, 2.0, 4.0)
    assert row["dlon_12h"] == 0.0
    assert row["speed_kmh"] == pytest.approx(18.53, abs=0.01)
    assert row["heading_sin"] == pytest.approx(0.0)
    assert row["heading_cos"] == pytest.approx(1.0)
    assert row["dlat_change"] == 0.0  # steady motion, so no change in it


def test_change_in_motion_shows_a_storm_speeding_up():
    # 1 degree north in the first 12 h, then 3 degrees in the last 12 h.
    storm = make_storm(n_fixes=5)
    storm["USA_LAT"] = [10.0, 10.5, 11.0, 12.5, 14.0]
    assert features_of(storm).iloc[0]["dlat_change"] == pytest.approx(2.0)


def test_heading_allows_for_shorter_degrees_of_longitude():
    # At 60N a degree of longitude is half a degree of latitude. So 1 degree north and
    # 2 degrees east is north-east, not the 63 degrees that plain subtraction gives.
    storm = make_storm(n_fixes=5)
    storm["USA_LAT"] = [58.0, 58.5, 59.0, 59.5, 60.0]
    storm["USA_LON"] = [84.0, 85.0, 86.0, 87.0, 88.0]
    row = features_of(storm).iloc[0]
    heading = np.degrees(np.arctan2(row["heading_sin"], row["heading_cos"]))
    assert heading == pytest.approx(45, abs=1)


def test_a_storm_that_has_not_moved_has_no_heading():
    storm = make_storm(n_fixes=5)
    storm["USA_LAT"], storm["USA_LON"] = 15.0, 88.0
    row = features_of(storm).iloc[0]
    assert row["speed_kmh"] == 0.0
    assert np.isnan(row["heading_sin"]) and np.isnan(row["heading_cos"])


def test_missing_wind_stays_missing_and_keeps_the_row():
    storm = make_storm(n_fixes=6)
    storm["USA_WIND"] = np.nan
    features = features_of(storm)
    assert len(features) == 2
    assert features[["wind", "dwind_12h", "dwind_24h"]].isna().all(axis=None)


def test_new_year_is_not_a_jump():
    # Two days apart here, since the first start time comes 24 h after the first fix.
    # Two days around a circle of 365 is 0.034. As plain numbers it would be 363 days.
    def day_on_circle(when: str) -> np.ndarray:
        storm = make_storm(n_fixes=5)
        storm["ISO_TIME"] = pd.Timestamp(when) + np.arange(5) * SIX_HOURS
        return features_of(storm).iloc[0][["day_sin", "day_cos"]].to_numpy(dtype=float)

    assert np.linalg.norm(day_on_circle("2013-12-30") - day_on_circle("2014-01-01")) < 0.05


def test_rows_stay_in_order_and_none_are_dropped():
    tracks = pd.concat([make_storm(sid="A"), make_storm(sid="B")], ignore_index=True)
    init_times = make_init_times(tracks)
    features = build_features(init_times)
    tm.assert_frame_equal(features[["SID", "init_time"]], init_times[["SID", "init_time"]])
    assert list(features.columns[3:]) == FEATURE_COLUMNS


def test_distance_to_land_is_zero_inland():
    # Nagpur, in the middle of India.
    assert distance_to_land_km([21.15], [79.09])[0] == 0.0


def test_distance_to_land_is_positive_in_the_open_bay():
    # The middle of the Bay of Bengal, far from India, Sri Lanka and the Andamans.
    assert distance_to_land_km([14.0], [87.0])[0] > 300.0


def test_distance_to_land_is_measured_in_km():
    # Half a degree of latitude south of Sri Lanka's southern tip, Dondra Head at
    # 5.92N, is about 55 km from the coast. This fixes the units and the lat/lon order.
    assert distance_to_land_km([5.42], [80.59])[0] == pytest.approx(55, abs=8)
