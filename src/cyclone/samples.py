"""Turn the cleaned table into the two tables every forecast model works from.

A row in the cleaned table is one moment in a storm. A forecast needs more than that:
where the storm is now, where it was, and later on where it really went. This module
keeps those apart on purpose:

    make_init_times: what a forecaster knows at the start time (t0 and earlier)
    make_truth:      where the storms really were, used only for targets and scoring

The start-time table has no future positions in it at all. A model built from it
can't look ahead by accident, because the answer isn't in the table it reads.
"""

import pandas as pd

# Hours before t0 that must all have a JTWC fix. 24 h of history lets the baselines
# measure the storm's motion over the past 12 h and the past 24 h.
HISTORY_HOURS = (6, 12, 18, 24)

# Hours after t0 that the project forecasts.
LEAD_HOURS = (24, 48, 72)


def make_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Return the cleaned table as the reference positions forecasts are scored on.

    Args:
        df: the cleaned table, with JTWC positions in USA_LAT/USA_LON.

    Returns:
        One row per fix, with columns SID, valid_time, lat, lon. These names don't
        mention IBTrACS, so scoring.py can take this table or any other track source.
    """
    truth = df[["SID", "ISO_TIME", "USA_LAT", "USA_LON"]].rename(
        columns={"ISO_TIME": "valid_time", "USA_LAT": "lat", "USA_LON": "lon"}
    )
    return truth.reset_index(drop=True)


def make_init_times(df: pd.DataFrame) -> pd.DataFrame:
    """Return every fix that has a full 24 h of history, as forecast start times.

    Past fixes are found by matching timestamps. Shifting rows by 1, 2, 3 and 4 would
    be wrong for the 4 storms with a 12 h gap, where the row 4 places back is 30 h
    earlier, not 24 h. Each merge below keeps a row only if a fix exists at exactly
    that past time, so a gap removes the start times it affects.

    Args:
        df: the cleaned table with its split column.

    Returns:
        One row per start time, with columns SID, init_time, split, lat, lon, wind,
        then lat_m06, lon_m06, ... lon_m24, the position 6 to 24 h before t0. wind is
        USA_WIND at t0 and can be missing in 1980.
    """
    table = df[["SID", "ISO_TIME", "split", "USA_LAT", "USA_LON", "USA_WIND"]].rename(
        columns={
            "ISO_TIME": "init_time",
            "USA_LAT": "lat",
            "USA_LON": "lon",
            "USA_WIND": "wind",
        }
    )

    for hours in HISTORY_HOURS:
        past = df[["SID", "ISO_TIME", "USA_LAT", "USA_LON"]].rename(
            columns={"USA_LAT": f"lat_m{hours:02d}", "USA_LON": f"lon_m{hours:02d}"}
        )
        # Move each past fix forward by `hours` so it lines up with the t0 it's
        # history for. validate="1:1" stops the run if a storm repeats a timestamp.
        past["init_time"] = past.pop("ISO_TIME") + pd.Timedelta(hours=hours)
        table = table.merge(past, on=["SID", "init_time"], how="inner", validate="1:1")

    return table.reset_index(drop=True)


def with_truth_at(
    init_times: pd.DataFrame,
    truth: pd.DataFrame,
    lead_h: int,
) -> pd.DataFrame:
    """Return the start times that have a true position lead_h hours later.

    This is the one place a start time meets a future position, so use it only for
    training targets and counts, never to build model inputs.

    Args:
        init_times: the table from ``make_init_times``.
        truth: the table from ``make_truth``.
        lead_h: hours after t0.

    Returns:
        The matching rows of init_times, plus truth_lat and truth_lon. The storm may
        have ended, left the basin or lost a fix by then, so some rows have no match.
    """
    target = truth.rename(columns={"lat": "truth_lat", "lon": "truth_lon"})
    wanted = init_times.assign(valid_time=init_times["init_time"] + pd.Timedelta(hours=lead_h))
    matched = wanted.merge(target, on=["SID", "valid_time"], how="inner", validate="1:1")
    return matched.drop(columns="valid_time")
