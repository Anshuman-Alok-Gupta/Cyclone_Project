"""The five row filters from CLAUDE.md, and clean_ni() which applies them in order.

Each filter is a small function that takes a table and returns the rows it keeps.
Keeping them separate lets later work reuse just the ones it needs: the WP
pretraining run wants everything here except the NI basin filter.
"""

import pandas as pd

FIRST_SEASON = 1980
LAST_SEASON = 2024
SYNOPTIC_HOURS = (0, 6, 12, 18)

# IBTrACS's derived columns STORM_SPEED, STORM_DIR and DIST2LAND are left out on
# purpose: they come from the combined LAT/LON, not JTWC positions, and the first
# two also use a future position. See "Model input rules" in CLAUDE.md.
KEEP_COLUMNS = [
    "SID",
    "NAME",
    "SEASON",
    "ISO_TIME",
    "USA_LAT",
    "USA_LON",
    "USA_WIND",
    "USA_PRES",
]


def main_track(df: pd.DataFrame) -> pd.DataFrame:
    """Keep main tracks, dropping provisional and spur tracks."""
    return df[df["TRACK_TYPE"] == "main"]


def ni_basin(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the part of each storm inside the North Indian Ocean.

    A storm that drifted in from the West Pacific keeps its NI portion only.
    """
    return df[df["BASIN"] == "NI"]


def synoptic_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the 00, 06, 12 and 18 UTC rows, the standard forecast times.

    The minute check is part of the definition in CLAUDE.md, but it removes nothing
    from today's files: the NI file's only off-the-hour rows are at 03:30 and 11:30,
    which the hour test already drops. It stays as a guard against future downloads.
    """
    is_synoptic_hour = df["ISO_TIME"].dt.hour.isin(SYNOPTIC_HOURS)
    is_on_the_hour = df["ISO_TIME"].dt.minute == 0
    return df[is_synoptic_hour & is_on_the_hour]


def seasons(
    df: pd.DataFrame,
    first: int = FIRST_SEASON,
    last: int = LAST_SEASON,
) -> pd.DataFrame:
    """Keep seasons from first to last, both included."""
    return df[df["SEASON"].between(first, last)]


def original_jtwc_fix(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where JTWC actually reported the position.

    USA_LAT being filled isn't enough: "P" and "I" rows have USA_LAT filled with
    interpolated values. Character 1 of IFLAG says which kind a row is, and "O"
    means an original fix.
    """
    has_position = df["USA_LAT"].notna()
    is_original = df["IFLAG"].str[0] == "O"
    return df[has_position & is_original]


# The order is part of the project's definition of the dataset (see CLAUDE.md).
FILTERS = [main_track, ni_basin, synoptic_hours, seasons, original_jtwc_fix]


def _print_counts(label: str, df: pd.DataFrame) -> None:
    print(f"{label:<20} {len(df):>7,} rows {df['SID'].nunique():>6,} storms")


def clean_ni(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the five filters in order and keep only KEEP_COLUMNS.

    Prints the row and storm count after each step, so if the final numbers ever
    change you can see which filter changed them.

    Args:
        df: the raw table from ``io.load_ibtracs``.

    Returns:
        The cleaned table, renumbered from 0. Rows stay in file order, which is
        by storm and then by time.
    """
    _print_counts("raw", df)
    for apply_filter in FILTERS:
        df = apply_filter(df)
        _print_counts(apply_filter.__name__, df)
    return df[KEEP_COLUMNS].reset_index(drop=True)
