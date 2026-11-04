"""Reading and writing tables. No filtering happens here.

``load_ibtracs`` returns every row of the file, including other basins and old
storms. Keeping the choice of rows in filters.py means the WP file can go through
this same loader later, and ``load_ibtracs`` never has to change.
"""

from pathlib import Path

import pandas as pd

# The IBTrACS columns that hold text. Everything else must be a number.
# Listing them, rather than letting pandas guess, matters in two ways:
#   1. A guess depends on the file. USA_RECORD is text in the NI file but entirely
#      empty in the WP file, so pandas would make it text in one and a float in
#      the other. The basin tables couldn't then be combined cleanly.
#   2. If a new column or a stray value appears, the numeric conversion below
#      fails loudly instead of the column quietly staying as text.
TEXT_COLUMNS = (
    "SID",
    "BASIN",
    "SUBBASIN",
    "NAME",
    "ISO_TIME",  # read as text, then parsed into datetimes below
    "NATURE",
    "WMO_AGENCY",
    "TRACK_TYPE",
    "IFLAG",
    "USA_AGENCY",
    "USA_ATCF_ID",
    "USA_RECORD",
    "USA_STATUS",
    "HKO_CAT",
    "KMA_CAT",
    "NEWDELHI_GRADE",
    "DS824_STAGE",
)


def load_ibtracs(path: str | Path) -> pd.DataFrame:
    """Load one IBTrACS CSV with correct missing values and column types.

    Args:
        path: the CSV file. Get it from ``config.ibtracs_path("NI")``.

    Returns:
        One row per track point, with all 174 columns. ISO_TIME is a datetime,
        and every column outside TEXT_COLUMNS is numeric.
    """
    df = pd.read_csv(
        path,
        skiprows=[1],  # line 2 of the file holds units ("kts", "degrees_north")
        keep_default_na=False,  # otherwise BASIN "NA" (North Atlantic) becomes missing
        na_values=[" "],  # IBTrACS marks a missing value with a single space
        dtype={column: str for column in TEXT_COLUMNS},
        low_memory=False,  # read the file in one pass so types can't vary by chunk
    )

    # pd.to_numeric raises on a non-number. errors="coerce" would hide the problem
    # by turning it into NaN, which is the kind of silent loss we want to avoid.
    df = df.apply(lambda col: col if col.name in TEXT_COLUMNS else pd.to_numeric(col))

    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], format="%Y-%m-%d %H:%M:%S")
    return df


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Write a table to Parquet, creating the folder if needed.

    data/ is git-ignored, so on a fresh Colab clone or a new Drive folder
    data/processed may not exist yet.

    Parquet stores each column's type, so ISO_TIME comes back as a datetime and
    SID as text. A CSV would lose that and need all of the parsing above again.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
