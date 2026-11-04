"""Assign each row to train, val or test by storm season.

The split is by year, never random. Rows from one storm are nearly identical six
hours apart, so a random split would put the same storm in both train and test,
and the scores would look better than they really are. Splitting by year also
matches real use: learn from the past, forecast the future.
"""

import numpy as np
import pandas as pd

# Last season of each split. These three numbers are the only definition of the
# boundaries, so nothing else in the project should repeat them.
TRAIN_LAST_SEASON = 2012
VAL_LAST_SEASON = 2017
TEST_LAST_SEASON = 2024


def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a "split" column added.

    Values are "train", "val" and "test". Seasons after TEST_LAST_SEASON get
    "excluded", because 2025 onward is provisional data that shouldn't be used.

    SEASON is used instead of the calendar year of ISO_TIME because a storm that
    runs across New Year has one SEASON but two calendar years, and it must not
    end up in two splits.
    """
    season = df["SEASON"]
    # np.select takes the first condition that is true, so each condition only
    # needs the upper limit.
    split = np.select(
        [
            season <= TRAIN_LAST_SEASON,
            season <= VAL_LAST_SEASON,
            season <= TEST_LAST_SEASON,
        ],
        ["train", "val", "test"],
        default="excluded",
    )
    return df.assign(split=split)
