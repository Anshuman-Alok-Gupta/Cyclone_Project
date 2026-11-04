"""Group the development seasons into blocks for cross-validation.

One validation split rests on whichever storms fell into it: 27 storms were too few to
decide between models. Cross-validation uses every storm for both jobs. The development
seasons are cut into five blocks, and each block in turn is held out and forecast by
models fitted on the other four. So every storm gets forecast once, by a model that
never saw it.

Each block is a run of whole, consecutive seasons, for the same reason the main split is
by year. Rows from one storm are nearly identical six hours apart, and storms in one
season share that season's weather, so a random split would let a model be tested on
storms it had nearly seen.

Blocks differ from the main split in one way: a model can learn from seasons later than
the ones it forecasts. That's fine for comparing models, because no storm's own future is
involved. It wouldn't be fine for the final exam, which is why test stays last in time.

The test seasons are in no block. They stay locked until the end of Project 1.
"""

import numpy as np
import pandas as pd

# First and last season of each block. These are the only definition of the boundaries,
# so nothing else in the project should repeat them. 38 seasons don't divide evenly by
# 5, so three blocks cover 8 seasons and two cover 7.
FOLD_SEASONS = ((1980, 1987), (1988, 1995), (1996, 2003), (2004, 2010), (2011, 2017))

FOLD_NAMES = tuple(f"{first}-{last}" for first, last in FOLD_SEASONS)

# The label for anything outside the development seasons, which means the test seasons
# and the excluded ones after them.
NOT_IN_A_FOLD = "none"


def assign_fold(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a "fold" column added.

    It uses SEASON, not the calendar year of ISO_TIME, so a storm that runs across New
    Year stays in one block.

    Args:
        df: any table with a SEASON column.

    Returns:
        df plus "fold", which holds one of FOLD_NAMES, or NOT_IN_A_FOLD.
    """
    in_block = [df["SEASON"].between(first, last) for first, last in FOLD_SEASONS]
    return df.assign(fold=np.select(in_block, FOLD_NAMES, default=NOT_IN_A_FOLD))


def add_fold(table: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of table with each row's fold, looked up from its storm.

    The start-time and feature tables have no SEASON column, so the cleaned table says
    which season each storm belongs to.

    Args:
        table: any table with a SID column.
        df: the cleaned table.

    Returns:
        table plus "fold". A storm in two blocks stops the run here, because looking it
        up would be ambiguous.
    """
    storms = assign_fold(df).drop_duplicates(["SID", "fold"]).set_index("SID")["fold"]
    return table.assign(fold=table["SID"].map(storms))


def development(table: pd.DataFrame) -> pd.DataFrame:
    """Return the rows that belong to a block, which are never test rows.

    It keeps named blocks rather than dropping NOT_IN_A_FOLD, so a row that somehow has
    no fold at all stays out too.
    """
    return table[table["fold"].isin(FOLD_NAMES)]
