"""The code that scores the final models on the test split, written before it's needed.

Test is scored once, so test day should only run code that has already been tested on
other storms. A bug that first shows up on test has no good way out. Once results have been
seen, fixing the code and scoring again is a second look at the exam, and leaving the bug
in means reporting a wrong number. Worse, the wish to go looking for a bug depends on
whether the number was welcome, so a disappointing result gets checked and a pleasing
one doesn't. Rehearsing all of this on the development seasons finds those bugs while
finding them costs nothing.

The rehearsal's scores mean nothing, because the final models have seen those storms. It
checks that the code runs and gives the right rows, not how good the models are.

The test seasons are guarded, so a slip can't use up the one look. Scoring them needs
``confirm_test=True``, and the guard runs before anything is read.
"""

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from cyclone import cv, final_model
from cyclone.features import build_features
from cyclone.folds import add_fold
from cyclone.samples import make_init_times, make_truth
from cyclone.scoring import score
from cyclone.sequences import build_sequences
from cyclone.splits import TEST_LAST_SEASON, VAL_LAST_SEASON

TEST_SEASONS = frozenset(range(VAL_LAST_SEASON + 1, TEST_LAST_SEASON + 1))


def check_seasons(seasons: Iterable[int], confirm_test: bool = False) -> frozenset[int]:
    """Stop the run if seasons includes a test season and that wasn't confirmed.

    One test season among others counts, since that would still be a look at it. Only
    ``True`` itself confirms, so that a value passed by mistake doesn't.

    Args:
        seasons: the seasons to score.
        confirm_test: True only when scoring the test split, once, under the test plan.

    Returns:
        The seasons, as a set.

    Raises:
        ValueError: if a test season is included without confirmation, or seasons is empty.
    """
    chosen = frozenset(seasons)
    if not chosen:
        raise ValueError("No seasons given.")

    locked = sorted(chosen & TEST_SEASONS)
    if locked and confirm_test is not True:
        raise ValueError(
            f"Seasons {locked} are in the test split, which is scored once at the end of "
            "Project 1. Pass confirm_test=True only for that run."
        )
    return chosen


def rows_for_seasons(df: pd.DataFrame, seasons: Iterable[int]) -> pd.DataFrame:
    """Return one table with everything any model reads, for the seasons given.

    This does for any seasons what ``cv.model_rows`` does for the development seasons, so
    a model reads test rows built in exactly the same way as the rows it was measured on.

    Only the chosen seasons are read, which keeps the other seasons' storms out. The rows
    hold nothing after t0, so this doesn't look at any forecast's answer.

    Args:
        df: the cleaned table.
        seasons: the seasons to build rows for.

    Returns:
        One row per start time, with a "fold" column. Seasons outside the blocks are
        labelled ``folds.NOT_IN_A_FOLD``.

    Raises:
        ValueError: if a season isn't in the table. That includes 2025 onward, which
            cleaning removes as provisional, and an empty list of seasons.
    """
    wanted = frozenset(seasons)
    missing = sorted(wanted - set(df["SEASON"]))
    if missing or not wanted:
        raise ValueError(f"Seasons not in the cleaned table: {missing or 'none given'}.")

    chosen = df[df["SEASON"].isin(wanted)]
    init_times = make_init_times(chosen)
    table = cv.join_inputs(init_times, build_features(init_times))
    return add_fold(table, chosen).reset_index(drop=True)


def score_final_models(
    df: pd.DataFrame,
    seasons: Iterable[int],
    directory: str | Path | None = None,
    *,
    confirm_test: bool = False,
) -> pd.DataFrame:
    """Score all seven saved final models on the seasons given.

    The saved models are loaded rather than fitted here, so nothing about these seasons
    can reach them. That includes the LSTM's scalers, which are saved with it. On the
    development seasons this is a rehearsal only: those models have seen those storms, so
    the errors say nothing about how good the models are.

    ``confirm_test`` has to be named in the call, so it can't be passed by accident.

    Args:
        df: the cleaned table.
        seasons: the seasons to score.
        directory: where the models are. Leave it out to use ``config.xgboost_final_dir``.
        confirm_test: True only when scoring the test split, once, under the test plan.

    Returns:
        The table from ``scoring.score``, ready for ``scoring.summarize`` and
        ``cv.bootstrap_skill``.

    Raises:
        ValueError: if a test season is included without confirmation.
    """
    chosen = check_seasons(seasons, confirm_test)

    in_seasons = df[df["SEASON"].isin(chosen)]
    rows = rows_for_seasons(df, chosen)
    final = final_model.load_final_models(directory)
    forecasts = final_model.final_forecasts(final, rows, build_sequences(rows, in_seasons))
    return score(forecasts, make_truth(in_seasons))
