"""Tests for the cross-validation blocks.

A storm in two blocks would be tested on a model that had trained on it. A test season
in a block would unlock the test split. Either would improve the scores, and nothing
else would notice, so these rules are tested directly.

Most tests use a small made-up table, so they run anywhere. The last group runs on the
real cleaned table, and skips if it hasn't been built on this machine.
"""

import pandas as pd
import pytest

from cyclone import checks, config
from cyclone.filters import FIRST_SEASON
from cyclone.folds import (
    FOLD_NAMES,
    FOLD_SEASONS,
    NOT_IN_A_FOLD,
    add_fold,
    assign_fold,
    development,
)
from cyclone.samples import make_init_times, make_truth
from cyclone.splits import TEST_LAST_SEASON, VAL_LAST_SEASON, assign_split


def make_table() -> pd.DataFrame:
    """Return two rows for one made-up storm in every season, 1980 to 2026."""
    seasons = range(FIRST_SEASON, TEST_LAST_SEASON + 3)
    rows = [
        {"SID": f"STORM{season}", "SEASON": season, "ISO_TIME": pd.Timestamp(f"{season}-05-01")}
        for season in seasons
    ]
    return pd.DataFrame(rows + rows)


def test_blocks_are_whole_consecutive_seasons_with_no_gap_or_overlap():
    for (_, last), (next_first, _) in zip(FOLD_SEASONS, FOLD_SEASONS[1:]):
        assert next_first == last + 1
    assert all(first <= last for first, last in FOLD_SEASONS)


def test_blocks_run_from_the_first_season_to_the_last_development_season():
    assert FOLD_SEASONS[0][0] == FIRST_SEASON
    assert FOLD_SEASONS[-1][1] == VAL_LAST_SEASON


def test_no_season_after_development_is_in_a_block():
    table = assign_fold(make_table())
    later = table[table["SEASON"] > VAL_LAST_SEASON]
    assert len(later) > 0
    assert (later["fold"] == NOT_IN_A_FOLD).all()
    assert development(table)["SEASON"].max() == VAL_LAST_SEASON


def test_every_development_season_is_in_exactly_one_block():
    table = assign_fold(make_table())
    seasons = table[table["SEASON"] <= VAL_LAST_SEASON]
    assert seasons["fold"].isin(FOLD_NAMES).all()
    assert seasons.groupby("SEASON")["fold"].nunique().eq(1).all()


def test_each_block_boundary_falls_between_the_right_seasons():
    table = assign_fold(make_table()).drop_duplicates("SEASON").set_index("SEASON")["fold"]
    for name, (first, last) in zip(FOLD_NAMES, FOLD_SEASONS):
        assert table[first] == name
        assert table[last] == name


def test_a_storm_across_new_year_stays_in_the_block_of_its_season():
    # The last season of the first block, running into January of the next block's years.
    last_season = FOLD_SEASONS[0][1]
    storm = pd.DataFrame(
        {
            "SID": "NEW_YEAR",
            "SEASON": last_season,
            "ISO_TIME": pd.to_datetime([f"{last_season}-12-31", f"{last_season + 1}-01-01"]),
        }
    )
    assert assign_fold(storm)["fold"].eq(FOLD_NAMES[0]).all()


def test_add_fold_gives_each_row_the_fold_of_its_storm():
    table = make_table()
    rows = pd.DataFrame({"SID": ["STORM1980", "STORM2000", "STORM2020"]})
    assert add_fold(rows, table)["fold"].tolist() == ["1980-1987", "1996-2003", NOT_IN_A_FOLD]


def test_a_storm_in_two_blocks_stops_the_run():
    table = pd.DataFrame({"SID": ["BOTH", "BOTH"], "SEASON": [1987, 1988]})
    with pytest.raises(Exception, match="(?i)index"):
        add_fold(pd.DataFrame({"SID": ["BOTH"]}), table)


def test_a_row_with_no_fold_is_not_development():
    rows = pd.DataFrame({"fold": [FOLD_NAMES[0], NOT_IN_A_FOLD, None]})
    assert len(development(rows)) == 1


needs_real_data = pytest.mark.skipif(
    not config.ni_clean_path().exists(), reason="the cleaned table hasn't been built here"
)


@pytest.fixture(scope="module")
def cleaned() -> pd.DataFrame:
    return pd.read_parquet(config.ni_clean_path())


@needs_real_data
def test_no_real_storm_is_in_two_blocks(cleaned):
    assert assign_fold(cleaned).groupby("SID")["fold"].nunique().eq(1).all()


@needs_real_data
def test_blocks_cover_every_real_development_storm_and_no_test_storm(cleaned):
    folded = assign_fold(cleaned)
    in_blocks = set(development(folded)["SID"])
    assert in_blocks == set(folded[folded["split"].isin(["train", "val"])]["SID"])
    assert in_blocks.isdisjoint(folded[folded["split"] == "test"]["SID"])


@needs_real_data
def test_real_fold_counts_match(cleaned):
    checks.check_folds(cleaned, make_init_times(cleaned), make_truth(cleaned))


@needs_real_data
def test_a_lost_storm_stops_the_fold_check(cleaned):
    one_storm = cleaned[cleaned["SEASON"] == FOLD_SEASONS[0][0]]["SID"].iloc[0]
    fewer = cleaned[cleaned["SID"] != one_storm]
    with pytest.raises(AssertionError, match="storms: expected"):
        checks.check_folds(fewer, make_init_times(fewer), make_truth(fewer))


@needs_real_data
def test_a_test_season_in_a_block_stops_the_fold_check(cleaned, monkeypatch):
    stretched = (*FOLD_SEASONS[:-1], (FOLD_SEASONS[-1][0], VAL_LAST_SEASON + 1))
    monkeypatch.setattr("cyclone.folds.FOLD_SEASONS", stretched)
    with pytest.raises(AssertionError, match="test must stay locked"):
        checks.check_folds(cleaned, make_init_times(cleaned), make_truth(cleaned))


def test_splits_and_folds_agree_on_where_development_ends():
    table = assign_split(assign_fold(make_table()))
    assert set(table[table["fold"] != NOT_IN_A_FOLD]["split"]) == {"train", "val"}
