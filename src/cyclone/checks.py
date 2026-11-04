"""Stop the pipeline if the cleaned data isn't the dataset CLAUDE.md describes.

Every model and score in this project depends on this table. A small silent change
(a bad filter, a new IBTrACS download) would still train and still produce numbers,
just wrong ones. So the expected counts are written down here and checked every run.
"""

import pandas as pd

from cyclone.features import distance_to_land_km
from cyclone.filters import FIRST_SEASON, LAST_SEASON, SYNOPTIC_HOURS, main_track
from cyclone.folds import FOLD_NAMES, NOT_IN_A_FOLD, add_fold, assign_fold, development
from cyclone.samples import LEAD_HOURS, with_truth_at
from cyclone.scoring import great_circle_km
from cyclone.splits import VAL_LAST_SEASON

# Verified against IBTrACS v04r01 on 2026-11-04.
EXPECTED_ROWS = 5_694
EXPECTED_STORMS = 258
EXPECTED_STORMS_BY_SPLIT = {"train": 184, "val": 27, "test": 47}
EXPECTED_INIT_TIMES_BY_SPLIT = {"train": 3_203, "val": 542, "test": 933}
# Start times with a true position at each lead time: {split: {lead_h: count}}.
EXPECTED_CASES_BY_SPLIT = {
    "train": {24: 2_515, 48: 1_869, 72: 1_324},
    "val": {24: 438, 48: 340, 72: 256},
    "test": {24: 749, 48: 578, 72: 430},
}

# One number per block, in the order of folds.FOLD_SEASONS, so that this file doesn't repeat
# the boundaries. Verified on 2026-11-04. They add up to the train and val counts above.
EXPECTED_STORMS_BY_FOLD = (39, 44, 48, 43, 37)
EXPECTED_INIT_TIMES_BY_FOLD = (598, 774, 952, 737, 684)
EXPECTED_CASES_BY_FOLD = {
    24: (452, 617, 774, 570, 540),
    48: (324, 465, 605, 413, 402),
    72: (218, 325, 463, 284, 290),
}

# DIST2LAND follows the combined position, so it's only comparable where JTWC's position
# is close to that one.
DIST2LAND_MAX_SEPARATION_KM = 10
# Far above today's 5 km. A gap beyond this means a broken calculation, such as swapped
# coordinates or wrong units, not the known difference between the two land masks.
DIST2LAND_MAX_MEDIAN_GAP_KM = 10


def _find_likely_causes(df: pd.DataFrame) -> list[str]:
    """Look in the table for the usual reasons the counts drift."""
    causes = []

    outside = ~df["SEASON"].between(FIRST_SEASON, LAST_SEASON)
    if outside.any():
        causes.append(f"{outside.sum()} rows are outside seasons {FIRST_SEASON}-{LAST_SEASON}.")

    off_hours = ~df["ISO_TIME"].dt.hour.isin(SYNOPTIC_HOURS) | (df["ISO_TIME"].dt.minute != 0)
    if off_hours.any():
        causes.append(f"{off_hours.sum()} rows are not at a synoptic time.")

    no_position = df["USA_LAT"].isna() | df["USA_LON"].isna()
    if no_position.any():
        causes.append(f"{no_position.sum()} rows have no JTWC position.")

    duplicates = df.duplicated(["SID", "ISO_TIME"])
    if duplicates.any():
        causes.append(f"{duplicates.sum()} rows repeat a storm and time already seen.")

    splits_per_storm = df.groupby("SID")["split"].nunique()
    if (splits_per_storm > 1).any():
        causes.append(f"{(splits_per_storm > 1).sum()} storms appear in more than one split.")

    if not causes:
        causes.append(
            "The table looks valid, so the source data may have changed. Check whether "
            "the IBTrACS file was re-downloaded (storms are revised between downloads), "
            "then compare the counts clean_ni printed with the notebook's saved output "
            "to see which filter differs. If only the split counts differ, check the "
            "boundaries in splits.py."
        )
    return causes


def check_expected(df: pd.DataFrame) -> None:
    """Raise an error explaining what differs if the counts aren't exactly right.

    The check runs in two parts because each number can hide a different problem.
    The row count catches lost points within a storm, and the storm counts catch
    whole storms moving. The three split counts add up to the 258 total, so if they
    all pass, no storm can be sitting in two splits.

    Args:
        df: the table after ``filters.clean_ni`` and ``splits.assign_split``.

    Raises:
        AssertionError: listing every count that differs and the likely causes.
        This uses ``raise`` rather than the ``assert`` keyword because Python skips
        ``assert`` lines when run with -O, and this check must always run.
    """
    storms_by_split = df.groupby("split")["SID"].nunique().to_dict()

    mismatches = []
    if len(df) != EXPECTED_ROWS:
        mismatches.append(f"rows: expected {EXPECTED_ROWS:,}, got {len(df):,}")
    if df["SID"].nunique() != EXPECTED_STORMS:
        mismatches.append(f"storms: expected {EXPECTED_STORMS}, got {df['SID'].nunique()}")
    if storms_by_split != EXPECTED_STORMS_BY_SPLIT:
        mismatches.append(
            f"storms by split: expected {EXPECTED_STORMS_BY_SPLIT}, got {storms_by_split}"
        )

    if mismatches:
        lines = ["Cleaned data does not match CLAUDE.md. Stop and resolve this first."]
        lines += ["Differences:"] + [f"  - {item}" for item in mismatches]
        lines += ["Likely causes:"] + [f"  - {item}" for item in _find_likely_causes(df)]
        raise AssertionError("\n".join(lines))

    print(
        f"Counts match: {len(df):,} rows, {df['SID'].nunique()} storms, "
        f"{storms_by_split['train']}/{storms_by_split['val']}/{storms_by_split['test']} "
        "train/val/test."
    )


def count_cases(init_times: pd.DataFrame, truth: pd.DataFrame) -> dict[str, dict[int, int]]:
    """Count the start times with a true position, by split and lead time."""
    counts: dict[str, dict[int, int]] = {split: {} for split in EXPECTED_CASES_BY_SPLIT}
    for lead_h in LEAD_HOURS:
        cases = with_truth_at(init_times, truth, lead_h)
        for split in counts:
            counts[split][lead_h] = int((cases["split"] == split).sum())
    return counts


def check_samples(init_times: pd.DataFrame, truth: pd.DataFrame) -> None:
    """Raise an error explaining what differs if the sample counts aren't exact.

    Each lead time has fewer cases than the one before, because storms end. A count
    that drifts here usually means the timestamp matching changed, which would still
    run and would quietly give every model a different set of cases.

    This counts test cases but never scores them, so the test split stays locked.

    Args:
        init_times: the table from ``samples.make_init_times``.
        truth: the table from ``samples.make_truth``.

    Raises:
        AssertionError: listing every count that differs.
    """
    init_counts = init_times["split"].value_counts().to_dict()
    init_counts = {split: init_counts.get(split, 0) for split in EXPECTED_INIT_TIMES_BY_SPLIT}
    case_counts = count_cases(init_times, truth)

    mismatches = []
    if init_counts != EXPECTED_INIT_TIMES_BY_SPLIT:
        mismatches.append(
            f"init times: expected {EXPECTED_INIT_TIMES_BY_SPLIT}, got {init_counts}"
        )
    for split, expected in EXPECTED_CASES_BY_SPLIT.items():
        if case_counts[split] != expected:
            mismatches.append(f"{split} cases: expected {expected}, got {case_counts[split]}")

    if mismatches:
        lines = ["Forecast samples do not match the verified counts. Stop and resolve this first."]
        lines += ["Differences:"] + [f"  - {item}" for item in mismatches]
        raise AssertionError("\n".join(lines))

    print(f"Sample counts match: {len(init_times):,} init times.")
    for split, counts in case_counts.items():
        by_lead = ", ".join(f"+{lead_h} h {count:,}" for lead_h, count in counts.items())
        print(f"  {split:<5} {init_counts[split]:>5,} init times; cases {by_lead}")


def _counts_by_fold(table: pd.DataFrame, count) -> tuple[int, ...]:
    """Return one count per block, in FOLD_NAMES order, with 0 for an empty block."""
    counts = development(table).groupby("fold").apply(count, include_groups=False)
    return tuple(int(counts.get(name, 0)) for name in FOLD_NAMES)


def check_folds(df: pd.DataFrame, init_times: pd.DataFrame, truth: pd.DataFrame) -> None:
    """Raise an error explaining what differs if the blocks aren't the verified ones.

    A block that quietly lost storms would still fit and still score. It would just
    change every cross-validated number, with no sign of why.

    Args:
        df: the cleaned table.
        init_times: the table from ``samples.make_init_times`` or ``features.build_features``.
        truth: the table from ``samples.make_truth``.

    Raises:
        AssertionError: listing every count that differs, or any season in the wrong place.
    """
    folded = assign_fold(df)
    rows = add_fold(init_times, df)

    storms = _counts_by_fold(folded, lambda block: block["SID"].nunique())
    starts = _counts_by_fold(rows, len)
    cases = {
        lead_h: _counts_by_fold(with_truth_at(rows, truth, lead_h), len) for lead_h in LEAD_HOURS
    }

    mismatches = []
    if storms != EXPECTED_STORMS_BY_FOLD:
        mismatches.append(f"storms: expected {EXPECTED_STORMS_BY_FOLD}, got {storms}")
    if starts != EXPECTED_INIT_TIMES_BY_FOLD:
        mismatches.append(f"init times: expected {EXPECTED_INIT_TIMES_BY_FOLD}, got {starts}")
    if cases != EXPECTED_CASES_BY_FOLD:
        mismatches.append(f"cases: expected {EXPECTED_CASES_BY_FOLD}, got {cases}")

    in_a_block = folded[folded["fold"] != NOT_IN_A_FOLD]
    if in_a_block["SEASON"].max() > VAL_LAST_SEASON:
        mismatches.append(f"a season after {VAL_LAST_SEASON} is in a block: test must stay locked")
    if (in_a_block.groupby("SID")["fold"].nunique() > 1).any():
        mismatches.append("a storm is in two blocks")

    if mismatches:
        lines = ["Cross-validation blocks do not match the verified counts. Stop and resolve this."]
        lines += ["Differences:"] + [f"  - {item}" for item in mismatches]
        raise AssertionError("\n".join(lines))

    print(f"Fold counts match: {sum(storms)} storms, {sum(starts):,} init times.")
    for number, name in enumerate(FOLD_NAMES):
        by_lead = ", ".join(f"+{lead_h} h {cases[lead_h][number]:,}" for lead_h in LEAD_HOURS)
        print(f"  {name}  {storms[number]:>3} storms, {starts[number]:>4} init times; cases {by_lead}")


def check_dist_to_land(df: pd.DataFrame, raw: pd.DataFrame) -> None:
    """Compare recomputed distance to land with IBTrACS DIST2LAND, and report it.

    This is a sanity check only. DIST2LAND is read from the raw table here, stays inside
    this function, and never becomes a model input. The cleaned table doesn't even hold
    it, which is why the raw table is needed.

    Expect close agreement over land and near large coasts, and a long tail elsewhere.
    The tail isn't an error: the land mask behind DIST2LAND leaves out islands that
    Natural Earth has, such as the Maldives. docs/components.md has the evidence.

    Args:
        df: the cleaned table.
        raw: the raw table from ``io.load_ibtracs``.

    Raises:
        AssertionError: if the median gap shows the calculation itself is broken.
    """
    combined = main_track(raw)[["SID", "ISO_TIME", "LAT", "LON", "DIST2LAND"]]
    rows = df.merge(combined, on=["SID", "ISO_TIME"], how="left", validate="1:1")
    separation_km = great_circle_km(rows["USA_LAT"], rows["USA_LON"], rows["LAT"], rows["LON"])
    rows = rows[separation_km <= DIST2LAND_MAX_SEPARATION_KM]

    ours = distance_to_land_km(rows["USA_LAT"], rows["USA_LON"])
    theirs = rows["DIST2LAND"].to_numpy()
    gap = abs(ours - theirs)
    median_gap = float(pd.Series(gap).median())

    print(
        f"Distance to land against DIST2LAND, on {len(rows):,} of {len(df):,} rows with the "
        f"JTWC and combined positions within {DIST2LAND_MAX_SEPARATION_KM} km:"
    )
    print(f"  median gap {median_gap:.1f} km, 90th percentile {pd.Series(gap).quantile(0.9):.0f} km")
    print(f"  within 10 km: {(gap <= 10).mean():.1%}, within 25 km: {(gap <= 25).mean():.1%}")
    print(f"  over land: {((ours == 0) & (theirs == 0)).sum()} rows in both, of {(ours == 0).sum()} "
          f"here and {(theirs == 0).sum()} in IBTrACS")

    if median_gap > DIST2LAND_MAX_MEDIAN_GAP_KM:
        raise AssertionError(
            f"Median gap to DIST2LAND is {median_gap:.1f} km, above the "
            f"{DIST2LAND_MAX_MEDIAN_GAP_KM} km limit. Check distance_to_land_km."
        )
