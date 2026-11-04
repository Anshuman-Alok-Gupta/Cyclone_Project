"""Stop the pipeline if the cleaned data isn't the dataset CLAUDE.md describes.

Every model and score in this project depends on this table. A small silent change
(a bad filter, a new IBTrACS download) would still train and still produce numbers,
just wrong ones. So the expected counts are written down here and checked every run.
"""

import pandas as pd

from cyclone.filters import FIRST_SEASON, LAST_SEASON, SYNOPTIC_HOURS
from cyclone.samples import LEAD_HOURS, with_truth_at

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
