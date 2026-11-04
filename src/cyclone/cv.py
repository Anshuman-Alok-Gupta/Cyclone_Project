"""Run any model through cross-validation, and measure how far to trust the result.

For each block of seasons, a model is fitted on the other four and forecasts the block
it never saw. Putting those forecasts together gives "out-of-fold" forecasts: every
development start time has one, made by a model that never saw that storm. They're in
the scoring format, so they go straight into scoring.py.

Any model can run here, as long as it comes as one function:

    forecaster(learning, learning_truth, held_out) -> forecasts

The rest of this file measures uncertainty. A skill score is one number from one set of
storms, and a different set would have given a different number. These functions show
how different, and all of them work from the table scoring.score returns.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from cyclone.folds import FOLD_NAMES, add_fold, development
from cyclone.scoring import CASE_COLUMNS, homogeneous_cases

# Enough that repeating the bootstrap with a new seed moves the interval by much less
# than the precision it's reported at.
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
INTERVAL_PERCENTILES = (2.5, 97.5)

Forecaster = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame]


def model_rows(
    init_times: pd.DataFrame,
    features: pd.DataFrame,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return one table with everything any model reads, for development storms only.

    CLIPER and persistence read the past positions in init_times, and XGBoost reads the
    columns in features. Holding both in one table lets the same rows go to every model,
    which is what makes their scores comparable.

    Test storms are dropped here, so they never enter the cross-validation at all.

    Args:
        init_times: the table from ``samples.make_init_times``.
        features: the table from ``features.build_features``.
        df: the cleaned table, which says which block each storm is in.

    Returns:
        One row per development start time, with a "fold" column.
    """
    keys = ["SID", "init_time"]
    if not features[keys].reset_index(drop=True).equals(init_times[keys].reset_index(drop=True)):
        raise ValueError("init_times and features must hold the same rows in the same order.")

    extra = features.drop(columns=[column for column in init_times if column in features.columns])
    table = pd.concat(
        [init_times.reset_index(drop=True), extra.reset_index(drop=True)], axis=1
    )
    return development(add_fold(table, df)).reset_index(drop=True)


def out_of_fold(rows: pd.DataFrame, truth: pd.DataFrame, forecaster: Forecaster) -> pd.DataFrame:
    """Return forecasts for every development start time, each from a model that never saw it.

    The held-out block's storms are removed from the truth table before the model gets
    it. So a model can't learn from a block's answers, whatever its code does, because
    they aren't in what it's handed. The rows it forecasts hold nothing after t0.

    Rows outside the blocks are left out, so a test storm can't be learned from or
    forecast even if a caller passes the full table.

    Args:
        rows: the table from ``model_rows``.
        truth: the table from ``samples.make_truth``.
        forecaster: fits on the first two arguments and forecasts the third.

    Returns:
        The forecasts from all five blocks, with a "fold" column added, ready for
        ``scoring.score``.
    """
    rows = development(rows)

    forecasts = []
    for fold in FOLD_NAMES:
        held_out = rows[rows["fold"] == fold]
        learning = rows[rows["fold"] != fold]
        learning_truth = truth[truth["SID"].isin(learning["SID"])]
        forecasts.append(forecaster(learning, learning_truth, held_out).assign(fold=fold))
    return pd.concat(forecasts, ignore_index=True)


def bootstrap_skill(
    scored: pd.DataFrame,
    reference: str = "cliper",
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Return each model's skill against a reference model, with a 95% interval.

    The bootstrap asks: if the storms had been a different draw from the same kind of
    storms, how much would the skill move? It redraws the storms many times, with
    replacement, so some are counted twice and some not at all, and works out the skill
    each time. The middle 95% of those results is the interval.

    It redraws whole storms, not rows. Forecasts from one storm, six hours apart, are
    nearly the same forecast of nearly the same situation. So 438 cases from 26 storms
    is much closer to 26 pieces of evidence than to 438, and redrawing rows would treat
    them as 438, giving an interval that looks far more certain than it is.

    Each lead time redraws from the storms with a case at that lead. Within a lead, every
    model is scored on the same redraws, so what's measured is the gap between models,
    not how hard the storms were.

    The interval covers only the luck of which storms occurred. It doesn't cover the
    choice of settings, or storms in one season being alike.

    Args:
        scored: the table from ``scoring.score``, holding the reference model.
        reference: the model to compare against, as named in the model column.
        n_resamples: how many times to redraw the storms.
        seed: fixes the redraws, so a run can be repeated.

    Returns:
        One row per model and lead time, with columns model, lead_h, skill, low, high,
        storms and cases. skill, low and high are percentages. Positive is better.
    """
    cases = homogeneous_cases(scored)
    rng = np.random.default_rng(seed)

    rows = []
    for lead_h, at_lead in cases.groupby("lead_h"):
        # A mean error is total error over case count, and each model has the same
        # cases, so skill can be worked out from each storm's total error.
        totals = at_lead.pivot_table(
            index="SID", columns="model", values="error_km", aggfunc="sum"
        )
        n_storms = len(totals)
        # One row per redraw, holding how many times each storm was drawn.
        draws = rng.multinomial(n_storms, np.full(n_storms, 1 / n_storms), size=n_resamples)
        reference_km = draws @ totals[reference].to_numpy()

        for model in totals.columns.drop(reference):
            resampled = 100 * (1 - (draws @ totals[model].to_numpy()) / reference_km)
            low, high = np.percentile(resampled, INTERVAL_PERCENTILES)
            rows.append(
                {
                    "model": model,
                    "lead_h": lead_h,
                    "skill": 100 * (1 - totals[model].sum() / totals[reference].sum()),
                    "low": low,
                    "high": high,
                    "storms": n_storms,
                    "cases": len(at_lead) // cases["model"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def skill_by_fold(scored: pd.DataFrame, reference: str = "cliper") -> pd.DataFrame:
    """Return each model's skill against a reference model, block by block.

    A model that wins overall on the strength of one block is less trustworthy than one
    that wins in all five, and an average hides the difference.

    Args:
        scored: the table from ``scoring.score``, with a "fold" column.
        reference: the model to compare against.

    Returns:
        One row per lead time and block, and one column per model other than the
        reference, as percentages.
    """
    errors = homogeneous_cases(scored)
    mean_km = errors.groupby(["lead_h", "fold", "model"])["error_km"].mean().unstack("model")
    return 100 * (1 - mean_km.drop(columns=reference).div(mean_km[reference], axis=0))


def win_rate(scored: pd.DataFrame, reference: str = "cliper") -> pd.DataFrame:
    """Return the percentage of cases where each model is closer than a reference model.

    Skill can come from a few large wins. The win rate shows how often a model is better,
    which skill can't. A case isn't independent of its neighbours, so this describes the
    forecasts and isn't a test of them.

    Args:
        scored: the table from ``scoring.score``.
        reference: the model to compare against.

    Returns:
        One row per lead time and one column per model other than the reference.
    """
    errors = homogeneous_cases(scored).pivot(index=CASE_COLUMNS, columns="model", values="error_km")
    wins = errors.drop(columns=reference).lt(errors[reference], axis=0)
    return 100 * wins.groupby(level="lead_h").mean()


def selection_scores(
    scored: pd.DataFrame,
    candidates: list[str],
    reference: str = "cliper",
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Return the score that picks the primary model, with a 95% interval, best first.

    A candidate's score is its mean skill against a reference model across the lead
    times, each lead counting equally. The rule itself, and why, is in CLAUDE.md. The
    interval is for information only: the highest score wins, whatever the intervals say.

    The interval can't be worked out from ``bootstrap_skill``, which redraws each lead
    separately. Here one redraw of the storms applies to all the leads at once, because
    a storm that's kind to a model at +24 h is the same storm at +72 h. Redrawing them
    separately would break that link. A storm with no case at a lead adds nothing there.

    Args:
        scored: the table from ``scoring.score``, holding the reference and candidates.
        candidates: the models to rank, simplest first. An exact tie keeps this order,
            so the simpler model wins it.
        reference: the model to compare against, as named in the model column.
        n_resamples: how many times to redraw the storms.
        seed: fixes the redraws, so a run can be repeated.

    Returns:
        One row per candidate, best first, with columns model, score, low and high, as
        percentages. Positive is better.
    """
    cases = homogeneous_cases(scored)
    # Each storm's total error, as in bootstrap_skill. Skill can be worked out from these
    # because every model has the same cases.
    totals = cases.pivot_table(
        index="SID", columns=["lead_h", "model"], values="error_km", aggfunc="sum", fill_value=0.0
    )
    leads = totals.columns.get_level_values("lead_h").unique()

    n_storms = len(totals)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n_storms, np.full(n_storms, 1 / n_storms), size=n_resamples)

    def mean_skill(weights: np.ndarray, model: str) -> np.ndarray:
        """Return the score under one set of storm weights, or one per row of weights."""
        ratios = [
            (weights @ totals[lead, model].to_numpy()) / (weights @ totals[lead, reference].to_numpy())
            for lead in leads
        ]
        return 100 * (1 - np.mean(ratios, axis=0))

    rows = []
    for model in candidates:
        low, high = np.percentile(mean_skill(draws, model), INTERVAL_PERCENTILES)
        rows.append(
            {"model": model, "score": mean_skill(np.ones(n_storms), model), "low": low, "high": high}
        )

    # A stable sort, so that an exact tie stays in the order given.
    table = pd.DataFrame(rows).sort_values("score", ascending=False, kind="stable")
    return table.reset_index(drop=True)


def storm_gaps(
    scored: pd.DataFrame,
    model: str,
    reference: str = "cliper",
    lead_h: int = 72,
) -> pd.DataFrame:
    """Return, storm by storm, how a model's mean error compares with a reference model's.

    Args:
        scored: the table from ``scoring.score``.
        model: the model to look at.
        reference: the model to compare against.
        lead_h: the lead time.

    Returns:
        One row per storm, sorted, with columns model_km, reference_km, gap_km and cases.
        A negative gap_km means the model was better on that storm.
    """
    errors = homogeneous_cases(scored)
    at_lead = errors[errors["lead_h"] == lead_h]
    mean_km = at_lead.pivot_table(index="SID", columns="model", values="error_km", aggfunc="mean")
    table = pd.DataFrame(
        {
            "model_km": mean_km[model],
            "reference_km": mean_km[reference],
            "gap_km": mean_km[model] - mean_km[reference],
            "cases": at_lead[at_lead["model"] == model].groupby("SID").size(),
        }
    )
    return table.sort_values("gap_km")
