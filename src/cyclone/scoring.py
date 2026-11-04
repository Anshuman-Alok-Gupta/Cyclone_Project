"""Score any table of forecast positions against reference tracks.

This file deliberately imports nothing else from the cyclone package. Project 3 will
score GraphCast and Pangu tracks with it, and those models know nothing about this
project's tables. Anything that can produce the columns below can be scored, and the
persistence baseline and an AI model are then measured by exactly the same ruler.

    forecasts: model, SID, init_time, lead_h, lat, lon
    truth:     SID, valid_time, lat, lon
"""

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0

FORECAST_COLUMNS = ["model", "SID", "init_time", "lead_h", "lat", "lon"]
TRUTH_COLUMNS = ["SID", "valid_time", "lat", "lon"]

# One case is one storm, start time and lead time. Each model forecasts it once.
CASE_COLUMNS = ["SID", "init_time", "lead_h"]


def great_circle_km(lat1, lon1, lat2, lon2):
    """Return the shortest distance over the Earth's surface, in km (haversine formula).

    Subtracting degrees isn't enough, because a degree of longitude shrinks towards
    the poles: it's 111 km at the equator but 96 km at 30N. Treating the Earth as a
    sphere is standard for track scoring and accurate to about 0.5%.

    Args:
        lat1, lon1: first point in degrees. Numbers, arrays or pandas columns.
        lat2, lon2: second point in degrees, of the same shape.

    Returns:
        The distance in km, of the same shape as the inputs.
    """
    phi1, lam1, phi2, lam2 = map(np.radians, (lat1, lon1, lat2, lon2))
    half_chord = (
        np.sin((phi2 - phi1) / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin((lam2 - lam1) / 2) ** 2
    )
    # Rounding can push this a hair above 1 for points on opposite sides of the Earth,
    # and arcsin of that would be NaN.
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(half_chord, 0.0, 1.0)))


def _require_columns(table: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{name} is missing columns {missing}. It needs {columns}.")


def score(forecasts: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Match each forecast to the truth at init_time + lead_h and add its error.

    Forecasts with no true position stay in the result with a blank error_km, and the
    count is printed for each model. Nothing is dropped here, because a model that
    lost half its cases should be visible. They can't be counted as right or wrong,
    so ``summarize`` leaves them out of the averages. Having some is normal: the storm
    may have ended, left the basin or lost a fix by then.

    Args:
        forecasts: one row per forecast, with FORECAST_COLUMNS.
        truth: one row per storm and time, with TRUTH_COLUMNS.

    Returns:
        forecasts plus valid_time, truth_lat, truth_lon and error_km, one row for each
        forecast and in the same order.

    Raises:
        ValueError: if columns are missing, a case is repeated for a model, or a
            forecast has no position. Each of these would give a wrong score with no
            visible sign, so they stop the run.
    """
    _require_columns(forecasts, FORECAST_COLUMNS, "forecasts")
    _require_columns(truth, TRUTH_COLUMNS, "truth")

    repeated = forecasts.duplicated(["model", *CASE_COLUMNS])
    if repeated.any():
        raise ValueError(f"{repeated.sum()} forecasts repeat a model and case already seen.")

    no_position = forecasts["lat"].isna() | forecasts["lon"].isna()
    if no_position.any():
        raise ValueError(
            f"{no_position.sum()} forecasts have no position. Remove them first and "
            "report them, for example as a storm the model's tracker lost."
        )

    reference = truth[TRUTH_COLUMNS].rename(columns={"lat": "truth_lat", "lon": "truth_lon"})
    reference["valid_time"] = pd.to_datetime(reference["valid_time"])

    # to_datetime lets a model's output carry its times as text. validate="m:1" stops
    # the run if truth repeats a time, which would silently copy forecast rows.
    init_time = pd.to_datetime(forecasts["init_time"])
    valid_time = init_time + pd.to_timedelta(forecasts["lead_h"], unit="h")
    scored = forecasts.assign(init_time=init_time, valid_time=valid_time).merge(
        reference, on=["SID", "valid_time"], how="left", validate="m:1"
    )
    scored["error_km"] = great_circle_km(
        scored["lat"], scored["lon"], scored["truth_lat"], scored["truth_lon"]
    )

    no_truth = scored["error_km"].isna().groupby(scored["model"]).sum()
    total = scored.groupby("model").size()
    for model in total.index:
        print(
            f"{model}: scored {total[model] - no_truth[model]:,} of {total[model]:,} "
            f"forecasts; {no_truth[model]:,} had no matching truth"
        )
    return scored


def summarize(scored: pd.DataFrame, homogeneous: bool = True) -> pd.DataFrame:
    """Return count, mean and median error by model and lead time.

    Comparing models on different cases isn't fair: easy storms in one model's average
    and hard ones in another's can make the worse model look better. With
    homogeneous=True, only cases that every model has are kept, so each row of the
    result covers identical storms and times. Leave it on when comparing models.

    Args:
        scored: the table from ``score``.
        homogeneous: keep only the cases that every model in the table scored.

    Returns:
        One row per model and lead time, with columns model, lead_h, count,
        mean_km and median_km.
    """
    usable = scored.dropna(subset=["error_km"])

    if homogeneous:
        # Count models from scored, not usable: a model with no scored cases at all
        # must still empty the table, not quietly stop being one of "every model".
        n_models = scored["model"].nunique()
        models_in_case = usable.groupby(CASE_COLUMNS)["model"].transform("nunique")
        kept = usable[models_in_case == n_models]
        print(
            f"Homogeneous comparison: kept {len(kept):,} of {len(usable):,} scored "
            f"forecasts, the cases all {n_models} models have."
        )
        usable = kept

    summary = usable.groupby(["model", "lead_h"])["error_km"].agg(
        count="count", mean_km="mean", median_km="median"
    )
    return summary.reset_index()


def skill(summary: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Return each model's percentage improvement in mean error over a reference model.

    Positive means better than the reference, and negative means worse. 10 means the
    model's mean error is 10% smaller. It's only fair on identical cases, so pass a
    table from ``summarize`` with homogeneous=True.

    Args:
        summary: the table from ``summarize``.
        reference: the model to compare against, as named in the model column.

    Returns:
        One row per lead time and one column per model other than the reference.
    """
    mean_km = summary.pivot(index="lead_h", columns="model", values="mean_km")
    return 100 * (1 - mean_km.drop(columns=reference).div(mean_km[reference], axis=0))
