"""The two simple forecasts that every later model has to beat.

A skill score means nothing alone. "XGBoost is 150 km off at 24 h" only becomes
useful next to what a method with no real intelligence achieves on the same cases.

    persistence: the storm keeps doing what it has been doing
    CLIPER:      a straight-line fit from climatology and persistence

Both return a table in the format scoring.py expects, so they're scored exactly as
XGBoost, the LSTM and the AI models will be.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from cyclone.samples import LEAD_HOURS, with_truth_at
from cyclone.scoring import FORECAST_COLUMNS

PERSISTENCE_NAME = "persistence"
CLIPER_NAME = "cliper"

# Hours of past motion persistence extends. 12 h rather than 6 h because a single
# 6 h step is short enough for small errors in each fix to swing the direction.
PERSISTENCE_WINDOW_H = 12

DAYS_PER_YEAR = 365.25


def _forecast_table(
    model: str,
    init_times: pd.DataFrame,
    lead_h: int,
    lat: pd.Series,
    lon: pd.Series,
) -> pd.DataFrame:
    """Put one model's positions for one lead time into the scoring format."""
    table = init_times[["SID", "init_time"]].assign(model=model, lead_h=lead_h, lat=lat, lon=lon)
    return table[FORECAST_COLUMNS]


def persistence(init_times: pd.DataFrame) -> pd.DataFrame:
    """Extend the average motion of the last 12 h in a straight line.

    A storm that moved 2 degrees north in the past 12 h is forecast to move 4 degrees
    north in the next 24 h, and 12 degrees north in the next 72 h. It works well at
    short range and gets worse as storms turn, because it can never predict a turn.

    The line is straight in latitude and longitude, not along a great circle. Over
    72 h in the tropics the difference is far smaller than the forecast error.

    Args:
        init_times: the table from ``samples.make_init_times``.

    Returns:
        Forecasts for every row of init_times at each lead time, even when no truth
        exists yet for that time. ``scoring.score`` reports those.
    """
    past_lat = init_times[f"lat_m{PERSISTENCE_WINDOW_H:02d}"]
    past_lon = init_times[f"lon_m{PERSISTENCE_WINDOW_H:02d}"]
    dlat = init_times["lat"] - past_lat
    dlon = init_times["lon"] - past_lon

    forecasts = []
    for lead_h in LEAD_HOURS:
        steps = lead_h / PERSISTENCE_WINDOW_H
        forecasts.append(
            _forecast_table(
                PERSISTENCE_NAME,
                init_times,
                lead_h,
                lat=init_times["lat"] + steps * dlat,
                lon=init_times["lon"] + steps * dlon,
            )
        )
    return pd.concat(forecasts, ignore_index=True)


def _cliper_inputs(init_times: pd.DataFrame) -> pd.DataFrame:
    """Build the eight CLIPER inputs, all known at t0.

    Position stands in for climatology, since storms at a given place tend to move a
    given way. Past motion is the persistence part. Two motion windows let the fit
    decide how to weigh recent motion against the steadier 24 h average.

    Day of year goes in as sin and cos so that the fit sees it as a circle. As a plain
    number, 31 December (365) and 1 January (1) would look a year apart, not a day.
    USA_WIND is left out, so the 24 rows with no wind need no special handling.
    """
    angle = 2 * np.pi * init_times["init_time"].dt.day_of_year / DAYS_PER_YEAR
    return pd.DataFrame(
        {
            "lat": init_times["lat"],
            "lon": init_times["lon"],
            "dlat_12h": init_times["lat"] - init_times["lat_m12"],
            "dlon_12h": init_times["lon"] - init_times["lon_m12"],
            "dlat_24h": init_times["lat"] - init_times["lat_m24"],
            "dlon_24h": init_times["lon"] - init_times["lon_m24"],
            "day_sin": np.sin(angle),
            "day_cos": np.cos(angle),
        }
    )


def fit_cliper_on(rows: pd.DataFrame, truth: pd.DataFrame) -> dict[int, LinearRegression]:
    """Fit one linear regression per lead time, on exactly the rows it's given.

    Each model predicts the change in position (dlat, dlon), not the position itself.
    Change is what's similar from storm to storm. Absolute position would make the
    fit spend its effort rediscovering where the storm already is.

    Each lead time gets its own model because the best weights can differ with lead
    time. A single set of weights would force the 72 h fit to match the 24 h one.

    Args:
        rows: the start times to learn from.
        truth: the table from ``samples.make_truth``.

    Returns:
        {lead_h: fitted model}, to pass to ``predict_cliper``.
    """
    models = {}
    for lead_h in LEAD_HOURS:
        cases = with_truth_at(rows, truth, lead_h)
        targets = pd.DataFrame(
            {
                "dlat": cases["truth_lat"] - cases["lat"],
                "dlon": cases["truth_lon"] - cases["lon"],
            }
        )
        models[lead_h] = LinearRegression().fit(_cliper_inputs(cases), targets)
    return models


def fit_cliper(init_times: pd.DataFrame, truth: pd.DataFrame) -> dict[int, LinearRegression]:
    """Fit CLIPER on the train split only, for notebooks 02 and 03.

    The train filter is inside this function, not left to the caller, so that passing
    the full table can't leak validation or test storms into the fit.

    Args:
        init_times: the table from ``samples.make_init_times``.
        truth: the table from ``samples.make_truth``.

    Returns:
        {lead_h: fitted model}, to pass to ``predict_cliper``.
    """
    train = init_times[init_times["split"] == "train"]
    models = fit_cliper_on(train, truth)
    for lead_h in LEAD_HOURS:
        print(f"CLIPER +{lead_h} h: fitted on {len(with_truth_at(train, truth, lead_h)):,} train cases")
    return models


def predict_cliper(
    models: dict[int, LinearRegression],
    init_times: pd.DataFrame,
) -> pd.DataFrame:
    """Return CLIPER forecasts for every row of init_times.

    Args:
        models: the result of ``fit_cliper``.
        init_times: the start times to forecast. Pass only the split being scored.

    Returns:
        Forecasts for every row of init_times at each lead time.
    """
    inputs = _cliper_inputs(init_times)

    forecasts = []
    for lead_h, model in models.items():
        change = model.predict(inputs)
        forecasts.append(
            _forecast_table(
                CLIPER_NAME,
                init_times,
                lead_h,
                lat=init_times["lat"] + change[:, 0],
                lon=init_times["lon"] + change[:, 1],
            )
        )
    return pd.concat(forecasts, ignore_index=True)


def persistence_forecast(
    learning: pd.DataFrame,
    truth: pd.DataFrame,
    held_out: pd.DataFrame,
) -> pd.DataFrame:
    """Persistence, in the form ``cv.out_of_fold`` takes. It has nothing to learn."""
    return persistence(held_out)


def cliper_forecast(
    learning: pd.DataFrame,
    truth: pd.DataFrame,
    held_out: pd.DataFrame,
) -> pd.DataFrame:
    """Fit CLIPER on the learning rows and forecast the held-out rows, for ``cv.out_of_fold``."""
    return predict_cliper(fit_cliper_on(learning, truth), held_out)
