"""Combine forecasts from several models by averaging their positions.

Two forecasts that are wrong in different ways can partly cancel out: if one is too far
north and the other too far south, the point between them is closer than either. Real
forecast centres lean on this heavily. It only helps as far as the members' errors
differ, and it has nothing to fit, so it adds no settings that could be tuned to storms.

This file reads only the scoring format, so it can average any models, including the AI
models in Project 3.
"""

import pandas as pd

from cyclone.scoring import CASE_COLUMNS, FORECAST_COLUMNS

CONSENSUS_NAME = "consensus"
CONSENSUS_LSTM_NAME = "consensus_lstm"


def consensus(
    forecasts: pd.DataFrame,
    members: tuple[str, ...],
    name: str = CONSENSUS_NAME,
) -> pd.DataFrame:
    """Return the average of the members' forecast positions, one row per case.

    Latitude and longitude are averaged separately. That isn't exactly the midpoint on a
    sphere, but the difference is tiny between points a few hundred km apart in the
    tropics, and the North Indian Ocean is nowhere near the date line, where averaging
    longitudes would fail.

    Every member must forecast a case for it to be averaged. Without that, a case with
    one member missing would quietly become that member's forecast under this name.

    Args:
        forecasts: forecasts in the scoring format, holding every member. Any other
            columns, such as "fold", are kept if they don't differ within a case.
        members: the models to average, as named in the model column.
        name: the name to give the average in the model column.

    Returns:
        Forecasts in the same format, with the model column set to name.

    Raises:
        ValueError: if a member is missing altogether, or a case lacks a member.
    """
    missing = set(members) - set(forecasts["model"])
    if missing:
        raise ValueError(f"Forecasts have no model named {sorted(missing)}.")

    chosen = forecasts[forecasts["model"].isin(members)]
    keys = [column for column in chosen.columns if column not in ("model", "lat", "lon")]
    grouped = chosen.groupby(keys, sort=False)

    incomplete = grouped["model"].nunique() != len(members)
    if incomplete.any():
        raise ValueError(f"{incomplete.sum()} cases don't have a forecast from every member.")

    averaged = grouped[["lat", "lon"]].mean().reset_index().assign(model=name)
    extras = [column for column in keys if column not in CASE_COLUMNS]
    return averaged[[*FORECAST_COLUMNS, *extras]]
