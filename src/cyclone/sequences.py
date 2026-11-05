"""Turn each forecast start time into the story an LSTM reads, one fix at a time.

XGBoost gets one row of numbers per start time, so someone has to decide in advance what
to summarize: motion over 6, 12 and 24 h, and so on. An LSTM reads the fixes in order and
works out its own summary, so it's given the track itself. That's what this file builds:

    steps: per fix, in time order: 6 h change in lat and lon, lat, lon, wind, wind missing
    fixed: per forecast: day of year as sin and cos, and distance to land at t0

**Same start times as every other model.** A sequence is built for each row given, so the
cases stay identical, and every model is judged on the same storms and times.

**Length varies.** A sequence holds the fixes from up to 72 h before t0 up to t0, so at
most 13. The fixes are matched by timestamp, walking back 6 h at a time, and the walk stops
at a gap: what comes before a gap isn't a steady 6 h history, so the sequence starts after
it. It can't be shorter than 5 fixes, because every start time has a full 24 h of history.
Short sequences are padded at the end so they fit in one array, and the padding is blank.

**Nothing after t0 can get in.** Only fixes at t0 or earlier are read, and this file never
sees the truth table. Targets belong to the model, in ``lstm_model.py``.

**Missing numbers.** XGBoost can take a blank, because a tree just sends it down one side.
An LSTM can't: it multiplies every input by a weight, and one blank turns everything after
it blank too. So each blank needs a number, and there are two kinds:

    wind:   USA_WIND is blank for two storms in 1980. It's filled with the average wind
            and a separate 0/1 "wind missing" flag says so at each step. The fill keeps
            the arithmetic working, and it's a harmless middle value. The flag keeps it
            honest, since the network can tell a real average wind from an invented one,
            and learn to ignore the wind where the flag is on.
    first change: the first fix in a sequence has no fix before it, so its change is
            unknown. It's filled in the same way, with the average change. It needs no
            flag, because it's always the first step, and the network knows that.

**Scaling** puts the inputs on similar ranges. Latitude averages about 13, longitude
about 80, wind about 39 and distance to land about 270 km, while a 6 h change is under 1. The network starts with small random weights and adjusts them all at the same
rate, so big inputs would swamp small ones and training would crawl. Scaling subtracts
each input's mean and divides by its spread, so all of them sit around 0, mostly between
-2 and 2.

It has to be fitted on the learning rows only, inside each fold. A mean and spread are
learned from data just as a weight is. Fit them on all rows and the held-out block's
storms have shaped how every input is read: an unusually strong block would shift the
wind scale, and the forecast of that block would depend on its own storms. It's a small
leak, but it's the kind cross-validation exists to rule out, and test day would have to
break the same rule by scaling with storms it hasn't met.

Filling comes after scaling, and the fill is 0, which by then is the learning rows'
average. So the fill is learned from the same rows, and held-out storms don't affect it.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

KEY_COLUMNS = ["SID", "init_time"]

STEP_COLUMNS = ["dlat", "dlon", "lat", "lon", "wind", "wind_missing"]
FIXED_COLUMNS = ["day_sin", "day_cos", "dist_to_land_km"]

# Hours between fixes in an unbroken stretch of track.
STEP_H = 6
MAX_HISTORY_H = 72
# The history every start time is guaranteed by samples.make_init_times.
MIN_HISTORY_H = 24
MIN_STEPS = MIN_HISTORY_H // STEP_H + 1

# A flag is already 0 or 1, and means something exact, so it isn't scaled.
_FLAG = STEP_COLUMNS.index("wind_missing")


@dataclass(frozen=True)
class Sequences:
    """The inputs for a set of forecasts, in the same order as the rows they came from.

    Attributes:
        keys: one row per forecast, with SID and init_time.
        steps: shape (forecasts, longest possible sequence, STEP_COLUMNS). Blank where a
            value is unknown, and in the padding after a sequence ends.
        lengths: the real number of fixes in each sequence.
        fixed: shape (forecasts, FIXED_COLUMNS).
    """

    keys: pd.DataFrame
    steps: np.ndarray
    lengths: np.ndarray
    fixed: np.ndarray

    def __len__(self) -> int:
        return len(self.keys)


@dataclass(frozen=True)
class Scaler:
    """Each input's mean and spread on the rows it was fitted on."""

    step_mean: np.ndarray
    step_std: np.ndarray
    fixed_mean: np.ndarray
    fixed_std: np.ndarray


def _first_fix(times: np.ndarray, end: int, max_steps: int) -> int:
    """Return where the sequence ending at fix ``end`` starts.

    It walks back while each earlier fix is exactly 6 h before, so a gap ends the walk.
    """
    first = end
    while (
        end - first + 1 < max_steps
        and first > 0
        and times[first] - times[first - 1] == np.timedelta64(STEP_H, "h")
    ):
        first -= 1
    return first


def _steps_of(track: pd.DataFrame, first: int, end: int) -> np.ndarray:
    """Return one sequence's steps, from fix ``first`` to fix ``end`` inclusive.

    The change at the first step is left blank, even if an earlier fix exists. Otherwise
    a limit on the history wouldn't be a limit, since the first change would still carry
    the fix from before it.
    """
    lat = track["USA_LAT"].to_numpy()[first : end + 1]
    lon = track["USA_LON"].to_numpy()[first : end + 1]
    wind = track["USA_WIND"].to_numpy()[first : end + 1]
    return np.column_stack(
        [
            np.diff(lat, prepend=np.nan),
            np.diff(lon, prepend=np.nan),
            lat,
            lon,
            wind,
            np.isnan(wind).astype(float),
        ]
    )


def build_sequences(
    rows: pd.DataFrame,
    df: pd.DataFrame,
    history_h: int = MAX_HISTORY_H,
) -> Sequences:
    """Return the sequence and fixed inputs for every row, with none added or removed.

    Args:
        rows: the start times, such as the table from ``cv.model_rows``. It supplies the
            fixed inputs, which ``features.build_features`` already worked out.
        df: the cleaned table, which holds the fixes further back than rows does.
        history_h: how far before t0 a sequence can reach. Pass 24 to give a model the
            same history as XGBoost.

    Returns:
        The inputs, in the same order as rows.

    Raises:
        ValueError: if a start time isn't a fix in df, a storm repeats a timestamp, or a
            sequence is shorter than the 24 h every start time should have.
    """
    max_steps = history_h // STEP_H + 1
    fixes = df[["SID", "ISO_TIME", "USA_LAT", "USA_LON", "USA_WIND"]].sort_values(
        ["SID", "ISO_TIME"]
    )
    if fixes.duplicated(["SID", "ISO_TIME"]).any():
        raise ValueError("A storm repeats a timestamp, so its fixes can't be put in order.")
    tracks = dict(iter(fixes[fixes["SID"].isin(rows["SID"])].groupby("SID")))

    steps = np.full((len(rows), max_steps, len(STEP_COLUMNS)), np.nan)
    lengths = np.zeros(len(rows), dtype=int)
    for number, (sid, init_time) in enumerate(zip(rows["SID"], rows["init_time"], strict=True)):
        track = tracks[sid]
        times = track["ISO_TIME"].to_numpy()
        end = int(np.searchsorted(times, np.datetime64(init_time)))
        if end == len(times) or times[end] != np.datetime64(init_time):
            raise ValueError(f"Storm {sid} has no fix at {init_time}.")

        first = _first_fix(times, end, max_steps)
        lengths[number] = end - first + 1
        steps[number, : lengths[number]] = _steps_of(track, first, end)

    if len(rows) and lengths.min() < min(MIN_STEPS, max_steps):
        raise ValueError(f"A sequence has {lengths.min()} fixes, fewer than the minimum.")

    return Sequences(
        keys=rows[KEY_COLUMNS].reset_index(drop=True),
        steps=steps,
        lengths=lengths,
        fixed=rows[FIXED_COLUMNS].to_numpy(dtype=float),
    )


def select(sequences: Sequences, rows: pd.DataFrame) -> Sequences:
    """Return the sequences for the rows given, in their order.

    Matching is by storm and start time, not by position, so a fold's rows can't be paired
    with another row's inputs.

    Raises:
        ValueError: if a row has no sequence.
    """
    numbered = sequences.keys.assign(position=np.arange(len(sequences)))
    matched = rows[KEY_COLUMNS].merge(numbered, on=KEY_COLUMNS, how="left", validate="1:1")
    if matched["position"].isna().any():
        raise ValueError(f"{matched['position'].isna().sum()} rows have no sequence.")

    positions = matched["position"].to_numpy(dtype=int)
    return Sequences(
        keys=sequences.keys.iloc[positions].reset_index(drop=True),
        steps=sequences.steps[positions],
        lengths=sequences.lengths[positions],
        fixed=sequences.fixed[positions],
    )


def fit_scaler(sequences: Sequences) -> Scaler:
    """Return each input's mean and spread, ignoring blanks and padding.

    Pass only the learning rows. Each fixed input counts once per forecast, and a step
    input once per step, so a fix that's in many sequences is counted that many times:
    this is the data as the network really sees it.

    An input with no spread is left unscaled, since dividing by 0 would give blanks.
    """
    step_mean = np.nanmean(sequences.steps, axis=(0, 1))
    step_std = np.nanstd(sequences.steps, axis=(0, 1))
    step_mean[_FLAG], step_std[_FLAG] = 0.0, 1.0
    fixed_std = sequences.fixed.std(axis=0)
    return Scaler(
        step_mean=step_mean,
        step_std=np.where(step_std > 0, step_std, 1.0),
        fixed_mean=sequences.fixed.mean(axis=0),
        fixed_std=np.where(fixed_std > 0, fixed_std, 1.0),
    )


def scale(sequences: Sequences, scaler: Scaler) -> tuple[np.ndarray, np.ndarray]:
    """Return the scaled steps and fixed inputs, with every blank filled.

    A blank becomes 0, which after scaling is the average on the rows the scaler saw. That
    covers a missing wind, the first step's change and the padding. The padding's value
    never matters, since a sequence's answer is read at its real last step.
    """
    steps = (sequences.steps - scaler.step_mean) / scaler.step_std
    fixed = (sequences.fixed - scaler.fixed_mean) / scaler.fixed_std
    return np.nan_to_num(steps, nan=0.0), fixed
