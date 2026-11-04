"""Draw tracks, forecasts and error charts, as visual checks of data and results.

Numbers can pass every check and still be wrong in ways a map shows at once: a
position in the wrong hemisphere, a jump across the ocean, a forecast heading the
wrong way.
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Fixed to the range of the cleaned table (10-150 kt), not to each storm's own
# range, so the same colour means the same wind on every map.
WIND_MIN_KT = 10
WIND_MAX_KT = 150

# One hue, light to dark: darker means stronger. The lightest step is still dark
# enough to see against the pale ocean.
WIND_COLORMAP = LinearSegmentedColormap.from_list(
    "wind", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
)

OCEAN_COLOR = "#fcfcfb"
LAND_COLOR = "#f0efec"
COAST_COLOR = "#898781"
LINE_COLOR = "#898781"
GRID_COLOR = "#e1e0d9"
TEXT_COLOR = "#0b0b0b"
MUTED_TEXT_COLOR = "#52514e"

MAP_PADDING_DEG = 4

# One fixed colour per model, so a model looks the same on every chart even when
# another is added or left out. Add new models here rather than cycling colours.
MODEL_LABELS = {"persistence": "Persistence", "cliper": "CLIPER"}
MODEL_COLORS = {"persistence": "#2a78d6", "cliper": "#eb6834"}


def _select_storm(
    df: pd.DataFrame,
    sid: str | None,
    name: str | None,
    season: int | None,
) -> pd.DataFrame:
    """Return the rows of exactly one storm, or raise an error saying why not."""
    if sid is not None:
        storm = df[df["SID"] == sid]
        if storm.empty:
            raise ValueError(f"No storm with SID {sid!r}.")
        return storm

    if name is None or season is None:
        raise ValueError("Pass either sid, or both name and season.")

    # NAME can hold two spellings or two names, as in "HIKAA:HIKKA" and
    # "BULBUL:MATMO", so compare against each part.
    in_season = df[df["SEASON"] == season]
    wanted = name.upper()
    matches = in_season["NAME"].str.split(":").apply(lambda names: wanted in names)
    storm = in_season[matches]

    if storm.empty:
        available = sorted(in_season["NAME"].unique())
        raise ValueError(f"No storm named {name!r} in {season}. Names that season: {available}")
    # One name can match several storms: UNNAMED often appears more than once.
    if storm["SID"].nunique() > 1:
        sids = sorted(storm["SID"].unique())
        raise ValueError(f"{name!r} in {season} matches several storms; pass sid= with one of {sids}")
    return storm


def _draw_base_map(ax: Axes, extent: list[float]) -> None:
    """Draw the ocean, land, coast and grid that every map here shares."""
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_facecolor(OCEAN_COLOR)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=LAND_COLOR, edgecolor="none")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=COAST_COLOR, linewidth=0.6)
    gridlines = ax.gridlines(draw_labels=True, color=GRID_COLOR, linewidth=0.5)
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 9, "color": MUTED_TEXT_COLOR}
    gridlines.ylabel_style = {"size": 9, "color": MUTED_TEXT_COLOR}


def _square_extent(lat: pd.Series, lon: pd.Series) -> list[float]:
    """Return map limits centred on the track, equally wide and tall in degrees.

    Most Bay of Bengal storms run north-south. Cutting the map tight to such a track
    gives a thin strip that leaves much of the figure blank and hides the coast on
    either side, so the narrower direction is widened to match.
    """
    half_size = max(lon.max() - lon.min(), lat.max() - lat.min()) / 2 + MAP_PADDING_DEG
    lon_mid = (lon.min() + lon.max()) / 2
    lat_mid = (lat.min() + lat.max()) / 2
    return [lon_mid - half_size, lon_mid + half_size, lat_mid - half_size, lat_mid + half_size]


def _label_point(
    ax: Axes,
    lon: float,
    lat: float,
    text: str,
    offset: tuple[int, int],
) -> None:
    ax.annotate(
        text,
        xy=(lon, lat),
        xytext=offset,
        textcoords="offset points",
        fontsize=9,
        color=TEXT_COLOR,
        ha="left" if offset[0] >= 0 else "right",
        va="center",
        arrowprops={"arrowstyle": "-", "color": COAST_COLOR, "linewidth": 0.6},
    )


def plot_track(
    df: pd.DataFrame,
    sid: str | None = None,
    name: str | None = None,
    season: int | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot one storm on a map, with dots coloured by JTWC wind speed.

    Pick the storm with either ``sid``, or ``name`` and ``season``:

        plot_track(df, name="FANI", season=2019)
        plot_track(df, sid="2019116N02090")

    Args:
        df: a table with the columns kept by ``filters.clean_ni``.
        sid: the IBTrACS storm ID.
        name: storm name, in any case. Either part of a joined name such as
            "HIKAA:HIKKA" works.
        season: the storm's SEASON, required with ``name``.
        ax: an existing map axes to draw on, for several storms in one figure.
            It must use a cartopy projection. Leave it out to create a new figure.

    Returns:
        The map axes, so a caller can add to it or save it.
    """
    storm = _select_storm(df, sid, name, season).sort_values("ISO_TIME")
    lat, lon, wind = storm["USA_LAT"], storm["USA_LON"], storm["USA_WIND"]

    if ax is None:
        _, ax = plt.subplots(
            figsize=(8, 7),
            subplot_kw={"projection": ccrs.PlateCarree()},
            layout="constrained",
        )

    _draw_base_map(ax, _square_extent(lat, lon))

    ax.plot(lon, lat, color=LINE_COLOR, linewidth=1, transform=ccrs.PlateCarree(), zorder=3)

    # scatter draws a NaN colour as nothing, which would make a real position vanish.
    # Points with no wind get a hollow marker instead (32 rows, all from 1980).
    no_wind = wind.isna()
    ax.scatter(
        lon[no_wind],
        lat[no_wind],
        s=36,
        facecolors="none",
        edgecolors=LINE_COLOR,
        transform=ccrs.PlateCarree(),
        zorder=4,
        label="No wind recorded",
    )
    dots = ax.scatter(
        lon[~no_wind],
        lat[~no_wind],
        c=wind[~no_wind],
        cmap=WIND_COLORMAP,
        norm=Normalize(WIND_MIN_KT, WIND_MAX_KT),
        s=36,
        edgecolors=OCEAN_COLOR,  # a thin ring keeps overlapping dots separate
        linewidths=0.6,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    ax.figure.colorbar(dots, ax=ax, label="Wind speed (kt, JTWC 1-minute)", shrink=0.8)
    if no_wind.any():
        ax.legend(loc="lower left", fontsize=9, frameon=False)

    # Label only the moments that matter. A number on every dot would be unreadable.
    first, last = storm.iloc[0], storm.iloc[-1]
    _label_point(
        ax, first["USA_LON"], first["USA_LAT"], f"Start {first['ISO_TIME']:%d %b %H}Z", (-14, -14)
    )
    _label_point(
        ax, last["USA_LON"], last["USA_LAT"], f"End {last['ISO_TIME']:%d %b %H}Z", (14, 14)
    )
    if wind.notna().any():
        peak = storm.loc[wind.idxmax()]
        # Place the label off to the side of the track, not along it. On an east-west
        # track it goes above and runs west, clear of the Start label, which sits below.
        runs_east_west = lon.max() - lon.min() > lat.max() - lat.min()
        _label_point(
            ax,
            peak["USA_LON"],
            peak["USA_LAT"],
            f"Peak {peak['USA_WIND']:.0f} kt, {peak['ISO_TIME']:%d %b %H}Z",
            (-4, 24) if runs_east_west else (16, 0),
        )

    ax.set_title(
        f"{first['NAME']} ({first['SEASON']}): JTWC track, {len(storm)} six-hourly fixes",
        fontsize=12,
        color=TEXT_COLOR,
        loc="left",
    )
    return ax


def plot_error_by_lead(summary: pd.DataFrame, title: str, ax: Axes | None = None) -> Axes:
    """Plot mean track error against lead time, one line per model.

    The y axis starts at 0 km, so the gap between lines reads as its true share of the
    error. Only the last value on each line is printed; the table has the rest.

    Args:
        summary: the table from ``scoring.summarize``.
        title: the chart title, which should say which storms and split it covers.
        ax: an existing axes to draw on. Leave it out to create a new figure.

    Returns:
        The axes, so a caller can add to it or save it.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.set_facecolor(OCEAN_COLOR)
    ax.figure.set_facecolor(OCEAN_COLOR)

    for model, rows in summary.groupby("model", sort=False):
        rows = rows.sort_values("lead_h")
        ax.plot(
            rows["lead_h"],
            rows["mean_km"],
            color=MODEL_COLORS[model],
            linewidth=2,
            marker="o",
            markersize=8,
            markeredgecolor=OCEAN_COLOR,  # a ring keeps the dot clear of the other line
            markeredgewidth=2,
            label=MODEL_LABELS[model],
        )
        last = rows.iloc[-1]
        ax.annotate(
            f"{last['mean_km']:,.0f} km",
            xy=(last["lead_h"], last["mean_km"]),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=10,
            color=TEXT_COLOR,
            va="center",
        )

    leads = sorted(summary["lead_h"].unique())
    ax.set_xticks(leads, [f"+{lead} h" for lead in leads])
    ax.set_xlim(leads[0] - 6, leads[-1] + 10)  # room for the end labels
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Mean great-circle error (km)", color=MUTED_TEXT_COLOR, fontsize=10)
    ax.set_xlabel("Forecast lead time", color=MUTED_TEXT_COLOR, fontsize=10)
    ax.tick_params(colors=MUTED_TEXT_COLOR, labelsize=10, length=0)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(COAST_COLOR)
    ax.legend(loc="upper left", fontsize=10, frameon=False, labelcolor=TEXT_COLOR)
    ax.set_title(title, fontsize=12, color=TEXT_COLOR, loc="left")
    return ax


def plot_forecasts(
    truth: pd.DataFrame,
    forecasts: pd.DataFrame,
    sid: str,
    init_time: pd.Timestamp,
    title: str,
    ax: Axes | None = None,
) -> Axes:
    """Map one storm's true track with each model's forecast from one start time.

    The true track is a dark neutral rather than the wind colours of ``plot_track``.
    Those are blue, and would be hard to tell from the blue persistence line.

    Args:
        truth: the table from ``samples.make_truth``.
        forecasts: forecasts in the scoring format, holding any number of models.
        sid: the IBTrACS storm ID.
        init_time: the forecast start time, which must be a fix of this storm.
        title: the map title.
        ax: an existing map axes. It must use a cartopy projection. Leave it out
            to create a new figure.

    Returns:
        The map axes, so a caller can add to it or save it.
    """
    track = truth[truth["SID"] == sid].sort_values("valid_time")
    chosen = forecasts[(forecasts["SID"] == sid) & (forecasts["init_time"] == init_time)]
    start = track[track["valid_time"] == init_time]
    if start.empty:
        raise ValueError(f"Storm {sid} has no fix at {init_time}.")
    if chosen.empty:
        raise ValueError(f"No forecasts for storm {sid} from {init_time}.")
    start = start.iloc[0]

    if ax is None:
        _, ax = plt.subplots(
            figsize=(8, 7),
            subplot_kw={"projection": ccrs.PlateCarree()},
            layout="constrained",
        )

    # Fit the map to the forecasts too, since a bad one can leave the storm's track.
    every_lat = pd.concat([track["lat"], chosen["lat"]])
    every_lon = pd.concat([track["lon"], chosen["lon"]])
    _draw_base_map(ax, _square_extent(every_lat, every_lon))

    transform = ccrs.PlateCarree()
    ax.plot(
        track["lon"],
        track["lat"],
        color=TEXT_COLOR,
        linewidth=2,
        marker="o",
        markersize=4,
        transform=transform,
        zorder=3,
        label="JTWC track",
    )

    for model, rows in chosen.groupby("model", sort=False):
        rows = rows.sort_values("lead_h")
        ax.plot(
            [start["lon"], *rows["lon"]],
            [start["lat"], *rows["lat"]],
            color=MODEL_COLORS[model],
            linewidth=2,
            marker="o",
            markersize=8,
            markevery=list(range(1, len(rows) + 1)),  # the shared start gets its own marker
            markeredgecolor=OCEAN_COLOR,
            markeredgewidth=2,
            transform=transform,
            zorder=4,
            label=MODEL_LABELS[model],
        )

    def mark_fix(lon: float, lat: float) -> None:
        ax.plot(
            lon,
            lat,
            marker="o",
            markersize=9,
            markerfacecolor=TEXT_COLOR,
            markeredgecolor=OCEAN_COLOR,
            markeredgewidth=2,
            transform=transform,
            zorder=5,
        )

    # Mark the truth at each lead time, so the eye can pair it with a forecast dot.
    # The labels stand well off the track with a leader line: right beside a true
    # position they'd land next to a forecast dot and appear to name that dot instead.
    for lead_h in sorted(chosen["lead_h"].unique()):
        valid_time = init_time + pd.Timedelta(hours=int(lead_h))
        fix = track[track["valid_time"] == valid_time]
        if fix.empty:
            continue
        lon, lat = fix["lon"].iloc[0], fix["lat"].iloc[0]
        mark_fix(lon, lat)
        _label_point(ax, lon, lat, f"True +{lead_h} h", (26, 42))

    mark_fix(start["lon"], start["lat"])
    _label_point(ax, start["lon"], start["lat"], f"Start {init_time:%d %b %H}Z", (24, 26))
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    ax.set_title(title, fontsize=12, color=TEXT_COLOR, loc="left")
    return ax
