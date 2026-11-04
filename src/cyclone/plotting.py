"""Draw one storm's track on a map, as a quick visual check of the data.

Numbers can pass every check and still be wrong in ways a map shows at once: a
position in the wrong hemisphere, a jump across the ocean, a storm far from where
it's known to have gone.
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


def _label_point(ax: Axes, row: pd.Series, text: str, offset: tuple[int, int]) -> None:
    ax.annotate(
        text,
        xy=(row["USA_LON"], row["USA_LAT"]),
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

    ax.set_extent(_square_extent(lat, lon), crs=ccrs.PlateCarree())
    ax.set_facecolor(OCEAN_COLOR)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=LAND_COLOR, edgecolor="none")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=COAST_COLOR, linewidth=0.6)
    gridlines = ax.gridlines(draw_labels=True, color=GRID_COLOR, linewidth=0.5)
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 9, "color": MUTED_TEXT_COLOR}
    gridlines.ylabel_style = {"size": 9, "color": MUTED_TEXT_COLOR}

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
    _label_point(ax, first, f"Start {first['ISO_TIME']:%d %b %H}Z", (-14, -14))
    _label_point(ax, last, f"End {last['ISO_TIME']:%d %b %H}Z", (14, 14))
    if wind.notna().any():
        peak = storm.loc[wind.idxmax()]
        # Place the label off to the side of the track, not along it. On an east-west
        # track it goes above and runs west, clear of the Start label, which sits below.
        runs_east_west = lon.max() - lon.min() > lat.max() - lat.min()
        _label_point(
            ax,
            peak,
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
