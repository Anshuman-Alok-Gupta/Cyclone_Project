"""Turn the start-time table into the numbers a model reads.

Every input here is built from ``samples.make_init_times``, the table that holds only
JTWC fixes at t0 and earlier, so nothing after the forecast start can reach a model.
Each row is calculated from that row alone. Nothing is averaged over the dataset, so
another storm or a later fix can't affect it either.

    position:    where the storm is
    motion:      how it has been moving, and whether that's changing
    intensity:   how strong it is, and whether it's strengthening
    season:      the time of year
    land:        how far the nearest coast is

Left out on purpose, as CLAUDE.md requires:

    USA_PRES:    missing before 2001, so a model could learn "missing means old storm"
                 instead of anything about pressure
    DIST2LAND:   follows the combined position, so it's recomputed here from JTWC's
    STORM_SPEED, STORM_DIR: use a position after t0
    LANDFALL:    looks ahead, and is banned for good
"""

from functools import lru_cache

import numpy as np
import pandas as pd
import shapely
from sklearn.neighbors import BallTree

from cyclone.baselines import DAYS_PER_YEAR
from cyclone.scoring import EARTH_RADIUS_KM, great_circle_km

# These say which storm and time a row is. They stay in the table so it can be matched
# to truth and split, but a model only ever reads FEATURE_COLUMNS, so a storm ID or a
# date can't become an input by accident.
ID_COLUMNS = ["SID", "init_time", "split"]

FEATURE_COLUMNS = [
    "lat",
    "lon",
    "dlat_6h",
    "dlon_6h",
    "dlat_12h",
    "dlon_12h",
    "dlat_24h",
    "dlon_24h",
    "speed_kmh",
    "heading_sin",
    "heading_cos",
    "dlat_change",
    "dlon_change",
    "wind",
    "dwind_12h",
    "dwind_24h",
    "day_sin",
    "day_cos",
    "dist_to_land_km",
]

# Hours of past motion behind speed, heading and the change in motion. The same window
# as persistence, because a single 6 h step is short enough for small errors in each
# fix to swing the direction.
MOTION_WINDOW_H = 12

# Spacing of the points placed along the coast, about 2 km. Distance is measured to the
# nearest point, and Natural Earth's own points can be far apart on a straight coast.
# At this spacing the distance stays within 0.5 km of one twice as fine.
COAST_SPACING_DEG = 0.02


@lru_cache(maxsize=1)
def _load_land() -> tuple[shapely.Geometry, BallTree]:
    """Return the land polygons and a search tree of points along every coast.

    This takes several seconds, so the result is kept for the rest of the run.

    cartopy is imported here and not at the top of the file so that importing this
    module for its column lists doesn't load it. It downloads the land file the first
    time, so that first run needs internet.
    """
    from cartopy.io import shapereader

    path = shapereader.natural_earth(resolution="10m", category="physical", name="land")
    # Merging removes any edge shared by two polygons, which isn't a real coast.
    land = shapely.union_all(list(shapereader.Reader(path).geometries()))
    shapely.prepare(land)  # makes the many over-land tests fast

    coast = shapely.segmentize(shapely.boundary(land), COAST_SPACING_DEG)
    lon_lat = shapely.get_coordinates(coast)
    # A ball tree finds the nearest of about a million points without checking each one,
    # and its haversine metric measures along the Earth's surface. It wants [lat, lon].
    tree = BallTree(np.radians(lon_lat[:, ::-1]), metric="haversine")
    return land, tree


def distance_to_land_km(lat, lon) -> np.ndarray:
    """Return the great-circle distance to the nearest coast, and 0 over land.

    Natural Earth's 10 m land polygons include small islands such as the Maldives. The
    land mask behind IBTrACS's DIST2LAND leaves those out, so the two differ near them.
    docs/components.md has the comparison.

    Args:
        lat, lon: positions in degrees. Lists, arrays or pandas columns.

    Returns:
        One distance in km for each position.
    """
    land, tree = _load_land()
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    # The tree returns an angle in radians, which the Earth's radius turns into km.
    angle, _ = tree.query(np.radians(np.column_stack([lat, lon])), k=1)
    distance = angle[:, 0] * EARTH_RADIUS_KM
    # A coast is equally close from either side, so land needs its own test.
    distance[shapely.contains_xy(land, lon, lat)] = 0.0
    return distance


def _motion_features(init_times: pd.DataFrame) -> dict[str, pd.Series]:
    """Build the motion inputs from the current and past JTWC positions.

    Three windows go in because each says something different: 6 h is the latest but
    the noisiest, and 24 h the steadiest but the slowest to show a turn.

    The change in motion is the last 12 h minus the 12 h before that. Positive
    dlat_change means the storm's northward motion has grown. Together the two
    components show speeding up, slowing down, or starting to turn.
    """
    lat, lon = init_times["lat"], init_times["lon"]
    lat_12h_ago = init_times[f"lat_m{MOTION_WINDOW_H:02d}"]
    lon_12h_ago = init_times[f"lon_m{MOTION_WINDOW_H:02d}"]

    # A degree of longitude is shorter away from the equator, so it's scaled before
    # taking a direction. Otherwise a storm heading north-east would look too easterly.
    north = lat - lat_12h_ago
    east = (lon - lon_12h_ago) * np.cos(np.radians((lat + lat_12h_ago) / 2))
    moved = np.hypot(north, east)

    return {
        "dlat_6h": lat - init_times["lat_m06"],
        "dlon_6h": lon - init_times["lon_m06"],
        "dlat_12h": north,
        "dlon_12h": lon - lon_12h_ago,
        "dlat_24h": lat - init_times["lat_m24"],
        "dlon_24h": lon - init_times["lon_m24"],
        "speed_kmh": great_circle_km(lat_12h_ago, lon_12h_ago, lat, lon) / MOTION_WINDOW_H,
        # Heading is a circle. As a plain angle, 359 and 1 degrees would look far apart
        # when they're almost the same direction, so it goes in as a point on a circle:
        # sin for east and cos for north. A storm that hasn't moved has no heading, and
        # 0/0 leaves these blank rather than inventing one. No such row exists today.
        "heading_sin": east / moved,
        "heading_cos": north / moved,
        "dlat_change": north - (lat_12h_ago - init_times["lat_m24"]),
        "dlon_change": (lon - lon_12h_ago) - (lon_12h_ago - init_times["lon_m24"]),
    }


def _intensity_features(init_times: pd.DataFrame) -> dict[str, pd.Series]:
    """Build the wind inputs, leaving a missing wind as NaN.

    XGBoost doesn't need these filled in. When a tree splits on a column, it tries
    sending the rows with a missing value down each side, and keeps whichever side
    fits better as that split's default direction. Filling them with an average would
    instead invent a wind for storms that have none on record.

    A change is blank if either wind is blank, since subtracting NaN gives NaN.
    """
    wind = init_times["wind"]
    return {
        "wind": wind,
        "dwind_12h": wind - init_times["wind_m12"],
        "dwind_24h": wind - init_times["wind_m24"],
    }


def _season_features(init_times: pd.DataFrame) -> dict[str, pd.Series]:
    """Build day of year as a point on a circle, as CLIPER does.

    As a plain number, 31 December (365) and 1 January (1) would look a year apart.
    """
    angle = 2 * np.pi * init_times["init_time"].dt.day_of_year / DAYS_PER_YEAR
    return {"day_sin": np.sin(angle), "day_cos": np.cos(angle)}


def build_features(init_times: pd.DataFrame) -> pd.DataFrame:
    """Return the model inputs for every start time, with no rows added or removed.

    A row with a blank input stays in, so every model is scored on the same cases.

    Args:
        init_times: the table from ``samples.make_init_times``.

    Returns:
        One row per start time, in the same order, with ID_COLUMNS then FEATURE_COLUMNS.
        ``samples.with_truth_at`` and ``checks.check_samples`` both accept this table.
    """
    columns = {
        **{column: init_times[column] for column in ID_COLUMNS},
        "lat": init_times["lat"],
        "lon": init_times["lon"],
        **_motion_features(init_times),
        **_intensity_features(init_times),
        **_season_features(init_times),
        "dist_to_land_km": distance_to_land_km(init_times["lat"], init_times["lon"]),
    }
    return pd.DataFrame(columns, index=init_times.index)[ID_COLUMNS + FEATURE_COLUMNS]
