"""XGBoost, the first machine-learning track model.

XGBoost builds a chain of small decision trees. Each tree is a set of yes/no questions
about the inputs ("is the storm north of 15N?"), and each new tree tries to correct
what the trees before it still get wrong. Unlike CLIPER's straight-line fit, this lets
it learn rules that only apply in some situations, such as "storms this far north in
May tend to recurve".

The target is the same as CLIPER's: the change in position from t0, not the position.

Six separate models are fitted, one for each pair of lead time and component:

    +24 h dlat, +24 h dlon, +48 h dlat, +48 h dlon, +72 h dlat, +72 h dlon

Lead times are separate because what matters changes with range, and each has its own
number of cases. Components are separate because XGBoost's tree regressor predicts one
number at a time. The cost is that a forecast's points aren't tied together: the +48 h
model doesn't know what the +24 h model said, so a track can wobble slightly.

    train:      the trees learn from these storms and nothing else
    validation: stops each model adding trees, and picks the settings
    test:       never read here, so it stays a fair exam for the end of Project 1
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import ParameterSampler

from cyclone import config
from cyclone.features import FEATURE_COLUMNS
from cyclone.samples import LEAD_HOURS, with_truth_at
from cyclone.scoring import FORECAST_COLUMNS, great_circle_km

MODEL_NAME = "xgboost"
COMPONENTS = ("dlat", "dlon")

# Fixes both the choice of settings to try and the random parts of training, so the
# same data and the same device give the same models.
SEED = 42

# Settings tried, out of the 180 possible combinations below. Sampling at random covers
# the range better than a small grid, which would try few values of each setting.
N_COMBINATIONS = 20
SEARCH_SPACE = {
    # How many questions a tree can ask in a row. Deeper trees can capture more
    # specific situations, and can memorize more noise.
    "max_depth": [2, 3, 4, 5, 6],
    # How much of its correction each tree applies. Smaller steps need more trees.
    "learning_rate": [0.02, 0.05, 0.1],
    # The fewest storm moments allowed in a leaf (the end of a chain of questions).
    # With a squared-error target this setting is exactly a count of rows. A higher
    # number stops a tree from building a rule around one storm.
    "min_child_weight": [5, 10, 20, 40],
    # The share of rows each tree sees, picked at random, so that no tree relies on
    # any one unusual row.
    "subsample": [0.6, 0.8, 1.0],
}

# The number of trees isn't sampled above. Each model keeps adding trees, up to this
# limit, until its validation error hasn't improved for EARLY_STOPPING_ROUNDS trees,
# then goes back to its best point. So each model finds its own number.
MAX_TREES = 3000
EARLY_STOPPING_ROUNDS = 50


def pick_device() -> str:
    """Return "cuda" if XGBoost can really train on a GPU here, otherwise "cpu".

    It tries a tiny fit, because seeing a GPU isn't the same as being able to use it.
    The MX130 on the local laptop shows up in nvidia-smi, and XGBoost still refuses it
    as too old. CPU is fast enough at this data size either way.

    A GPU and a CPU can give slightly different models, so results repeat exactly on
    the same device only.
    """
    tiny = xgb.DMatrix(np.array([[0.0], [1.0]]), label=[0.0, 1.0])
    try:
        xgb.train({"device": "cuda", "tree_method": "hist"}, tiny, num_boost_round=1)
    except xgb.core.XGBoostError:
        return "cpu"
    return "cuda"


def sample_settings() -> list[dict]:
    """Return the combinations of settings the search will try, always the same ones."""
    return list(ParameterSampler(SEARCH_SPACE, n_iter=N_COMBINATIONS, random_state=SEED))


def _cases(features: pd.DataFrame, truth: pd.DataFrame, split: str, lead_h: int) -> pd.DataFrame:
    """Return one split's start times with a true position at lead_h, and their targets."""
    cases = with_truth_at(features[features["split"] == split], truth, lead_h)
    return cases.assign(
        dlat=cases["truth_lat"] - cases["lat"],
        dlon=cases["truth_lon"] - cases["lon"],
    )


def fit_models(
    features: pd.DataFrame,
    truth: pd.DataFrame,
    settings: dict,
    device: str,
) -> dict[tuple[int, str], xgb.XGBRegressor]:
    """Fit the six models with one combination of settings.

    The split filters are inside this function, not left to the caller, so passing the
    full table can't leak other storms into the fit. Test rows are never read.

    A missing wind goes in as NaN, which XGBoost handles itself, so no row is dropped.

    Args:
        features: the table from ``features.build_features``.
        truth: the table from ``samples.make_truth``.
        settings: one combination from ``sample_settings``.
        device: "cpu" or "cuda", from ``pick_device``.

    Returns:
        {(lead_h, component): fitted model}, to pass to ``predict_xgboost``.
    """
    models = {}
    for lead_h in LEAD_HOURS:
        train = _cases(features, truth, "train", lead_h)
        val = _cases(features, truth, "val", lead_h)
        for component in COMPONENTS:
            model = xgb.XGBRegressor(
                **settings,
                n_estimators=MAX_TREES,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                objective="reg:squarederror",
                tree_method="hist",
                device=device,
                random_state=SEED,
            )
            model.fit(
                train[FEATURE_COLUMNS],
                train[component],
                eval_set=[(val[FEATURE_COLUMNS], val[component])],
                verbose=False,
            )
            models[(lead_h, component)] = model
    return models


def predict_xgboost(
    models: dict[tuple[int, str], xgb.XGBRegressor],
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Return XGBoost forecasts for every row of features.

    Passing a table rather than bare numbers lets XGBoost check the input names, so a
    saved model refuses inputs that have changed since it was trained.

    Args:
        models: the result of ``fit_models`` or ``load_models``.
        features: the start times to forecast. Pass only the split being scored.

    Returns:
        Forecasts in the format ``scoring.score`` takes, at each lead time.
    """
    inputs = features[FEATURE_COLUMNS]

    forecasts = []
    for lead_h in LEAD_HOURS:
        table = features[["SID", "init_time"]].assign(
            model=MODEL_NAME,
            lead_h=lead_h,
            lat=features["lat"] + models[(lead_h, "dlat")].predict(inputs),
            lon=features["lon"] + models[(lead_h, "dlon")].predict(inputs),
        )
        forecasts.append(table[FORECAST_COLUMNS])
    return pd.concat(forecasts, ignore_index=True)


def _validation_error_km(
    models: dict[tuple[int, str], xgb.XGBRegressor],
    features: pd.DataFrame,
    truth: pd.DataFrame,
) -> dict[int, float]:
    """Return the mean great-circle error on the validation split at each lead time."""
    errors = {}
    for lead_h in LEAD_HOURS:
        val = _cases(features, truth, "val", lead_h)
        lat = val["lat"] + models[(lead_h, "dlat")].predict(val[FEATURE_COLUMNS])
        lon = val["lon"] + models[(lead_h, "dlon")].predict(val[FEATURE_COLUMNS])
        errors[lead_h] = float(great_circle_km(lat, lon, val["truth_lat"], val["truth_lon"]).mean())
    return errors


def search(
    features: pd.DataFrame,
    truth: pd.DataFrame,
    device: str | None = None,
) -> tuple[dict[tuple[int, str], xgb.XGBRegressor], pd.DataFrame]:
    """Try each combination of settings and keep the one best on validation.

    One combination is chosen for all six models, not one per model. Validation is only
    27 storms, so giving each model its own choice would tune harder to those storms.

    A combination's score is great-circle km, the error the project cares about, rather
    than the RMSE in degrees that each model trains on. Each lead time's error is first
    divided by the best any combination reached at that lead. Otherwise the large 72 h
    errors would decide everything: this way, 5% worse at 24 h counts the same as 5%
    worse at 72 h.

    Picking on validation makes the validation score optimistic. Of 20 combinations,
    the winner is partly the best and partly the luckiest on these particular storms,
    and early stopping has also looked at them. docs/components.md measures the effect.
    The fair number comes later, from the test split: it played no part in any of these
    choices, and is scored once.

    Args:
        features: the table from ``features.build_features``.
        truth: the table from ``samples.make_truth``.
        device: "cpu" or "cuda". Leave it out to use ``pick_device``.

    Returns:
        The best combination's models, and a table of every combination tried with its
        validation error at each lead time, best first.
    """
    device = device or pick_device()
    print(f"Searching {N_COMBINATIONS} combinations on {device}")

    attempts = []
    results = []
    for number, settings in enumerate(sample_settings()):
        models = fit_models(features, truth, settings, device)
        errors = _validation_error_km(models, features, truth)
        attempts.append(models)
        results.append({**settings, **{f"km_{lead_h}h": km for lead_h, km in errors.items()}})
        print(f"  {number + 1:>2} of {N_COMBINATIONS}: {settings}")

    table = pd.DataFrame(results)
    error_columns = [f"km_{lead_h}h" for lead_h in LEAD_HOURS]
    table["relative_error"] = (table[error_columns] / table[error_columns].min()).mean(axis=1)

    best = table["relative_error"].idxmin()
    return attempts[best], table.sort_values("relative_error").reset_index(drop=True)


def best_settings(search_table: pd.DataFrame) -> dict:
    """Return the winning settings from ``search``'s table, ready for ``fit_models``.

    A table row holds every value as a float, so each is turned back into the type
    SEARCH_SPACE gives it. XGBoost refuses a tree depth of 4.0.
    """
    winner = search_table.iloc[0]
    return {name: type(values[0])(winner[name]) for name, values in SEARCH_SPACE.items()}


def tree_counts(models: dict[tuple[int, str], xgb.XGBRegressor]) -> pd.DataFrame:
    """Return how many trees early stopping kept, by lead time and component."""
    counts = pd.Series({key: model.best_iteration + 1 for key, model in models.items()})
    counts.index.names = ["lead_h", "component"]
    return counts.unstack("component")


def gain_shares(models: dict[tuple[int, str], xgb.XGBRegressor]) -> pd.DataFrame:
    """Return each input's share of the total gain, by lead time and overall.

    Gain is how much a tree's error fell at each split on that input, added up. It's
    counted over the trees a model really uses. A model keeps the trees built after its
    best point, and scikit-learn-style ``feature_importances_`` would count those too.

    Args:
        models: the result of ``fit_models`` or ``load_models``.

    Returns:
        One row per input, largest first, with one column per lead time (the average of
        dlat and dlon) and "mean" across all six models. Each column adds up to 1.
    """
    shares = {}
    for key, model in models.items():
        kept_trees = model.get_booster()[: model.best_iteration + 1]
        gain = pd.Series(kept_trees.get_score(importance_type="total_gain"))
        shares[key] = gain.reindex(FEATURE_COLUMNS, fill_value=0.0) / gain.sum()

    by_model = pd.DataFrame(shares)
    table = by_model.T.groupby(level=0).mean().T
    table.columns = [f"+{lead_h} h" for lead_h in table.columns]
    table["mean"] = by_model.mean(axis=1)
    return table.sort_values("mean", ascending=False)


def _model_path(directory: Path, lead_h: int, component: str) -> Path:
    return directory / f"lead{lead_h:02d}_{component}.json"


def save_models(
    models: dict[tuple[int, str], xgb.XGBRegressor],
    search_table: pd.DataFrame,
    directory: str | Path | None = None,
) -> None:
    """Write the fitted models and the search results, creating the folder if needed.

    The models go in XGBoost's own JSON format, which keeps each model's best tree count
    and input names, and can be read by other XGBoost versions.

    Args:
        models: the result of ``search``.
        search_table: the table from ``search``, kept as a record of what was tried.
        directory: where to write. Leave it out to use ``config.xgboost_dir``.
    """
    directory = Path(directory) if directory else config.xgboost_dir()
    directory.mkdir(parents=True, exist_ok=True)
    for (lead_h, component), model in models.items():
        model.save_model(_model_path(directory, lead_h, component))
    search_table.to_csv(directory / "search_results.csv", index=False)
    print(f"Saved {len(models)} models to {directory}")


def load_models(directory: str | Path | None = None) -> dict[tuple[int, str], xgb.XGBRegressor]:
    """Read fitted models back, so forecasts don't need retraining.

    Args:
        directory: where they were written. Leave it out to use ``config.xgboost_dir``.

    Returns:
        {(lead_h, component): fitted model}, to pass to ``predict_xgboost``.
    """
    directory = Path(directory) if directory else config.xgboost_dir()
    models = {}
    for lead_h in LEAD_HOURS:
        for component in COMPONENTS:
            model = xgb.XGBRegressor()
            model.load_model(_model_path(directory, lead_h, component))
            models[(lead_h, component)] = model
    return models
