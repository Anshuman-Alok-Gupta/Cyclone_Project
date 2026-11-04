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

There are two ways through this file. Notebook 03 uses the older one, on the train and
validation split: ``search`` and ``fit_models``. Choosing a model now uses cross-validation:
``search_cv`` and ``fit_fixed``, with one fit per block of seasons.

Test is never read in either, so it stays a fair exam for the end of Project 1.

There are also two targets. Plain XGBoost predicts the change in position. xgb_residual
predicts CLIPER's error and adds that to CLIPER's forecast, so it starts from a sensible
answer and only has to learn what a straight line can't.

Under cross-validation and in the final models, a forecast is the average of several
seeds. ``fit_seeds`` fits one set of six models per seed and ``predict_seeds`` averages
them, so the number being compared isn't partly the luck of one seed.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import ParameterSampler

from cyclone import config, cv
from cyclone.baselines import fit_cliper_on, predict_cliper
from cyclone.features import FEATURE_COLUMNS
from cyclone.folds import development
from cyclone.samples import LEAD_HOURS, with_truth_at
from cyclone.scoring import FORECAST_COLUMNS, great_circle_km, score

MODEL_NAME = "xgboost"
RESIDUAL_NAME = "xgb_residual"
COMPONENTS = ("dlat", "dlon")

# Fixes both the choice of settings to try and the random parts of training, so the
# same data and the same device give the same models.
SEED = 42

# The seeds averaged for a measured or final forecast. Training has a random part here:
# subsample gives each tree a random share of the rows. One seed's forecast is partly that
# luck, and consensus's skill at +72 h moved between 2.1 and 2.4% from seed to seed.
# Averaging cancels much of that out. Averaging forecasts, rather than picking a good
# seed, gives the luck nowhere to hide.
#
# The settings search stays on SEED alone, since it fits 20 combinations. A setting chosen
# with one seed is then measured with all five.
SEEDS = (42, 43, 44, 45, 46)

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

# Cross-validation can't stop early, so the number of trees is searched like any other
# setting. Early stopping watches a set of storms to decide when to stop. The only storms
# left over in a fold are the held-out block, and if those chose the tree count, the
# block's own answers would have shaped the forecast being scored. Holding back some of
# the learning blocks instead would cost training storms, and give each fold and each of
# the six models a different count, leaving no single setting to refit with at the end.
#
# With a slower learning rate a model needs more trees, so the range is wide enough to
# suit all three rates. One count is used for all six models, like the other settings.
CV_SEARCH_SPACE = {**SEARCH_SPACE, "n_estimators": [50, 100, 200, 400]}


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


def sample_cv_settings() -> list[dict]:
    """Return the combinations cross-validation will try, always the same ones."""
    return list(ParameterSampler(CV_SEARCH_SPACE, n_iter=N_COMBINATIONS, random_state=SEED))


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


def _start_points(
    rows: pd.DataFrame,
    lead_h: int,
    base: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the latitude and longitude a model's output is added to, one per row.

    Plain XGBoost adds its change in position to where the storm is at t0. xgb_residual
    adds its correction to CLIPER's forecast. Everything else is the same, so both use
    this one function.

    Args:
        rows: the start times.
        lead_h: the lead time.
        base: another model's forecasts to correct, or None to start from t0.
    """
    if base is None:
        return rows["lat"].to_numpy(), rows["lon"].to_numpy()

    at_lead = base.loc[base["lead_h"] == lead_h, ["SID", "init_time", "lat", "lon"]]
    start = rows[["SID", "init_time"]].merge(
        at_lead, on=["SID", "init_time"], how="left", validate="1:1"
    )
    return start["lat"].to_numpy(), start["lon"].to_numpy()


def fit_fixed(
    rows: pd.DataFrame,
    truth: pd.DataFrame,
    settings: dict,
    device: str,
    base: pd.DataFrame | None = None,
    seed: int = SEED,
) -> dict[tuple[int, str], xgb.XGBRegressor]:
    """Fit the six models on exactly the rows given, with no early stopping.

    Every tree in settings["n_estimators"] is kept. Nothing here watches other storms,
    so what a model learns comes from these rows alone.

    Args:
        rows: the start times to learn from.
        truth: the table from ``samples.make_truth``.
        settings: one combination from ``sample_cv_settings``.
        device: "cpu" or "cuda", from ``pick_device``.
        base: another model's forecasts for these rows. If given, the target is the
            truth minus that forecast, rather than the truth minus the position at t0.
        seed: fixes the random part of training.

    Returns:
        {(lead_h, component): fitted model}, to pass to ``predict_xgboost``.
    """
    models = {}
    for lead_h in LEAD_HOURS:
        cases = with_truth_at(rows, truth, lead_h)
        start_lat, start_lon = _start_points(cases, lead_h, base)
        targets = {
            "dlat": cases["truth_lat"].to_numpy() - start_lat,
            "dlon": cases["truth_lon"].to_numpy() - start_lon,
        }
        for component in COMPONENTS:
            model = xgb.XGBRegressor(
                **settings,
                objective="reg:squarederror",
                tree_method="hist",
                device=device,
                random_state=seed,
            )
            model.fit(cases[FEATURE_COLUMNS], targets[component])
            models[(lead_h, component)] = model
    return models


def fit_seeds(
    rows: pd.DataFrame,
    truth: pd.DataFrame,
    settings: dict,
    device: str,
    base: pd.DataFrame | None = None,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[int, dict[tuple[int, str], xgb.XGBRegressor]]:
    """Fit one set of six models per seed, on the same rows and settings.

    Args:
        rows: the start times to learn from.
        truth: the table from ``samples.make_truth``.
        settings: one combination from ``sample_cv_settings``.
        device: "cpu" or "cuda", from ``pick_device``.
        base: as in ``fit_fixed``.
        seeds: the seeds to fit.

    Returns:
        {seed: models}, to pass to ``predict_seeds``.
    """
    return {seed: fit_fixed(rows, truth, settings, device, base, seed) for seed in seeds}


def predict_xgboost(
    models: dict[tuple[int, str], xgb.XGBRegressor],
    features: pd.DataFrame,
    name: str = MODEL_NAME,
    base: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return XGBoost forecasts for every row of features.

    Passing a table rather than bare numbers lets XGBoost check the input names, so a
    saved model refuses inputs that have changed since it was trained.

    Args:
        models: the result of ``fit_models``, ``fit_fixed`` or ``load_models``.
        features: the start times to forecast. Pass only the split being scored.
        name: the name to give these forecasts in the model column.
        base: the forecasts the models were trained to correct, if any.

    Returns:
        Forecasts in the format ``scoring.score`` takes, at each lead time.
    """
    inputs = features[FEATURE_COLUMNS]

    forecasts = []
    for lead_h in LEAD_HOURS:
        start_lat, start_lon = _start_points(features, lead_h, base)
        table = features[["SID", "init_time"]].assign(
            model=name,
            lead_h=lead_h,
            lat=start_lat + models[(lead_h, "dlat")].predict(inputs),
            lon=start_lon + models[(lead_h, "dlon")].predict(inputs),
        )
        forecasts.append(table[FORECAST_COLUMNS])
    return pd.concat(forecasts, ignore_index=True)


def predict_seeds(
    fitted: dict[int, dict[tuple[int, str], xgb.XGBRegressor]],
    features: pd.DataFrame,
    name: str = MODEL_NAME,
    base: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the average of each seed's forecast positions.

    Latitude and longitude are averaged separately, as in ``consensus.consensus``.

    Args:
        fitted: the result of ``fit_seeds`` or ``fit_final``.
        features: the start times to forecast. Pass only the split being scored.
        name: the name to give these forecasts in the model column.
        base: the forecasts the models were trained to correct, if any.

    Returns:
        Forecasts in the format ``scoring.score`` takes, at each lead time.
    """
    # Each run lists the same cases in the same order, so they can be averaged by position.
    runs = [predict_xgboost(models, features, name, base) for models in fitted.values()]
    return runs[0].assign(
        lat=np.mean([run["lat"] for run in runs], axis=0),
        lon=np.mean([run["lon"] for run in runs], axis=0),
    )


def xgboost_forecaster(
    settings: dict,
    device: str,
    seeds: tuple[int, ...] = (SEED,),
) -> cv.Forecaster:
    """Return plain XGBoost with these settings, in the form ``cv.out_of_fold`` takes.

    One seed is the default because that's what the search fits. Pass ``SEEDS`` to measure
    a model.
    """

    def forecast(learning: pd.DataFrame, truth: pd.DataFrame, held_out: pd.DataFrame) -> pd.DataFrame:
        return predict_seeds(fit_seeds(learning, truth, settings, device, seeds=seeds), held_out)

    return forecast


def residual_forecaster(
    settings: dict,
    device: str,
    seeds: tuple[int, ...] = (SEED,),
) -> cv.Forecaster:
    """Return xgb_residual with these settings, in the form ``cv.out_of_fold`` takes.

    CLIPER is fitted inside this function, on the learning rows only. Using CLIPER's
    forecasts from a fit on all the storms would carry the held-out block's answers into
    the fold, through CLIPER.

    On the learning rows CLIPER is forecasting storms it was fitted on, so its errors
    there are a little smaller than on new storms. With only eight inputs per lead time
    and about two thousand rows, that difference is very small.

    CLIPER has no random part, so every seed corrects the same CLIPER forecast.
    """

    def forecast(learning: pd.DataFrame, truth: pd.DataFrame, held_out: pd.DataFrame) -> pd.DataFrame:
        cliper = fit_cliper_on(learning, truth)
        learning_base = predict_cliper(cliper, learning)
        fitted = fit_seeds(learning, truth, settings, device, base=learning_base, seeds=seeds)
        return predict_seeds(fitted, held_out, RESIDUAL_NAME, base=predict_cliper(cliper, held_out))

    return forecast


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
    and early stopping has also looked at them. docs/components.md measures the effect,
    which is why ``search_cv`` replaced this for choosing models.

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

    table = _with_relative_error(pd.DataFrame(results))
    best = table["relative_error"].idxmin()
    return attempts[best], table.sort_values("relative_error").reset_index(drop=True)


def _with_relative_error(table: pd.DataFrame) -> pd.DataFrame:
    """Add each combination's error relative to the best at each lead, averaged over leads."""
    error_columns = [f"km_{lead_h}h" for lead_h in LEAD_HOURS]
    return table.assign(
        relative_error=(table[error_columns] / table[error_columns].min()).mean(axis=1)
    )


def search_cv(
    rows: pd.DataFrame,
    truth: pd.DataFrame,
    make_forecaster: Callable[[dict, str], cv.Forecaster] = xgboost_forecaster,
    device: str | None = None,
    checkpoint: str | Path | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    """Try each combination under cross-validation and rank them, best first.

    Each combination forecasts every development start time out of fold, and is judged
    on the mean error over all those cases. So every case counts once, and the score is
    the same one models are compared on later. The relative-error rule is the same as in
    ``search``: 5% worse at 24 h counts the same as 5% worse at 72 h.

    This is much harder to fool than one validation split, since the winner has to do
    well across all the storms rather than 27 of them. It still isn't free: the winner
    is partly the luckiest of 20, so its score leans slightly optimistic.

    A search takes minutes on a CPU, so with ``checkpoint`` set, the results so far are
    written after every combination, and a dropped session loses one combination at
    most. Resuming has to be asked for. If it happened by itself, rerunning after a
    change to the data or inputs would quietly reuse old results.

    Args:
        rows: the table from ``cv.model_rows``.
        truth: the table from ``samples.make_truth``.
        make_forecaster: ``xgboost_forecaster`` or ``residual_forecaster``.
        device: "cpu" or "cuda". Leave it out to use ``pick_device``.
        checkpoint: a CSV file to write progress to, in the data directory.
        resume: skip the combinations already in the checkpoint file.

    Returns:
        Every combination tried with its error at each lead time, best first.
    """
    device = device or pick_device()
    print(f"Searching {N_COMBINATIONS} combinations on {device}, 5 folds each")

    checkpoint = Path(checkpoint) if checkpoint else None
    if checkpoint:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

    results = []
    if resume and checkpoint and checkpoint.exists():
        results = pd.read_csv(checkpoint).to_dict("records")
        print(f"  Resuming: {len(results)} combinations already done")
    finished = [{name: row[name] for name in CV_SEARCH_SPACE} for row in results]

    for number, settings in enumerate(sample_cv_settings()):
        if settings in finished:
            continue
        forecasts = cv.out_of_fold(rows, truth, make_forecaster(settings, device))
        errors = score(forecasts, truth, report=False).groupby("lead_h")["error_km"].mean()
        results.append({**settings, **{f"km_{lead_h}h": errors[lead_h] for lead_h in LEAD_HOURS}})
        if checkpoint:
            pd.DataFrame(results).to_csv(checkpoint, index=False)
        print(f"  {number + 1:>2} of {N_COMBINATIONS}: {settings}")

    table = _with_relative_error(pd.DataFrame(results))
    return table.sort_values("relative_error").reset_index(drop=True)


def fit_final(
    rows: pd.DataFrame,
    truth: pd.DataFrame,
    settings: dict,
    device: str,
    base: pd.DataFrame | None = None,
) -> dict[int, dict[tuple[int, str], xgb.XGBRegressor]]:
    """Refit chosen settings with every seed, on every development season.

    Cross-validation's models each saw four blocks, and their job was to measure the
    settings. Once the settings are chosen, using all five blocks gives the final model
    more storms to learn from. It can't be scored on these storms, since it has seen all
    of them, so the cross-validated score stands in for it until the test split.

    All the seeds are fitted, because the model that was measured is the average of them.

    The filter is inside this function, so passing the full table can't reach test.

    Args:
        rows: a table with a "fold" column, such as the one from ``cv.model_rows``.
        truth: the table from ``samples.make_truth``.
        settings: the chosen settings, from ``best_settings``.
        device: "cpu" or "cuda", from ``pick_device``.
        base: as in ``fit_fixed``, for xgb_residual.

    Returns:
        {seed: models}, to pass to ``predict_seeds``.
    """
    return fit_seeds(development(rows), truth, settings, device, base)


def best_settings(search_table: pd.DataFrame) -> dict:
    """Return the winning settings from a search table, ready to fit with.

    It reads a table from ``search`` or ``search_cv``. A table row holds every value as a
    float, so each is turned back into the type the search space gives it. XGBoost
    refuses a tree depth of 4.0.
    """
    winner = search_table.iloc[0]
    return {
        name: type(values[0])(winner[name])
        for name, values in CV_SEARCH_SPACE.items()
        if name in search_table.columns
    }


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


def _seed_dir(directory: Path, seed: int) -> Path:
    return directory / f"seed{seed}"


def save_final(
    fitted: dict[int, dict[tuple[int, str], xgb.XGBRegressor]],
    search_table: pd.DataFrame,
    directory: str | Path | None = None,
) -> None:
    """Write every seed's models, one folder per seed.

    Args:
        fitted: the result of ``fit_final``.
        search_table: the table from ``search_cv``, kept as a record of what was tried.
        directory: where to write. Leave it out to use ``config.xgboost_final_dir``.
    """
    directory = Path(directory) if directory else config.xgboost_final_dir()
    for seed, models in fitted.items():
        save_models(models, search_table, _seed_dir(directory, seed))


def load_final(
    directory: str | Path | None = None,
) -> dict[int, dict[tuple[int, str], xgb.XGBRegressor]]:
    """Read every seed's models back. A missing seed stops the run.

    A final forecast from four seeds would look normal and quietly not be the model that
    was measured, so all of them have to be there.

    Args:
        directory: where they were written. Leave it out to use ``config.xgboost_final_dir``.

    Returns:
        {seed: models}, to pass to ``predict_seeds``.
    """
    directory = Path(directory) if directory else config.xgboost_final_dir()
    return {seed: load_models(_seed_dir(directory, seed)) for seed in SEEDS}
