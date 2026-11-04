# Components

What each module in `src/cyclone/` does and why. The pipeline runs in this order:

```
config → io.load_ibtracs → filters.clean_ni → splits.assign_split → checks.check_expected → io.save_parquet
```

Forecasting then starts from the saved table:

```
samples.make_truth, samples.make_init_times → checks.check_samples → baselines → scoring.score → scoring.summarize
```

XGBoost adds one step, and is scored the same way:

```
samples.make_init_times → features.build_features → xgb_model → scoring.score → scoring.summarize
```

Model choice runs each model through cross-validation, and is scored the same way again:

```
folds → checks.check_folds → cv.model_rows → cv.out_of_fold → scoring.score → cv.bootstrap_skill
```

The project decisions behind these modules are in `CLAUDE.md`.

## config.py

**What it does:** the one place that says where data lives.

**Inputs and outputs:** reads the `CYCLONE_DATA_DIR` environment variable. `data_dir()`, `ibtracs_path(basin)`, `ni_clean_path()`, `xgboost_dir()` and `xgboost_final_dir()` return paths.

**Why it's built this way:**
- Changing one variable moves all the data between your laptop and Colab, with no code changes.
- The default is `<repo>/data`, worked out from the file's own location. A literal `./data` would break in Jupyter, which starts in `notebooks/`.
- That default needs the `pip install -e .` install, where the code stays in the repo. On a plain `pip install .`, set the variable.
- Paths are functions, not constants, so a Colab cell can set the variable after importing.

**Reused by:** every later step. Checkpoint paths belong here too.

## io.py

**What it does:** reads a raw IBTrACS CSV and writes tables to Parquet. It doesn't filter.

**Inputs and outputs:**
- `load_ibtracs(path)` takes a CSV path and returns every row and all 174 columns.
- `save_parquet(df, path)` writes a table, creating the folder if needed.

**Why it's built this way:**
- Skips the units row and handles the `"NA"` basin trap (see `CLAUDE.md`).
- Text columns are listed by hand. If pandas guesses, `USA_RECORD` is text in NI but a float in WP, because it's empty there.
- Numeric conversion is strict, so an unexpected value crashes instead of silently becoming `NaN`.
- Parquet keeps column types, so nothing has to be parsed again.

**Reused by:** the WP pretraining experiment, unchanged. Project 3 uses it for the IMD sensitivity check, since the raw table still has the `NEWDELHI_*` columns.

## filters.py

**What it does:** the five row filters from `CLAUDE.md`, plus `clean_ni`, which applies them in order.

**Inputs and outputs:**
- Each filter takes a table and returns the rows it keeps.
- `clean_ni(df)` takes the raw table and returns 8 columns: `SID`, `NAME`, `SEASON`, `ISO_TIME`, `USA_LAT`, `USA_LON`, `USA_WIND`, `USA_PRES`.

**Why it's built this way:**
- Small separate functions can be reused one at a time and read like the list in `CLAUDE.md`.
- `clean_ni` prints counts after every step, so a changed total can be traced to one filter.
- Only columns allowed as inputs or scoring data are kept. `STORM_SPEED`, `STORM_DIR` and `DIST2LAND` were removed for the reasons under "Before building features".

| After filter | Rows | Storms |
|---|---:|---:|
| raw | 62,908 | 1,861 |
| `main_track` | 62,455 | 1,849 |
| `ni_basin` | 57,587 | 1,849 |
| `synoptic_hours` | 28,857 | 1,807 |
| `seasons` | 7,625 | 449 |
| `original_jtwc_fix` | 5,694 | 258 |

**Reused by:** WP pretraining, which needs every filter except `ni_basin`. Project 3 scores on the same storms and times.

## splits.py

**What it does:** labels each row `train`, `val` or `test` by season.

**Inputs and outputs:** `assign_split(df)` takes any table with `SEASON` and returns a copy with a `split` column. Seasons after 2024 are labelled `excluded`.

**Why it's built this way:**
- Rows from one storm are nearly identical, so a random split would put the same storm on both sides and inflate scores.
- It uses `SEASON`, not calendar year, so the two storms that cross New Year stay in one split.
- The boundaries are three constants that nothing else repeats.

**Reused by:** notebooks 02 and 03, which keep the train and validation split so they still run. Model choice now uses `folds.py`. Project 3 reports on the same test years so its scores compare directly.

## folds.py

**What it does:** cuts the development seasons, 1980–2017, into the five blocks used for cross-validation.

**Inputs and outputs:**
- `FOLD_SEASONS` and `FOLD_NAMES` define the blocks: 1980–87, 1988–95, 1996–2003, 2004–10 and 2011–17.
- `assign_fold(df)` takes any table with `SEASON` and returns a copy with a `fold` column. Seasons outside the blocks get `NOT_IN_A_FOLD`.
- `add_fold(table, df)` does the same for a table with `SID`, such as the start times, looking up each storm's season in the cleaned table.
- `development(table)` returns only the rows in a block.

**Why it's built this way:**
- It's a separate file because it does a different job from `splits.py`, and `splits.py` now only serves the older notebooks.
- Blocks are whole, consecutive seasons, for the same reason the main split is by year: rows from one storm are nearly identical, and storms in one season share that season's weather.
- A model can learn from seasons later than the ones it forecasts. That's fine for comparing models, because no storm's own future is involved. It's the one way this differs from the final test, which stays last in time.
- It uses `SEASON`, not calendar year, so a storm that crosses New Year stays in one block.
- `add_fold` stops the run on a storm in two blocks rather than picking one.
- `development` keeps named blocks rather than dropping `NOT_IN_A_FOLD`, so a row with no fold at all stays out too.
- The test seasons are in no block, so nothing that loops over the blocks can reach them.

**Tests:** `tests/test_folds.py`. The rules are tested on a small made-up table, so they run anywhere. Tests on the real table skip if it hasn't been built.

**Reused by:** every model comparison from here on, including the LSTM.

## checks.py

**What it does:** stops the run if the cleaned table or the forecast samples aren't the expected ones.

**Inputs and outputs:**
- `check_expected(df)` takes the table with its `split` column. It prints a confirmation if all counts match. Otherwise it raises an `AssertionError` listing each difference and the likely causes.
- `check_samples(init_times, truth)` does the same for the start times and for the cases with truth at each lead time.
- `count_cases(init_times, truth)` returns those case counts as `{split: {lead_h: count}}`.
- `check_folds(df, init_times, truth)` does the same for the cross-validation blocks. It also stops the run if a storm is in two blocks or a test season is in any block.
- `check_dist_to_land(df, raw)` compares `features.distance_to_land_km` with IBTrACS `DIST2LAND` and prints the agreement. It stops the run only if the median gap passes 10 km, which would mean a broken calculation.

**Why it's built this way:**
- Wrong data still trains and still scores, so nothing downstream would notice.
- It checks rows and storms separately because they catch different problems: lost points inside a storm, or whole storms moving.
- It uses `raise` rather than `assert`, because `python -O` removes `assert` lines.
- It only checks counts. A table with the right counts but wrong values would pass, which is why the notebook also draws a map.
- `check_samples` counts test cases but never scores them, so the test split stays locked.

| Split | Init times | Cases +24 h | Cases +48 h | Cases +72 h |
|---|---:|---:|---:|---:|
| train | 3,203 | 2,515 | 1,869 | 1,324 |
| val | 542 | 438 | 340 | 256 |
| test | 933 | 749 | 578 | 430 |

| Block | Storms | Init times | Cases +24 h | Cases +48 h | Cases +72 h |
|---|---:|---:|---:|---:|---:|
| 1980–87 | 39 | 598 | 452 | 324 | 218 |
| 1988–95 | 44 | 774 | 617 | 465 | 325 |
| 1996–2003 | 48 | 952 | 774 | 605 | 463 |
| 2004–10 | 43 | 737 | 570 | 413 | 284 |
| 2011–17 | 37 | 684 | 540 | 402 | 290 |
| All | 211 | 3,745 | 2,953 | 2,209 | 1,580 |

- These add up to the train and validation rows of the table above.
- 201 of the 211 storms have a start time, and 194, 171 and 140 have a scored case at +24, +48 and +72 h. The rest are too short.
- The expected counts are stored in block order, not by season, so `checks.py` doesn't repeat the boundaries.

`DIST2LAND` is a sanity check only, never an input. It stays inside `check_dist_to_land`, and the cleaned table doesn't hold it. On the 2,645 rows where the JTWC and combined positions are within 10 km:

| Land mask | Median gap | 90th percentile | Within 10 km | Correlation |
|---|---:|---:|---:|---:|
| All Natural Earth land (what `features.py` uses) | 5.1 km | 343 km | 63.5% | 0.84 |
| Without islands under 5,000 km² (one-off test) | 2.0 km | 7.6 km | 92.7% | 0.99 |

- They agree closely over land (376 rows in both, of 383 each) and near large coasts, but not everywhere.
- The gap comes from the land masks, not the calculation. `DIST2LAND` leaves out small islands that Natural Earth has. At 1.6°N 72.4°E, next to the Maldives, it says 908 km where the nearest Natural Earth land is 95 km away.
- It isn't a simple size cut. Removing Sri Lanka as well makes agreement worse (within 10 km falls to 82.7%), so Sri Lanka is in the IBTrACS mask. With the 5,000 km² cut, the 159 rows still over 25 km apart lie almost entirely in the Arabian Sea around Socotra (3,600 km²), with `DIST2LAND` the smaller value, so Socotra seems to be in it too.
- `features.py` keeps all Natural Earth land, since a model only needs a true distance from JTWC's position, not a match to `DIST2LAND`.

**Reused by:** every run of the notebooks. If the data is re-downloaded on purpose, review the differences and then update the numbers here and in `CLAUDE.md`.

## samples.py

**What it does:** turns the cleaned table into the two tables forecasts work from: where storms really were, and what was known at each start time.

**Inputs and outputs:**
- `make_truth(df)` returns `SID`, `valid_time`, `lat`, `lon`, one row per fix.
- `make_init_times(df)` returns every fix with JTWC fixes at exactly 6, 12, 18 and 24 h earlier: `SID`, `init_time`, `split`, `lat`, `lon`, `wind`, then `lat_m06`, `lon_m06`, `wind_m06` through to `wind_m24`.
- `with_truth_at(init_times, truth, lead_h)` returns the start times that have a true position `lead_h` hours later, with `truth_lat` and `truth_lon` added.

**Why it's built this way:**
- The two tables are separate so that future positions aren't in the table models read. A model can't look ahead by accident if the answer isn't there.
- Past fixes are matched by timestamp, as `CLAUDE.md` requires. Shifting by 4 rows would accept 4,683 start times, and 5 of those have a 12 h gap in their history, so their "24 h ago" is really 30 h ago. Matching gives the correct 4,678.
- Deleting every fix after a storm's midpoint and rebuilding left the history of all 1,910 remaining start times unchanged, which confirms nothing after t0 is used.
- `with_truth_at` is the one place a start time meets a future position, so a future leak has only one place to come from.
- `USA_WIND` is missing for 24 start times, all in train, in 1980. The same 24 rows have no wind at t0, 12 h or 24 h earlier, because they're the two storms with no wind at all. They stay in, since the baselines don't use wind, and `features.py` leaves them blank for XGBoost to handle.
- Past winds are carried alongside past positions because wind change is a model input. A blank wind doesn't remove a row: a past fix only has to exist, with a position, so adding them left all the counts unchanged.

**Reused by:** baselines, XGBoost and the LSTM. Using the same start times is what makes their scores comparable.

## scoring.py

**What it does:** measures how far any forecast position was from the truth, in km, and summarizes that by model and lead time. It imports nothing else from this package.

**Inputs and outputs:**
- `great_circle_km(lat1, lon1, lat2, lon2)` returns the haversine distance, on a sphere of radius 6371.0 km. It takes numbers or whole columns.
- `score(forecasts, truth, report=True)` takes forecasts with `model`, `SID`, `init_time`, `lead_h`, `lat`, `lon` and truth with `SID`, `valid_time`, `lat`, `lon`. It returns every forecast plus `valid_time`, `truth_lat`, `truth_lon` and `error_km`, and prints how many had no truth for each model.
- `homogeneous_cases(scored)` returns the rows for the cases every model has.
- `summarize(scored, homogeneous=True)` returns `count`, `mean_km` and `median_km` by `model` and `lead_h`.
- `skill(summary, reference)` returns each model's percentage improvement in mean error over a reference model, by lead time. Positive is better.

**Why it's built this way:**
- It stands alone so Project 3 can score GraphCast and Pangu with the same code. A test fails if it ever imports anything other than numpy and pandas.
- Great-circle distance is needed because a degree of longitude shrinks away from the equator: 111 km at the equator, 96 km at 30°N.
- `report=False` stops `score` printing, because a settings search scores hundreds of times. It changes nothing else.
- A forecast with no truth stays in the table with a blank `error_km`. Dropping it would hide a model that lost half its cases. Having some is normal, since the storm may have ended or left the basin. `summarize` leaves them out of the averages, because they can't be judged right or wrong.
- A repeated forecast, a forecast with no position and a repeated truth time all stop the run. Each would otherwise change the score with no visible sign.
- `homogeneous=True` keeps only the cases every model has. Otherwise a model could look better just by skipping hard storms. Leave it on when comparing models.
- A model with no scored cases at all still counts towards "every model", so it empties the table rather than quietly dropping out of the comparison.
- Times can arrive as text or in any precision, since AI-model output won't match this project's tables.

**Tests:** `tests/test_scoring.py`. Install pytest with `pip install -e ".[dev]"`, then run `pytest`. It's a dev dependency so Colab and Kaggle never install it.

**Reused by:** every model in this project, and the AI models in Project 3.

## baselines.py

**What it does:** the two simple forecasts every later model has to beat.

**Inputs and outputs:**
- `persistence(init_times)` extends the average motion from t0−12 h to t0 in a straight line, to +24, +48 and +72 h.
- `fit_cliper_on(rows, truth)` fits one linear regression per lead time on exactly the rows it's given, and returns `{lead_h: model}`.
- `fit_cliper(init_times, truth)` does that on the train split, for notebooks 02 and 03.
- `predict_cliper(models, init_times)` returns CLIPER forecasts for the start times it's given.
- `persistence` and `predict_cliper` both return tables in the format `scoring.score` takes.
- `persistence_forecast` and `cliper_forecast` wrap them in the form `cv.out_of_fold` takes.

**Why it's built this way:**
- A score means little alone. "150 km at 24 h" is only useful next to what a simple method gets on the same cases.
- `fit_cliper` filters to the train split itself, so passing the full table can't leak other storms into the fit. Corrupting every non-train position by 30° left the coefficients unchanged.
- CLIPER predicts the change in position, as `CLAUDE.md` requires. Absolute position would make the fit spend its effort rediscovering where the storm already is.
- Its eight inputs are all known at t0: position, motion over the past 12 and 24 h (Δlat and Δlon for each), and day of year as sin and cos. As a plain number, 31 December and 1 January would look a year apart, not a day.
- There's no wind input, so the 24 start times with no wind need no special handling. That's why this is a simplified CLIPER: the classic versions also use intensity.
- Both baselines forecast every start time, even with no truth at that lead time. `scoring.score` then reports those, the same way it would for any model.

Mean error on the validation split, on identical cases:

| Lead time | Cases | Persistence | CLIPER |
|---:|---:|---:|---:|
| +24 h | 438 | 137.6 km | 129.0 km |
| +48 h | 340 | 319.7 km | 289.0 km |
| +72 h | 256 | 509.8 km | 459.1 km |

**Reused by:** every later model, as the reference it has to beat.

## features.py

**What it does:** turns the start-time table into the 19 numbers a model reads.

**Inputs and outputs:**
- `build_features(init_times)` takes the table from `samples.make_init_times` and returns one row per start time, in the same order: `ID_COLUMNS` (`SID`, `init_time`, `split`), then `FEATURE_COLUMNS`.
- `distance_to_land_km(lat, lon)` returns the distance to the nearest coast, and 0 over land.

| Group | Inputs | What it tells the model |
|---|---|---|
| Position | `lat`, `lon` | Storms in one place tend to move one way |
| Motion | `dlat_6h`, `dlon_6h`, `dlat_12h`, `dlon_12h`, `dlat_24h`, `dlon_24h` | Recent movement. 6 h is the latest but noisiest, 24 h the steadiest |
| Motion | `speed_kmh`, `heading_sin`, `heading_cos` | The 12 h movement as speed and direction, which a tree can't easily work out from Δlat and Δlon |
| Motion | `dlat_change`, `dlon_change` | The last 12 h minus the 12 h before: speeding up, slowing down or starting to turn |
| Intensity | `wind`, `dwind_12h`, `dwind_24h` | Strength, and whether it's growing |
| Season | `day_sin`, `day_cos` | Time of year, which changes the winds that steer storms |
| Land | `dist_to_land_km` | How close the coast is |

**Why it's built this way:**
- Inputs come only from `make_init_times`, which holds no position or wind after t0. Each row is calculated from that row alone, with no dataset-wide averages, so another storm or a later fix can't affect it.
- Heading and day of year go in as sin and cos, because both are circles. As plain numbers, 359° and 1°, or 31 December and 1 January, would look far apart when they're neighbours.
- Longitude is scaled by cos(latitude) before taking a heading, because a degree of longitude is shorter away from the equator.
- A storm that hasn't moved has no heading, so sin and cos are left blank rather than defaulting to north. No row is like this today: every start time moved over the past 12 h.
- A missing wind stays `NaN`. When a tree splits on a column, XGBoost tries the missing rows down each side and keeps the better side as that split's default direction. Filling in an average would invent a wind.
- That matters less than it sounds. Only 24 start times have no wind, all in train in 1980. No validation or test row has a blank input, so none of them ever takes a default direction.
- `ID_COLUMNS` stay in the table for matching, but a model reads only `FEATURE_COLUMNS`, so a storm ID or date can't become an input by accident.
- `USA_PRES` is left out: it's missing before 2001, so a model could learn "missing means old storm". `DIST2LAND`, `STORM_SPEED`, `STORM_DIR` and `LANDFALL` are left out for the reasons in `CLAUDE.md`.
- Distance to land is measured along the Earth's surface to points every 0.02° (about 2 km) along Natural Earth's 10 m coast. That's within 0.5 km of a spacing twice as fine, while the file's own points alone can be 7 km off. Land is tested separately, since a coast is equally close from either side.
- cartopy is imported inside `_load_land`, so importing this module for its column lists doesn't load it. The land data loads once per run, in several seconds, and the file is downloaded the first time.

**Tests:** `tests/test_features.py`.
- The leakage test rewrites every position and wind after t0, rebuilds through `make_init_times` and `build_features`, and requires every row up to t0 to be unchanged.
- It catches real leaks: adding a future wind, a future position, or history matched forwards in time each made it fail.
- Its partner test moves one past fix and requires the features to change, because inputs that ignored the track entirely would pass the leakage test too.
- On the real data, corrupting every fix after each storm's midpoint left all 2,031 start times up to the midpoint unchanged, across 242 storms, while all 2,647 later ones changed.

**Reused by:** the LSTM can read the same inputs, under the same leakage test. WP pretraining can use it unchanged, since the land data is global.

## xgb_model.py

**What it does:** fits XGBoost, picks its settings, forecasts, and saves and reloads the models. It has two routes: cross-validation, which is how models are chosen now, and the older train and validation route, which notebook 03 still uses.

**Inputs and outputs:**
- `search_cv(rows, truth, make_forecaster)` tries 20 combinations under cross-validation, and returns a table of them, best first.
- `fit_fixed(rows, truth, settings, device, base=None)` fits the six models on exactly the rows given, keeping every tree. With `base`, the target is the truth minus that forecast.
- `xgboost_forecaster(settings, device)` and `residual_forecaster(settings, device)` return the two models in the form `cv.out_of_fold` takes.
- `fit_final(rows, truth, settings, device)` refits chosen settings on all of 1980–2017.
- `predict_xgboost(models, features, name, base=None)` returns forecasts in the format `scoring.score` takes.
- `search(features, truth)` and `fit_models(features, truth, settings, device)` are the older route, for notebook 03.
- `save_models(models, search_table, directory)` and `load_models(directory)` write and read a folder of models, which is git-ignored. `data/models/xgboost` is rebuilt by notebook 03, and `data/models/xgboost_final` by notebook 04.
- `pick_device()`, `sample_settings()`, `sample_cv_settings()`, `best_settings(search_table)`, `tree_counts(models)` and `gain_shares(models)` are the supporting pieces. `tree_counts` and `gain_shares` need early stopping, so they only work on the older route.

**Why it's built this way:**
- XGBoost builds a chain of small decision trees, each correcting what the trees before it got wrong. Unlike CLIPER's straight line, it can learn rules that only apply in some situations.
- There are six models, one per lead time and component. Lead times are separate because what matters changes with range, and each has its own cases. Components are separate because the tree regressor predicts one number at a time. The cost is that a forecast's points aren't tied together, so a track can wobble.
- It predicts the change in position, as `CLAUDE.md` requires, so its forecasts come out on the same footing as CLIPER's.
- `fit_models` filters to the splits itself, as `fit_cliper` does. Trees learn only from train, validation only stops them adding trees, and test is never read. Overwriting all test inputs and truth left the forecasts identical, while changing train inputs did change them.
- One combination is used for all six models. With so few validation storms, a separate choice per model would tune harder to those storms.
- Combinations are compared in great-circle km, not the RMSE in degrees the trees train on. Each lead's error is divided by the best any combination reached there, so 72 h's larger errors don't decide everything.
- `min_child_weight` is the minimum samples per leaf: with a squared-error target it's exactly a count of rows.
- `pick_device` runs a tiny fit, because seeing a GPU isn't enough. The local MX130 shows up in `nvidia-smi`, and XGBoost still refuses it (`This program was not compiled for SM 50`), so this returns `cpu` there. The GPU path hasn't been run yet, since no GPU here can use it. The search takes about a minute on this laptop's CPU.
- Results are repeatable: the same data on the same device gave identical forecasts across separate runs. A GPU can give slightly different models.
- `gain_shares` counts only the trees a model really uses. A model keeps the 50 trees after its best point, and `feature_importances_` counts those too.
- Models are saved in XGBoost's JSON format, and reloading gives identical forecasts. It keeps the best tree count and the input names, so a saved model refuses inputs that have changed.
- `search` has no checkpoints: it takes about a minute, so a dropped session loses little.

**Under cross-validation:**
- Early stopping can't be used. It watches a set of storms to decide when to stop, and in a fold the only storms left over are the held-out block. If those chose the tree count, the block's own answers would have shaped the forecast being scored.
- So the number of trees is searched, like any other setting. Holding back part of the learning blocks to stop on would cost training storms, and would give each fold and each of the six models a different count, leaving no single setting to refit with at the end.
- One count is used for all six models. Early stopping used to give each its own, and the longer leads stopped sooner, so this may suit some leads better than others.
- `search_cv` judges each combination on the mean error over every out-of-fold case, so every case counts once and the score is the same one models are compared on later. The relative-error rule is unchanged.
- `search_cv` takes about 5 minutes on this laptop's CPU, so with `checkpoint` set it writes its results after every combination. Resuming has to be asked for with `resume=True`. If it happened by itself, a rerun after a change to the data would quietly reuse old results.
- `fit_final` filters to the development seasons itself, so passing the full table can't reach test. It can't be scored on these storms, since it has seen them all, so the cross-validated score stands in for it until the test split.

**xgb_residual** predicts CLIPER's error and adds it to CLIPER's forecast, so it starts from a sensible answer and only has to learn what a straight line can't.
- It's the same code as plain XGBoost with a different starting point. Plain XGBoost adds its output to the position at t0, and `xgb_residual` adds it to CLIPER's forecast. `_start_points` is the only place they differ.
- CLIPER is fitted inside `residual_forecaster`, on the learning rows only. A CLIPER fitted on all the storms would carry the held-out block's answers into the fold.
- On the learning rows, CLIPER is forecasting storms it was fitted on, so its errors there are a little smaller than on new storms. With eight inputs per lead and about two thousand rows, that difference is very small.
- It gets its own search with the same 20 combinations. It has a much smaller target than plain XGBoost, so borrowing plain XGBoost's settings would handicap it.

Results from the older route follow. Notebook 04's cross-validated results replace them for choosing a model.

Mean error on the validation split, on identical cases, with the winning settings (`max_depth` 4, `learning_rate` 0.05, `min_child_weight` 40, `subsample` 0.6):

| Lead time | Cases | Persistence | CLIPER | XGBoost | Skill vs CLIPER |
|---:|---:|---:|---:|---:|---:|
| +24 h | 438 | 137.6 km | 129.0 km | 118.8 km | 7.9% |
| +48 h | 340 | 319.7 km | 289.0 km | 275.4 km | 4.7% |
| +72 h | 256 | 509.8 km | 459.1 km | 422.4 km | 8.0% |

**How far to trust that skill.** These are one-off tests, on validation only:

| Test | +24 h | +48 h | +72 h |
|---|---:|---:|---:|
| Skill as reported | 7.9% | 4.7% | 8.0% |
| 90% range, resampling whole storms 4,000 times | 4.3 to 11.2% | −1.5 to 10.3% | −1.6 to 16.7% |
| Pick on half the storms, score on those | 8.4% | 7.2% | 9.1% |
| Pick on half the storms, score on the other half | 5.6% | 0.0% | 2.9% |

- The last two rows pool four random halvings, each used both ways round. Early stopping also used only the picking half.
- The gap between them is the optimism from choosing on validation, and it's not small. The fresh row leans a little low, because only half the storms guided the choice.
- So XGBoost's lead at +24 h looks real, while at +48 and +72 h it may be small or nothing. The test split will settle that.
- Shuffling the train targets sent validation error to 384, 628 and 883 km, so the skill comes from real signal and not from a bug.

**Other one-off tests:**

| Mean validation error (km) | +24 h | +48 h | +72 h |
|---:|---:|---:|---:|
| All inputs | 118.8 | 275.4 | 422.4 |
| Without 6 h motion (54% of the gain) | 125.2 | 278.6 | 430.4 |
| Without any motion | 205.6 | 391.9 | 542.7 |
| Land distance without islands under 5,000 km² | 119.1 | 276.8 | 423.2 |
| Without distance to land | 118.7 | 272.5 | 423.7 |

- Gain doesn't mean need. The two inputs with the most gain cost only 3 to 8 km to lose, because the 12 and 24 h windows overlap with them.
- The choice of land mask makes almost no difference, even though it changes the distance for 1,613 of the 4,678 start times.
- Distance to land itself adds nothing measurable on validation. It stays because it was asked for, and it may matter more for intensity than for track.

**Reused by:** notebooks 03 and 04. The LSTM has to go through the same cross-validation, on the same cases.

## consensus.py

**What it does:** averages the forecast positions of several models into one forecast.

**Inputs and outputs:** `consensus(forecasts, members)` takes forecasts in the scoring format and the names of the models to average. It returns one forecast per case, named `consensus`, and keeps extra columns such as `fold`.

**Why it's built this way:**
- Two forecasts that are wrong in different ways can partly cancel out. If one is too far north and the other too far south, the point between them is closer than either. Real forecast centres lean on this heavily.
- It has nothing to fit, so it adds no settings that could be tuned to the storms. It's an even split, not a weighted one, because a fitted weight would be one more thing to choose.
- Each member's forecasts are already out of fold, so their average is too.
- Latitude and longitude are averaged separately. That isn't exactly the midpoint on a sphere, but the difference is tiny between points a few hundred km apart in the tropics, and this basin is nowhere near the date line.
- A case missing a member stops the run. Otherwise it would quietly become the other member's forecast under this name.
- It reads only the scoring format, so it imports nothing but the column names from `scoring.py`.

**Tests:** `tests/test_consensus.py`.

**Reused by:** notebook 04, and Project 3 can use it to average AI models.

## cv.py

**What it does:** runs any model through cross-validation, and measures how far to trust the result.

**Inputs and outputs:**
- `model_rows(init_times, features, df)` returns one table with everything any model reads, for development storms only, with a `fold` column.
- `out_of_fold(rows, truth, forecaster)` returns forecasts for every development start time, each from a model that never saw that storm, in the format `scoring.score` takes.
- A forecaster is one function: `forecaster(learning, learning_truth, held_out)` returns forecasts for `held_out`.
- `bootstrap_skill(scored)` returns each model's skill against CLIPER with a 95% interval.
- `skill_by_fold(scored)` returns skill block by block.
- `win_rate(scored)` returns the percentage of cases where each model is closer than CLIPER.
- `storm_gaps(scored, model)` returns each storm's mean error for a model minus CLIPER's, sorted.

**Why it's built this way:**
- A single validation split rests on whichever storms fell into it. Here every development storm is forecast once, so a comparison rests on all of them.
- `out_of_fold` removes the held-out block's storms from the truth table before the model gets it. So a model can't learn from a block's answers whatever its code does, because they aren't in what it's handed.
- Test storms are kept out twice: `model_rows` doesn't include them, and `out_of_fold` ignores any row outside a block, so a caller passing the full table still can't reach them.
- Every model comes as the same kind of function, so all of them take exactly the same route, and the LSTM can too.
- `model_rows` refuses tables that don't hold the same storms and times in the same order, because each storm would silently get another storm's inputs.
- The bootstrap redraws whole storms, not rows. Forecasts from one storm, six hours apart, are nearly the same forecast of nearly the same situation, so 20 forecasts from one storm are much closer to 1 piece of evidence than to 20. Redrawing rows would count them as 20 and give an interval that looks far more certain than it is.
- Each lead time redraws from the storms with a case at that lead, and every model is scored on the same redraws. So the interval measures the gap between models, not how hard the storms were.
- The interval covers only the luck of which storms occurred. It doesn't cover the choice of settings, or storms in one season being alike.
- Skill per block and win rate each catch something an average hides: a model that wins on the strength of one block, or through a few large wins.

**Tests:** `tests/test_cv.py`.
- The leakage test moves every truth position in one block, reruns, and requires that block's forecasts to be identical, for each of the five blocks.
- Its partner test requires the other four blocks to change, because a model that ignored truth altogether would pass the leakage test too.
- It catches real leaks. Letting a model learn from all five blocks, handing it the full truth table, and removing the block filter each made the tests fail.
- A storm-level test repeats each storm's errors at 20 start times and requires the interval to stay the same, which a bootstrap of rows would fail.

**Reused by:** every model comparison from here on, including the LSTM.

## plotting.py

**What it does:** draws storm tracks, forecasts and error charts as visual checks.

**Inputs and outputs:**
- `plot_track(df, sid=...)` or `plot_track(df, name=..., season=...)` takes a table and returns a map, with dots coloured by wind.
- `plot_error_by_lead(summary, title)` takes the table from `scoring.summarize` and returns a chart of mean error against lead time.
- `plot_forecasts(truth, forecasts, sid, init_time, title)` returns a map of one storm's true track with each model's forecast from one start time. `labels_below=True` moves the labels below the track, for forecasts that run through the space above it.
- `plot_feature_importance(shares, title)` takes the table from `xgb_model.gain_shares` and returns ranked bars of each input's share of gain.
- `plot_storm_gaps(gaps, model, title)` takes the table from `cv.storm_gaps` and returns one sorted bar per storm.

**Why it's built this way:**
- A map shows position errors that counts can't.
- The colour scale is fixed at 10–150 kt, so a shade means the same wind on every map.
- Points with no wind get hollow markers. Without them, scatter would silently drop those positions.
- Name lookup handles joined names like `HIKAA:HIKKA`. If a name matches several storms, as `UNNAMED` often does, it lists their SIDs instead of guessing.
- It accepts an existing `ax`, so several tracks can share one figure.
- cartopy downloads coastline files the first time it runs, so that first plot needs internet.
- Each model has one fixed colour in `MODEL_COLORS`, so it looks the same on every chart. A new model needs an entry there and in `MODEL_LABELS`.
- `plot_forecasts` draws the truth in a dark neutral, not the wind colours, because those are blue and would be hard to tell from the blue persistence line.
- Its lead-time labels stand off the track on leader lines. Placed beside a true position, they land next to a forecast dot and appear to name that dot.
- The error chart's y axis starts at 0 km, so the gap between lines reads as its true share of the error.
- Importance bars are one colour, XGBoost's, because they're one model's numbers. Colouring them by size would only repeat what their length shows.
- Only bars of 5% or more are labelled. A number on every bar would hide the few that matter, and the notebook's table has the rest.
- The storm chart's bars are sorted, so it shows whether a lead comes from most storms or a few big ones. They're blue or red for which side of zero they fall on, not in model colours, because yellow and pink fail the palette check beside CLIPER's orange. A pair of opposites also reads as "better" and "worse" for any model.
- Five model colours pass the check for lines and bars side by side, but not for every pair at once. So they're safe for a chart like the error-by-lead lines, with labels, and shouldn't all share a scatter plot.

**Reused by:** `plot_forecasts` and `plot_error_by_lead` work for any model in the scoring format, including the AI-model tracks in Project 3.

## Before building features

The rules these lead to are in "Model input rules" in `CLAUDE.md`.

- **`STORM_SPEED` and `STORM_DIR` were dropped from the cleaned table.**
  - They don't come from JTWC positions. They match a centred difference of the combined `LAT`/`LON` over the points 3 h either side (96.7% within 1 kt, compared with 66.9% for the same calculation on `USA_LAT`/`USA_LON`, and 56% for a backward-only one).
  - So they use the source `CLAUDE.md` rules out, and they use a position 3 h after the forecast start, which a real forecast wouldn't have.
  - Calculate motion from `USA_LAT`/`USA_LON` at the current and previous fixes instead.
- **`DIST2LAND` was dropped too, and is recomputed from `USA_LAT`/`USA_LON` in `features.py`.**
  - It follows the combined position. On the 1,000 cleaned rows under 600 km from land where the combined and JTWC positions are at least 25 km apart, distance to a Natural Earth 10m coastline was within 10 km of `DIST2LAND` for 81.6% of rows using the combined position, but 26.7% using JTWC's (median error 2.7 km against 26.9 km).
  - It uses no future data. The same calculation from the combined position 3 h later matched 30.9% of those rows, and 3 h earlier 33.2%.
  - The test coastline isn't the one IBTrACS uses, so the comparison between sources is what counts, not the size of the errors.
- **`USA_PRES` has no values before 2001.** It's filled for 35% of train rows, but 100% of val and 99.5% of test. A model given this column would see a different pattern in testing than in training, and could learn "missing means old".
- **`USA_WIND` is missing in 32 rows,** all in season 1980. They make up two whole storms.
- **Rows aren't always 6 h apart.** Four storms have a 12 h gap where an interpolated fix (`P` or `I`) was removed: `2000331N09092`, `2005007N04085` and `2005330N09096` in train, `2019337N11072` in test. Build the 24/48/72 h targets by matching timestamps, not by shifting 4, 8 or 12 rows.
- **Some storms are too short for every lead time.** 11 storms span less than 24 h, 48 span less than 72 h, and 3 are a single row.
