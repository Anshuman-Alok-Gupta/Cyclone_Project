# Components

What each module in `src/cyclone/` does and why. The pipeline runs in this order:

```
config → io.load_ibtracs → filters.clean_ni → splits.assign_split → checks.check_expected → io.save_parquet
```

The project decisions behind these modules are in `CLAUDE.md`.

## config.py

**What it does:** the one place that says where data lives.

**Inputs and outputs:** reads the `CYCLONE_DATA_DIR` environment variable. `data_dir()`, `ibtracs_path(basin)` and `ni_clean_path()` return paths.

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
- `clean_ni(df)` takes the raw table and returns 11 columns: `SID`, `NAME`, `SEASON`, `ISO_TIME`, `USA_LAT`, `USA_LON`, `USA_WIND`, `USA_PRES`, `DIST2LAND`, `STORM_SPEED`, `STORM_DIR`.

**Why it's built this way:**
- Small separate functions can be reused one at a time and read like the list in `CLAUDE.md`.
- `clean_ni` prints counts after every step, so a changed total can be traced to one filter.

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

**Reused by:** baselines, XGBoost, the LSTM and the scoring harness. Project 3 reports on the same test years so its scores compare directly.

## checks.py

**What it does:** stops the run if the cleaned table isn't the expected dataset.

**Inputs and outputs:** `check_expected(df)` takes the table with its `split` column. It prints a confirmation if all counts match. Otherwise it raises an `AssertionError` listing each difference and the likely causes.

**Why it's built this way:**
- Wrong data still trains and still scores, so nothing downstream would notice.
- It checks rows and storms separately because they catch different problems: lost points inside a storm, or whole storms moving.
- It uses `raise` rather than `assert`, because `python -O` removes `assert` lines.
- It only checks counts. A table with the right counts but wrong values would pass, which is why the notebook also draws a map.

**Reused by:** every run of the notebook. If the data is re-downloaded on purpose, review the differences and then update the numbers here and in `CLAUDE.md`.

## plotting.py

**What it does:** draws one storm's track on a map, with dots coloured by wind.

**Inputs and outputs:** `plot_track(df, sid=...)` or `plot_track(df, name=..., season=...)` takes a table and returns the map axes.

**Why it's built this way:**
- A map shows position errors that counts can't.
- The colour scale is fixed at 10–150 kt, so a shade means the same wind on every map.
- Points with no wind get hollow markers. Without them, scatter would silently drop those positions.
- Name lookup handles joined names like `HIKAA:HIKKA`. If a name matches several storms, as `UNNAMED` often does, it lists their SIDs instead of guessing.
- It accepts an existing `ax`, so several tracks can share one figure.
- cartopy downloads coastline files the first time it runs, so that first plot needs internet.

**Reused by:** forecast error checks, by drawing forecast positions on the returned axes. Project 3 can use the same approach to compare AI-model tracks.

## Before building features

These don't affect the cleaned table, but they matter for the next step.

- **`STORM_SPEED` and `STORM_DIR` shouldn't be model inputs as they are.**
  - They're kept in the table as requested, but they don't come from JTWC positions. They match a centred difference of the combined `LAT`/`LON` over the points 3 h either side (96.7% within 1 kt, compared with 66.9% for the same calculation on `USA_LAT`/`USA_LON`, and 56% for a backward-only one).
  - So they use the source `CLAUDE.md` rules out, and they use a position 3 h in the future, which a real forecast wouldn't have.
  - Compute motion from `USA_LAT`/`USA_LON` at the current and previous fixes instead.
- **`DIST2LAND` may have the same source problem.** This wasn't tested, but it probably comes from the combined position too. It uses no future data, as far as is known.
- **`USA_PRES` has no values before 2001.** It's filled for 35% of train rows, but 100% of val and 99.5% of test. A model given this column would see a different pattern in testing than in training, and could learn "missing means old".
- **`USA_WIND` is missing in 32 rows,** all in season 1980. They make up two whole storms.
- **Rows aren't always 6 h apart.** Four storms have a 12 h gap where a fix was dropped (`2000331N09092`, `2005007N04085`, `2005330N09096`, `2019337N11072`). Build the 24/48/72 h targets by matching timestamps, not by shifting 4, 8 or 12 rows.
- **Some storms are too short for every lead time.** 11 storms span less than 24 h, 48 span less than 72 h, and 3 are a single row.
