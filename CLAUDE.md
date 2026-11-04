# Cyclone track forecasting — North Indian Ocean

## Goal
Forecast tropical cyclone positions at 24, 48 and 72 h lead times.
This is Project 1 of 3, leading to benchmarking AI weather models (GraphCast, Pangu) on Indian cyclones in Project 3.

## Working style
- The user wants to understand every piece. When writing code, explain the design choices; don't just dump code.

## Data
- Source: IBTrACS v04r01 (NOAA NCEI), downloaded 2026-11-04.
  - `data/raw/ibtracs.NI.list.v04r01.csv`: North Indian Ocean, the main dataset.
  - `data/raw/ibtracs.WP.list.v04r01.csv`: West Pacific, for a later pretraining experiment.
  - URL: https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/
- Row 2 of every IBTrACS CSV is a units row, not data. Skip it when loading.
- **NA basin trap:** load IBTrACS with `keep_default_na=False, na_values=[" "]`.
  IBTrACS marks missing values with a single space, and `BASIN="NA"` means North Atlantic.
  pandas' default settings silently turn "NA" into NaN (482 rows in the NI file).
- Basin files contain a storm's full track, including points in other basins (e.g. the NI file has WP rows, the WP file has EP and NI rows).
- `data/raw` and `data/processed` are git-ignored. Re-download them rather than committing them.

## Data decisions
- **Reference track source: JTWC throughout.** Positions from `USA_LAT`/`USA_LON`, winds from `USA_WIND`.
  Never use the combined `LAT`/`LON` (it blends agencies) or any `NEWDELHI_*`/`WMO_*` positions or winds for training or scoring.
  - Why JTWC: the WP pretraining file has no IMD data (`NEWDELHI_WIND` is 0% filled for WP-basin rows), and taking
    positions and winds from one agency keeps them consistent. IMD reports 3-minute winds and JTWC 1-minute winds, so they aren't comparable.
- **Interpolation flag:** keep only rows where `IFLAG` character 1 (index 0) == `"O"`, i.e. an original JTWC fix.
  - Character 1 is the USA/JTWC flag: it is `"_"` exactly when `USA_LAT` is missing (100% match in the NI file). Character 6 is IMD.
  - `"P"`/`"I"` rows have `USA_LAT` filled with interpolated values, so `USA_LAT` being present does NOT mean it's an original fix.
- **Row filters for the NI dataset, applied in this order:**
  1. `TRACK_TYPE == "main"`: drops provisional and spur tracks
  2. `BASIN == "NI"`: keeps only the NI-basin portion of each storm
  3. synoptic hours only: 00/06/12/18 UTC (minute 00)
  4. `SEASON` 1980–2024
  5. `USA_LAT` present and `IFLAG[0] == "O"`
- **Expected result:** 5,694 rows, 258 storms (by split: train 184, validate 27, test 47). These were verified against v04r01 on 2026-11-04.
  If the cleaning code gets different numbers, stop and report why before continuing.
- **Known limitation:** where both exist, JTWC and IMD positions differ by a median of ~51 km (p90 ~133 km).
  Keep JTWC as the reference everywhere. In Project 3, also report scores against IMD tracks as a sensitivity check.

## Evaluation
- Split by year (`SEASON`), never randomly:
  - development: 1980–2017, for choosing and comparing models
  - test: 2018–2024
  - excluded: 2025 onward (provisional data)
- **Model selection uses 5-fold blocked cross-validation over 1980–2017.** The blocks are whole, consecutive seasons:
  1980–87, 1988–95, 1996–2003, 2004–10 and 2011–17, defined once in `src/cyclone/folds.py`.
  Each block is forecast by models fitted on the other four, and models are compared on those out-of-fold forecasts.
  - Why: a single validation split of 27 storms was too few. On storms held back from the choice, XGBoost's lead over
    CLIPER at +48 h fell from 4.7% to 0%. Evidence is in `docs/components.md`.
  - No early stopping on a held-out block: searching the number of trees replaces it, since a block that chooses
    its own stopping point leaks into its own forecasts.
  - Report skill against CLIPER with a 95% interval from a bootstrap of whole storms, not rows.
- **The old train/val split stays only so notebooks 02–03 still run:** train up to and including 2012, validate 2013–2017.
  Don't use it to choose or compare models.
- **Test plan, decided on 2026-11-04 before anyone has seen test results.** Test (2018–2024) is in no block, and
  scoring it earlier or going back to change something after seeing it would turn it into a second validation set.
  - The test split is scored exactly once, at the end of Project 1, after the LSTM and West Pacific pretraining steps.
  - Primary result: the model with the best cross-validated relative error over 1980–2017 at that point, compared
    with CLIPER. If two are within their cross-validation uncertainty, prefer the simpler one.
  - Secondary results, reported alongside without changing the primary choice: persistence, CLIPER, XGBoost and consensus.
  - Final models are refitted on all of 1980–2017 with the settings chosen by cross-validation. Any XGBoost member is
    the average of 5 seeds, because single-seed results varied (consensus +72 h ranged 2.1–2.4% across seeds).
  - Current leader (2026-11-04): consensus of CLIPER and XGBoost.
- Scoring harness: great-circle error in km. It must be a standalone module that accepts any table of forecast positions, so it can score AI-model output in Project 3.

## Model input rules
- **No future information.** A model input may only use information available at the forecast start time (t0):
  JTWC fixes at t0 and earlier. Never anything after t0.
- **No IBTrACS derived columns.** Never use IBTrACS's derived columns (`STORM_SPEED`, `STORM_DIR`, `DIST2LAND`,
  etc.) as inputs unless they're verified as JTWC-based and free of future information.
  Calculate motion from `USA_LAT`/`USA_LON` at the current and previous fixes.
  - Why: `STORM_SPEED`/`STORM_DIR` are a centred difference of the combined `LAT`/`LON`, so they use the position 3 h
    after t0. `DIST2LAND` uses no future data but also follows the combined position, so it's recomputed from
    `USA_LAT`/`USA_LON` in the features step. Evidence is in `docs/components.md`.
- **`LANDFALL` is permanently banned.** Never use it as an input. The "unless verified" exception above doesn't apply.
  - Why: it looks ahead. Tested on 2026-11-04, it matches min(distance to land now, at the next point) for 81.4% of
    rows against 47.4% for the current distance alone, and in 8,774 of 16,696 rows it's smaller than the current
    distance to land.
- **Match fixes by timestamp.** The cleaned data has 4 storms with a 12 h gap, where an interpolated fix was removed
  (3 in train, 1 in test). Never assume consecutive rows are 6 h apart; always match fixes by timestamp.
- **Scope wording for results:** "JTWC-warned systems in the North Indian Ocean", not all depressions.

## Models
- Baselines: persistence and CLIPER.
- Models: XGBoost first, then LSTM.
- Predict the change in position (Δlat, Δlon), not absolute lat/lon.

## Code structure and cloud portability
Heavy training runs on Google Colab (student GPU access), with Kaggle as a backup.
- Project code lives in `src/` as an importable package. Notebooks only call into it.
  A Colab notebook must be able to clone the repo, `pip install -r requirements.txt`, and run.
- All code must be device-agnostic: use the GPU if available, otherwise the CPU. Never hardcode a device.
- Never hardcode file paths. Use a single config for the data directory that works both locally and on
  Colab with Google Drive mounted (e.g. `MyDrive/cyclone-project/data`).
- Save model checkpoints regularly to the data directory, so a dropped Colab session doesn't lose training progress.

## Environment
- Local: Windows 11, Python 3.12.12 (uv-managed interpreter), venv at `.venv`, Jupyter kernel `cyclone`.
- Activate the venv (PowerShell): `.\.venv\Scripts\Activate.ps1`
- Requirements files:
  - `requirements.txt`: direct dependencies only, with loose `>=` minimums, for Colab/Kaggle and any fresh machine.
    Excludes torch (see below) and jupyter (Colab has it). Pinning everything would make Colab replace its preinstalled
    packages and would pull in Windows-only packages such as `pywinpty`.
  - `requirements-lock-windows.txt`: full `pip freeze` of the local Windows venv (excluding torch). Use it only to rebuild
    this exact local environment, never on Colab.
- **Torch note:** torch is NOT in either requirements file. Locally it stays CPU-only: the GeForce MX130
  (compute capability 5.0, 2 GB) isn't supported by current CUDA builds. Pinning that CPU build would break on GPU machines,
  and Colab and Kaggle come with CUDA torch preinstalled.
  Local install: `pip install torch --index-url https://download.pytorch.org/whl/cpu` (currently 2.15.0+cpu).
