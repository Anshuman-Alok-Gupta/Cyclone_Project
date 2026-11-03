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
- Use 6-hourly synoptic times only (00/06/12/18 UTC). Drop interpolated rows:
  - drop the off-synoptic rows (03/09/15/21 UTC and other odd times), AND
  - drop synoptic-hour rows that `IFLAG` marks as interpolated (about 5% of NI synoptic rows since 1980).
- Keep storms from 1980 onward.
- Track types: keep `TRACK_TYPE == "main"` only. Drop provisional (`PROVISIONAL`, `US-PROVISIONAL`) and spur (`spur-*`) tracks.
- For the NI dataset, keep only the NI-basin portion of each storm (`BASIN == "NI"`).
- Wind source: JTWC (`USA_WIND`), because the WP pretraining file has no IMD winds (`NEWDELHI_WIND` is 0% filled for WP-basin rows).
  Never mix in `NEWDELHI_WIND` or `WMO_WIND`. IMD reports 3-minute winds and JTWC 1-minute winds, so they aren't comparable.

## Evaluation
- Split by year (`SEASON`), never randomly:
  - train: up to and including 2012
  - validate: 2013–2017
  - test: 2018–2024
  - excluded: 2025 onward (provisional data)
- Scoring harness: great-circle error in km. It must be a standalone module that accepts any table of forecast positions, so it can score AI-model output in Project 3.

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
