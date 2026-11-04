"""Tests for XGBoost under cross-validation, and for xgb_residual.

Leakage through a held-out block is tested in test_cv.py. These test what's particular
to this file: nothing stops early, the residual target is what it claims to be, the search
ranks combinations properly, and the final fit can't reach test.
"""

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest

from cyclone import cv, xgb_model
from cyclone.baselines import cliper_forecast, fit_cliper_on, predict_cliper
from cyclone.features import build_features
from cyclone.folds import NOT_IN_A_FOLD, add_fold
from cyclone.samples import LEAD_HOURS, make_init_times, with_truth_at
from cyclone.scoring import FORECAST_COLUMNS

TINY_SETTINGS = {
    "max_depth": 2,
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "subsample": 1.0,
    "n_estimators": 5,
}


def test_the_search_for_cross_validation_includes_the_number_of_trees():
    combinations = xgb_model.sample_cv_settings()
    assert len(combinations) == xgb_model.N_COMBINATIONS
    assert all("n_estimators" in settings for settings in combinations)


def test_the_search_is_the_same_every_time():
    assert xgb_model.sample_cv_settings() == xgb_model.sample_cv_settings()


def test_notebook_03s_search_is_left_as_it_was():
    combinations = xgb_model.sample_settings()
    assert all("n_estimators" not in settings for settings in combinations)
    assert combinations[0].keys() == xgb_model.SEARCH_SPACE.keys()


def test_fit_fixed_keeps_every_tree_and_never_stops_early(rows, truth):
    models = xgb_model.fit_fixed(rows, truth, TINY_SETTINGS, "cpu")
    assert len(models) == len(LEAD_HOURS) * len(xgb_model.COMPONENTS)
    for model in models.values():
        assert model.get_booster().num_boosted_rounds() == TINY_SETTINGS["n_estimators"]
        assert model.early_stopping_rounds is None


def test_plain_forecasts_are_the_position_at_t0_plus_the_change(rows, truth):
    models = xgb_model.fit_fixed(rows, truth, TINY_SETTINGS, "cpu")
    forecasts = xgb_model.predict_xgboost(models, rows)
    at_24 = forecasts[forecasts["lead_h"] == 24].reset_index(drop=True)
    change = models[(24, "dlat")].predict(rows[xgb_model.FEATURE_COLUMNS])
    np.testing.assert_allclose(at_24["lat"], rows["lat"] + change)


def test_residual_target_is_the_truth_minus_the_base_forecast(rows, truth, monkeypatch):
    base = predict_cliper(fit_cliper_on(rows, truth), rows)
    captured = {}

    class Recorder(xgb_model.xgb.XGBRegressor):
        def fit(self, inputs, target):
            captured[len(captured)] = target
            return super().fit(inputs, target)

    monkeypatch.setattr(xgb_model.xgb, "XGBRegressor", Recorder)
    xgb_model.fit_fixed(rows, truth, TINY_SETTINGS, "cpu", base=base)

    cases = with_truth_at(rows, truth, 24)
    at_24 = base[base["lead_h"] == 24][["SID", "init_time", "lat"]].rename(columns={"lat": "base"})
    expected = cases.merge(at_24, on=["SID", "init_time"])
    # The first model fitted is +24 h dlat.
    np.testing.assert_allclose(captured[0], expected["truth_lat"] - expected["base"])


def test_residual_forecasts_are_clipers_forecast_plus_the_correction(rows, truth):
    base = predict_cliper(fit_cliper_on(rows, truth), rows)
    models = xgb_model.fit_fixed(rows, truth, TINY_SETTINGS, "cpu", base=base)
    forecasts = xgb_model.predict_xgboost(models, rows, xgb_model.RESIDUAL_NAME, base=base)

    at_48 = forecasts[forecasts["lead_h"] == 48].reset_index(drop=True)
    base_48 = base[base["lead_h"] == 48].reset_index(drop=True)
    correction = models[(48, "dlon")].predict(rows[xgb_model.FEATURE_COLUMNS])
    np.testing.assert_allclose(at_48["lon"], base_48["lon"] + correction)
    assert (forecasts["model"] == xgb_model.RESIDUAL_NAME).all()
    assert forecasts.columns.tolist() == FORECAST_COLUMNS


def test_a_correction_of_zero_gives_back_clipers_forecast(rows, truth):
    # With a learning rate of 0 the trees add nothing but XGBoost's starting value, which
    # is the mean target. So the forecasts stay within that much of CLIPER's.
    settings = {**TINY_SETTINGS, "learning_rate": 0.0}
    forecaster = xgb_model.residual_forecaster(settings, "cpu")
    residual = cv.out_of_fold(rows, truth, forecaster)
    cliper = cv.out_of_fold(rows, truth, cliper_forecast)
    gap = np.abs(residual["lat"].to_numpy() - cliper["lat"].to_numpy())
    assert gap.max() < 0.01


def test_search_cv_ranks_combinations_best_first(rows, truth, monkeypatch):
    weak = {**TINY_SETTINGS, "n_estimators": 1, "learning_rate": 0.02}
    strong = {**TINY_SETTINGS, "n_estimators": 40, "max_depth": 3}
    monkeypatch.setattr(xgb_model, "sample_cv_settings", lambda: [weak, strong])

    table = xgb_model.search_cv(rows, truth, device="cpu")

    assert table["relative_error"].is_monotonic_increasing
    assert xgb_model.best_settings(table) == strong
    error_columns = [f"km_{lead_h}h" for lead_h in LEAD_HOURS]
    assert (table[error_columns].min() > 0).all()


def test_a_dropped_search_resumes_and_gives_the_same_table(rows, truth, tmp_path, monkeypatch):
    first = {**TINY_SETTINGS, "n_estimators": 2}
    second = {**TINY_SETTINGS, "n_estimators": 4}
    third = {**TINY_SETTINGS, "n_estimators": 6, "subsample": 0.6}
    monkeypatch.setattr(xgb_model, "sample_cv_settings", lambda: [first, second, third])

    checkpoint = tmp_path / "models" / "search.csv"
    whole = xgb_model.search_cv(rows, truth, device="cpu", checkpoint=checkpoint)
    assert len(pd.read_csv(checkpoint)) == 3

    # As if the session died after one combination.
    pd.read_csv(checkpoint).iloc[:1].to_csv(checkpoint, index=False)
    fitted = []

    def counting_forecaster(settings, device):
        fitted.append(settings)
        return xgb_model.xgboost_forecaster(settings, device)

    resumed = xgb_model.search_cv(
        rows, truth, counting_forecaster, device="cpu", checkpoint=checkpoint, resume=True
    )
    assert fitted == [second, third]
    tm.assert_frame_equal(whole, resumed)


def test_a_search_ignores_an_old_checkpoint_unless_asked(rows, truth, tmp_path, monkeypatch):
    settings = {**TINY_SETTINGS, "n_estimators": 2}
    monkeypatch.setattr(xgb_model, "sample_cv_settings", lambda: [settings])
    checkpoint = tmp_path / "search.csv"
    pd.DataFrame([{**settings, "km_24h": 1.0, "km_48h": 1.0, "km_72h": 1.0}]).to_csv(
        checkpoint, index=False
    )

    table = xgb_model.search_cv(rows, truth, device="cpu", checkpoint=checkpoint)

    assert table["km_24h"].iloc[0] > 1.0


def test_best_settings_restores_each_type_for_both_kinds_of_table():
    columns = {**TINY_SETTINGS, "km_24h": 1.0, "km_48h": 2.0, "km_72h": 3.0}
    settings = xgb_model.best_settings(pd.DataFrame([columns]))
    assert isinstance(settings["n_estimators"], int)
    assert isinstance(settings["max_depth"], int)

    without_trees = {key: value for key, value in columns.items() if key != "n_estimators"}
    assert "n_estimators" not in xgb_model.best_settings(pd.DataFrame([without_trees]))


def test_fit_final_learns_from_every_block_and_nothing_else(table, rows, truth):
    init_times = make_init_times(table)
    extra = build_features(init_times).drop(columns=["split", "lat", "lon", "wind"])
    later = add_fold(init_times.merge(extra, on=["SID", "init_time"]), table)
    later = later[later["fold"] == NOT_IN_A_FOLD]
    assert len(later) > 0

    # Later storms with wildly wrong positions: if they were used, the models would move.
    corrupted = truth.copy()
    corrupted.loc[corrupted["SID"].isin(later["SID"]), ["lat", "lon"]] += 40.0
    full = pd.concat([rows, later[rows.columns]], ignore_index=True)

    clean = xgb_model.fit_final(rows, truth, TINY_SETTINGS, "cpu")
    reached = xgb_model.fit_final(full, corrupted, TINY_SETTINGS, "cpu")
    tm.assert_frame_equal(
        xgb_model.predict_xgboost(clean, rows),
        xgb_model.predict_xgboost(reached, rows),
    )

    every_block = xgb_model.fit_fixed(rows, truth, TINY_SETTINGS, "cpu")
    tm.assert_frame_equal(
        xgb_model.predict_xgboost(clean, rows),
        xgb_model.predict_xgboost(every_block, rows),
    )


def test_final_models_reload_with_identical_forecasts(rows, truth, tmp_path):
    models = xgb_model.fit_final(rows, truth, TINY_SETTINGS, "cpu")
    xgb_model.save_models(models, pd.DataFrame([TINY_SETTINGS]), tmp_path)
    reloaded = xgb_model.load_models(tmp_path)
    tm.assert_frame_equal(
        xgb_model.predict_xgboost(models, rows),
        xgb_model.predict_xgboost(reloaded, rows),
    )


def test_a_repeat_run_gives_identical_forecasts(rows, truth):
    settings = {**TINY_SETTINGS, "subsample": 0.6}
    forecaster = xgb_model.xgboost_forecaster(settings, "cpu")
    tm.assert_frame_equal(
        cv.out_of_fold(rows, truth, forecaster),
        cv.out_of_fold(rows, truth, forecaster),
    )
