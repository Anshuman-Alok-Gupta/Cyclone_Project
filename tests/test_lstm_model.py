"""Tests for the LSTM.

The data is the set of made-up storms in conftest.py, and each fit stops after a few
epochs. So these check the rules a fit has to obey, not how good its forecasts are.
"""

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
import torch

from cyclone import cv, lstm_model, sequences
from cyclone.folds import FOLD_NAMES, development
from cyclone.lstm_model import TrackLSTM, masked_loss
from cyclone.samples import LEAD_HOURS
from cyclone.scoring import FORECAST_COLUMNS, score
from cyclone.sequences import FIXED_COLUMNS, STEP_COLUMNS, Sequences, build_sequences

SETTINGS = {
    "hidden_size": 4,
    "num_layers": 1,
    "dropout": 0.0,
    "learning_rate": 0.01,
    "weight_decay": 0.0,
}
OTHER_SETTINGS = {**SETTINGS, "hidden_size": 6}
TWO_SEEDS = (42, 43)


@pytest.fixture(scope="module")
def built(rows, table) -> Sequences:
    return build_sequences(rows, table)


@pytest.fixture(scope="module")
def fitted(rows, truth, built) -> lstm_model.FittedLSTM:
    return lstm_model.fit_lstm(rows, truth, built, SETTINGS, "cpu")


def one_block(rows: pd.DataFrame, number: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one fold's learning rows and held-out rows."""
    fold = FOLD_NAMES[number]
    return rows[rows["fold"] != fold], rows[rows["fold"] == fold]


def tiny_batch(forecasts: int = 3, steps: int = 13) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    return (
        torch.randn(forecasts, steps, len(STEP_COLUMNS)),
        torch.tensor([5, 9, 13][: forecasts]),
        torch.randn(forecasts, len(FIXED_COLUMNS)),
    )


def test_the_network_gives_two_components_at_each_lead():
    network = TrackLSTM(hidden_size=8, num_layers=2, dropout=0.2).eval()
    assert network(*tiny_batch()).shape == (3, len(LEAD_HOURS), 2)


def test_the_padding_after_a_sequence_ends_cannot_change_the_answer():
    network = TrackLSTM(hidden_size=8, num_layers=2, dropout=0.0).eval()
    steps, lengths, fixed = tiny_batch()
    changed = steps.clone()
    changed[0, lengths[0] :] = 999.0
    assert torch.equal(network(steps, lengths, fixed)[0], network(changed, lengths, fixed)[0])


def test_the_last_real_fix_does_change_the_answer():
    # A partner to the test above: an answer read from the wrong step might pass that one.
    network = TrackLSTM(hidden_size=8, num_layers=1, dropout=0.0).eval()
    steps, lengths, fixed = tiny_batch()
    changed = steps.clone()
    changed[0, lengths[0] - 1] += 1.0
    assert not torch.equal(network(steps, lengths, fixed)[0], network(changed, lengths, fixed)[0])


def test_the_fixed_inputs_reach_the_answer():
    network = TrackLSTM(hidden_size=8, num_layers=1, dropout=0.0).eval()
    steps, lengths, fixed = tiny_batch()
    assert not torch.equal(network(steps, lengths, fixed), network(steps, lengths, fixed + 1.0))


def test_dropout_is_used_with_a_single_layer():
    network = TrackLSTM(hidden_size=8, num_layers=1, dropout=0.4)
    assert network.lstm.dropout == 0.0
    assert network.head[0].p == 0.4


def test_the_loss_is_ordinary_mean_squared_error_when_nothing_is_missing():
    predicted = torch.ones(4, 3, 2)
    assert masked_loss(predicted, torch.zeros(4, 3, 2)).item() == pytest.approx(1.0)


def test_a_missing_lead_adds_nothing_to_the_loss():
    target = torch.zeros(2, 3, 2)
    target[0, 2] = float("nan")
    predicted = torch.zeros(2, 3, 2)
    predicted[0, 2] = 1000.0
    assert masked_loss(predicted, target).item() == 0.0


def test_a_missing_lead_gives_no_pull_on_its_output_and_no_blanks():
    target = torch.ones(2, 3, 2)
    target[0, 2] = float("nan")
    predicted = torch.zeros(2, 3, 2, requires_grad=True)
    masked_loss(predicted, target).backward()

    assert not torch.isnan(predicted.grad).any()
    assert (predicted.grad[0, 2] == 0).all()
    assert (predicted.grad[0, 0] != 0).all()


def test_a_case_missing_72_h_still_teaches_the_earlier_leads():
    target = torch.full((1, 3, 2), float("nan"))
    target[0, 0] = 1.0
    predicted = torch.zeros(1, 3, 2)
    assert masked_loss(predicted, target).item() == pytest.approx(1.0)


def test_each_lead_counts_equally_however_many_answers_it_has():
    target = torch.zeros(4, 3, 2)
    target[1:, 2] = float("nan")  # one +72 h answer against four at the other leads
    predicted = torch.zeros(4, 3, 2)
    predicted[:, 0] = 1.0  # +24 h is wrong by 1 in every case
    predicted[0, 2] = 1.0  # +72 h is wrong by 1 in its one case
    assert masked_loss(predicted, target).item() == pytest.approx(2 / 3)


def test_targets_are_blank_only_where_the_storm_has_no_fix_at_that_lead(rows, truth):
    targets = lstm_model._targets(rows, truth)
    assert targets.shape == (len(rows), len(LEAD_HOURS), 2)

    last_of_storm = rows.groupby("SID")["init_time"].transform("max") == rows["init_time"]
    assert np.isnan(targets[last_of_storm.to_numpy()]).all()
    # These made-up storms are 24 fixes long, so the first start time has truth at all leads.
    earliest = rows.groupby("SID")["init_time"].transform("min") == rows["init_time"]
    assert not np.isnan(targets[earliest.to_numpy()]).any()


def test_a_target_is_the_truth_minus_the_position_at_t0(rows, truth):
    targets = lstm_model._targets(rows, truth)
    row = rows.iloc[0]
    later = truth[
        (truth["SID"] == row["SID"])
        & (truth["valid_time"] == row["init_time"] + pd.Timedelta(hours=48))
    ].iloc[0]
    assert targets[0, 1, 0] == pytest.approx(later["lat"] - row["lat"])
    assert targets[0, 1, 1] == pytest.approx(later["lon"] - row["lon"])


def test_cases_with_some_leads_missing_are_trained_on(rows, truth, built, monkeypatch):
    targets_seen = []

    def spying_loss(predicted, target):
        targets_seen.append(target.clone())
        return masked_loss(predicted, target)

    monkeypatch.setattr(lstm_model, "masked_loss", spying_loss)
    lstm_model.fit_lstm(rows, truth, built, SETTINGS, "cpu")

    every_target = torch.cat(targets_seen)
    missing = torch.isnan(every_target).all(dim=2)
    partly = missing.any(dim=1) & ~missing.all(dim=1)
    assert partly.any()
    assert not missing.all(dim=1).any()


def test_the_inner_storms_are_about_15_percent_chosen_whole_and_repeatable():
    storms = [f"S{number}" for number in range(40)]
    chosen = lstm_model.inner_storms(storms)
    assert len(chosen) == 6
    assert chosen <= set(storms)
    assert chosen == lstm_model.inner_storms(list(reversed(storms * 3)))


def test_only_the_learning_storms_can_be_inner_storms(rows):
    learning, held_out = one_block(rows)
    assert not lstm_model.inner_storms(learning["SID"]) & set(held_out["SID"])


def test_inner_storms_are_watched_and_never_trained_on(rows, truth, built, monkeypatch):
    learning, _ = one_block(rows)
    has_answer = ~np.isnan(lstm_model._targets(learning, truth)).all(axis=(1, 2))
    inner = lstm_model.inner_storms(learning["SID"].to_numpy()[has_answer])
    expected_inner = (learning["SID"].isin(inner).to_numpy() & has_answer).sum()
    expected_fit = (~learning["SID"].isin(inner).to_numpy() & has_answer).sum()

    counts = {True: 0, False: 0}

    def counting_loss(predicted, target):
        counts[torch.is_grad_enabled()] += len(target)
        return masked_loss(predicted, target)

    monkeypatch.setattr(lstm_model, "masked_loss", counting_loss)
    monkeypatch.setattr(lstm_model, "MAX_EPOCHS", 1)
    lstm_model.fit_lstm(learning, truth, built, SETTINGS, "cpu")

    assert counts[True] == expected_fit
    assert counts[False] == expected_inner


def scripted_inner_errors(monkeypatch, errors: list[float]) -> list[float]:
    """Make the inner error follow a script, and return the list of errors reached."""
    remaining = iter(errors)
    reached = []

    def scripted_loss(predicted, target):
        if torch.is_grad_enabled():
            return masked_loss(predicted, target)
        reached.append(next(remaining))
        return torch.tensor(reached[-1])

    monkeypatch.setattr(lstm_model, "masked_loss", scripted_loss)
    return reached


def test_training_stops_after_patience_runs_out_and_keeps_the_best_epoch(rows, truth, built, monkeypatch):
    monkeypatch.setattr(lstm_model, "MAX_EPOCHS", 20)
    monkeypatch.setattr(lstm_model, "PATIENCE", 4)
    reached = scripted_inner_errors(monkeypatch, [5.0, 4.0, 3.0] + [3.5] * 20)

    model = lstm_model.fit_lstm(rows, truth, built, SETTINGS, "cpu")

    assert model.best_epoch == 3
    assert len(reached) == 3 + 4


def test_a_stopped_network_holds_its_best_weights_not_its_last(rows, truth, built, monkeypatch):
    # A run that ends at epoch 3, with epoch 3 as its best, holds that epoch's weights.
    monkeypatch.setattr(lstm_model, "MAX_EPOCHS", 3)
    scripted_inner_errors(monkeypatch, [5.0, 4.0, 3.0])
    best_only = lstm_model.fit_lstm(rows, truth, built, SETTINGS, "cpu")

    monkeypatch.setattr(lstm_model, "MAX_EPOCHS", 20)
    monkeypatch.setattr(lstm_model, "PATIENCE", 4)
    scripted_inner_errors(monkeypatch, [5.0, 4.0, 3.0] + [3.5] * 20)
    stopped = lstm_model.fit_lstm(rows, truth, built, SETTINGS, "cpu")

    for name, value in best_only.network.state_dict().items():
        assert torch.equal(value, stopped.network.state_dict()[name])


def test_the_same_seed_repeats_exactly_and_another_seed_differs(rows, truth, built):
    first = lstm_model.fit_lstm(rows, truth, built, SETTINGS, "cpu", seed=42)
    again = lstm_model.fit_lstm(rows, truth, built, SETTINGS, "cpu", seed=42)
    other = lstm_model.fit_lstm(rows, truth, built, SETTINGS, "cpu", seed=43)

    forecasts = lstm_model.predict_lstm(first, rows, built)
    tm.assert_frame_equal(forecasts, lstm_model.predict_lstm(again, rows, built))
    assert not forecasts.equals(lstm_model.predict_lstm(other, rows, built))


def test_there_is_a_forecast_for_every_row_at_every_lead(rows, built, fitted):
    forecasts = lstm_model.predict_lstm(fitted, rows, built)
    assert forecasts.columns.tolist() == FORECAST_COLUMNS
    assert len(forecasts) == len(rows) * len(LEAD_HOURS)
    assert forecasts["lat"].notna().all()


def test_forecasts_are_the_start_position_plus_the_unscaled_change(rows, built, fitted):
    forecasts = lstm_model.predict_lstm(fitted, rows, built)
    at_24 = forecasts[forecasts["lead_h"] == LEAD_HOURS[0]].reset_index(drop=True)

    steps, lengths, fixed = lstm_model._tensors(sequences.select(built, rows), fitted.scaler, "cpu")
    with torch.no_grad():
        scaled = fitted.network(steps, lengths, fixed).numpy()
    change = scaled[:, 0, 0] * fitted.target_scaler.std[0, 0] + fitted.target_scaler.mean[0, 0]
    np.testing.assert_allclose(at_24["lat"], rows["lat"].to_numpy() + change, atol=1e-5)


def test_targets_are_scaled_per_lead_and_component_from_the_rows_fitted(rows, truth, fitted):
    change = lstm_model._targets(rows, truth)
    np.testing.assert_allclose(fitted.target_scaler.mean, np.nanmean(change, axis=0))
    np.testing.assert_allclose(fitted.target_scaler.std, np.nanstd(change, axis=0))
    assert fitted.target_scaler.std[2, 0] > fitted.target_scaler.std[0, 0]


@pytest.mark.parametrize("fold", FOLD_NAMES)
def test_changing_a_blocks_truth_leaves_its_forecasts_unchanged(rows, truth, built, fold):
    forecaster = lstm_model.lstm_forecaster(SETTINGS, "cpu", built)
    before = cv.out_of_fold(rows, truth, forecaster)

    storms = rows.loc[rows["fold"] == fold, "SID"]
    moved = truth.assign(lat=truth["lat"].where(~truth["SID"].isin(storms), truth["lat"] + 30))
    after = cv.out_of_fold(rows, moved, forecaster)

    same = before["fold"] == fold
    tm.assert_frame_equal(before[same], after[same])
    assert not before[~same].equals(after[~same])


def test_the_scaler_only_ever_sees_a_folds_learning_rows(rows, truth, built, monkeypatch):
    fitted_on = []
    real = lstm_model.fit_scaler

    def spying_scaler(chosen):
        fitted_on.append(set(chosen.keys["SID"]))
        return real(chosen)

    monkeypatch.setattr(lstm_model, "fit_scaler", spying_scaler)
    cv.out_of_fold(rows, truth, lstm_model.lstm_forecaster(SETTINGS, "cpu", built))

    assert len(fitted_on) == len(FOLD_NAMES)
    for fold, storms in zip(FOLD_NAMES, fitted_on, strict=True):
        assert storms == set(rows.loc[rows["fold"] != fold, "SID"])


def test_corrupting_held_out_inputs_leaves_the_folds_model_unchanged(rows, truth, built):
    learning, held_out = one_block(rows)
    learning_truth = truth[truth["SID"].isin(learning["SID"])]

    corrupted = built.steps.copy()
    corrupted[held_out.index] += 1000.0
    changed = Sequences(built.keys, corrupted, built.lengths, built.fixed)

    forecast = lstm_model.lstm_forecaster(SETTINGS, "cpu", built)
    forecast_changed = lstm_model.lstm_forecaster(SETTINGS, "cpu", changed)
    tm.assert_frame_equal(
        forecast(learning, learning_truth, learning),
        forecast_changed(learning, learning_truth, learning),
    )
    assert not forecast(learning, learning_truth, held_out).equals(
        forecast_changed(learning, learning_truth, held_out)
    )


def test_a_measured_forecast_is_the_average_of_its_seeds(rows, truth, built):
    fitted = lstm_model.fit_seeds(rows, truth, built, SETTINGS, "cpu", seeds=TWO_SEEDS)
    averaged = lstm_model.predict_seeds(fitted, rows, built)
    first = lstm_model.predict_lstm(fitted[42], rows, built)
    second = lstm_model.predict_lstm(fitted[43], rows, built)
    np.testing.assert_allclose(averaged["lat"], (first["lat"] + second["lat"]) / 2)
    np.testing.assert_allclose(averaged["lon"], (first["lon"] + second["lon"]) / 2)
    assert averaged["model"].eq(lstm_model.MODEL_NAME).all()


def test_the_24_h_model_runs_on_its_shorter_sequences(rows, truth, table):
    short = build_sequences(rows, table, history_h=24)
    forecaster = lstm_model.lstm_forecaster(
        SETTINGS, "cpu", short, name=lstm_model.HISTORY_24H_NAME
    )
    forecasts = cv.out_of_fold(rows, truth, forecaster)
    assert set(forecasts["model"]) == {lstm_model.HISTORY_24H_NAME}
    assert len(forecasts) == len(rows) * len(LEAD_HOURS)


def test_a_saved_network_gives_the_same_forecasts(rows, built, fitted, tmp_path):
    path = tmp_path / "models" / "seed42.pt"
    lstm_model.save_fitted(fitted, path)
    loaded = lstm_model.load_fitted(path)

    tm.assert_frame_equal(
        lstm_model.predict_lstm(fitted, rows, built),
        lstm_model.predict_lstm(loaded, rows, built),
    )
    assert loaded.settings == SETTINGS
    assert loaded.best_epoch == fitted.best_epoch


def refuse_to_fit(*args, **kwargs):
    raise AssertionError("A saved network should have been used.")


def test_fitting_saves_each_seed_and_resuming_fits_nothing(rows, truth, built, tmp_path, monkeypatch):
    first = lstm_model.fit_seeds(
        rows, truth, built, SETTINGS, "cpu", TWO_SEEDS, directory=tmp_path
    )
    monkeypatch.setattr(lstm_model, "fit_lstm", refuse_to_fit)
    resumed = lstm_model.fit_seeds(
        rows, truth, built, SETTINGS, "cpu", TWO_SEEDS, directory=tmp_path, resume=True
    )
    tm.assert_frame_equal(
        lstm_model.predict_seeds(first, rows, built),
        lstm_model.predict_seeds(resumed, rows, built),
    )


def test_a_session_dropped_between_seeds_refits_only_the_missing_one(rows, truth, built, tmp_path, monkeypatch):
    lstm_model.fit_seeds(rows, truth, built, SETTINGS, "cpu", TWO_SEEDS, directory=tmp_path)
    (tmp_path / "seed43.pt").unlink()

    fitted_seeds = []
    real = lstm_model.fit_lstm

    def counting_fit(rows, truth, sequences, settings, device, seed):
        fitted_seeds.append(seed)
        return real(rows, truth, sequences, settings, device, seed)

    monkeypatch.setattr(lstm_model, "fit_lstm", counting_fit)
    lstm_model.fit_seeds(
        rows, truth, built, SETTINGS, "cpu", TWO_SEEDS, directory=tmp_path, resume=True
    )
    assert fitted_seeds == [43]


def test_saved_networks_are_ignored_unless_asked(rows, truth, built, tmp_path, monkeypatch):
    lstm_model.fit_seeds(rows, truth, built, SETTINGS, "cpu", (42,), directory=tmp_path)
    monkeypatch.setattr(lstm_model, "fit_lstm", refuse_to_fit)
    with pytest.raises(AssertionError, match="saved network"):
        lstm_model.fit_seeds(rows, truth, built, SETTINGS, "cpu", (42,), directory=tmp_path)


def test_a_saved_network_with_other_settings_is_not_reused(rows, truth, built, tmp_path):
    lstm_model.fit_seeds(rows, truth, built, SETTINGS, "cpu", (42,), directory=tmp_path)
    fitted = lstm_model.fit_seeds(
        rows, truth, built, OTHER_SETTINGS, "cpu", (42,), directory=tmp_path, resume=True
    )
    assert fitted[42].settings == OTHER_SETTINGS


def test_each_block_gets_its_own_folder_of_saved_networks(rows, truth, built, tmp_path):
    forecaster = lstm_model.lstm_forecaster(SETTINGS, "cpu", built, directory=tmp_path)
    cv.out_of_fold(rows, truth, forecaster)
    assert {path.name for path in tmp_path.iterdir()} == set(FOLD_NAMES)


def test_the_learning_error_reuses_the_networks_cross_validation_saved(rows, truth, built, tmp_path, monkeypatch):
    forecaster = lstm_model.lstm_forecaster(SETTINGS, "cpu", built, directory=tmp_path)
    cv.out_of_fold(rows, truth, forecaster)

    monkeypatch.setattr(lstm_model, "fit_lstm", refuse_to_fit)
    resumed = lstm_model.lstm_forecaster(SETTINGS, "cpu", built, directory=tmp_path, resume=True)
    errors = cv.learning_error(rows, truth, resumed)
    assert errors.index.tolist() == list(LEAD_HOURS)
    assert (errors > 0).all()


def test_the_search_has_12_balanced_repeatable_combinations():
    combinations = lstm_model.sample_settings()
    assert len(combinations) == lstm_model.N_COMBINATIONS <= 12
    assert combinations == lstm_model.sample_settings()
    assert len({tuple(sorted(item.items())) for item in combinations}) == len(combinations)

    for name, values in lstm_model.SEARCH_SPACE.items():
        counts = pd.Series([item[name] for item in combinations]).value_counts()
        assert set(counts.index) == set(values)
        assert counts.nunique() == 1


def test_the_settings_are_plain_python_numbers():
    # A numpy number would stop a saved network reading back with the safe loader.
    for item in lstm_model.sample_settings():
        assert type(item["hidden_size"]) is int
        assert type(item["num_layers"]) is int
        assert type(item["dropout"]) is float


def test_search_cv_ranks_combinations_best_first(rows, truth, built, monkeypatch):
    weak = {**SETTINGS, "learning_rate": 1e-7}
    strong = {**SETTINGS, "learning_rate": 0.01}
    monkeypatch.setattr(lstm_model, "sample_settings", lambda: [weak, strong])

    table = lstm_model.search_cv(rows, truth, built, device="cpu")

    assert table["relative_error"].is_monotonic_increasing
    assert len(table) == 2


def test_a_dropped_search_resumes_and_gives_the_same_table(rows, truth, built, tmp_path, monkeypatch):
    first = {**SETTINGS, "hidden_size": 2}
    second = {**SETTINGS, "hidden_size": 3}
    third = {**SETTINGS, "hidden_size": 4}
    monkeypatch.setattr(lstm_model, "sample_settings", lambda: [first, second, third])

    checkpoint = tmp_path / "models" / "search.csv"
    whole = lstm_model.search_cv(rows, truth, built, device="cpu", checkpoint=checkpoint)
    assert len(pd.read_csv(checkpoint)) == 3

    # As if the session died after one combination.
    pd.read_csv(checkpoint).iloc[:1].to_csv(checkpoint, index=False)
    fitted = []
    real = lstm_model.lstm_forecaster

    def counting_forecaster(settings, device, sequences):
        fitted.append(settings)
        return real(settings, device, sequences)

    monkeypatch.setattr(lstm_model, "lstm_forecaster", counting_forecaster)
    resumed = lstm_model.search_cv(
        rows, truth, built, device="cpu", checkpoint=checkpoint, resume=True
    )
    assert fitted == [second, third]
    tm.assert_frame_equal(whole, resumed)


def test_best_settings_restores_each_type(rows):
    table = pd.DataFrame([{**SETTINGS, "km_24h": 1.0, "km_48h": 2.0, "km_72h": 3.0}]).astype(float)
    settings = lstm_model.best_settings(table)
    assert settings == SETTINGS
    assert type(settings["hidden_size"]) is int
    assert type(settings["num_layers"]) is int
    TrackLSTM(settings["hidden_size"], settings["num_layers"], settings["dropout"])


def test_the_device_falls_back_to_the_cpu_without_a_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert lstm_model.pick_device() == "cpu"


def test_the_device_falls_back_when_a_gpu_is_seen_but_cannot_be_used(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("no kernel image is available for this GPU")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.nn.LSTM, "to", refuse)
    assert lstm_model.pick_device() == "cpu"


def test_a_fitted_network_ends_on_the_cpu(fitted):
    assert all(value.device.type == "cpu" for value in fitted.network.state_dict().values())


def test_fit_final_is_the_measured_model_handed_every_block(rows, truth, built):
    final = lstm_model.fit_final(rows, truth, built, SETTINGS, "cpu")
    assert list(final) == list(lstm_model.SEEDS)

    learning = development(rows)
    forecaster = lstm_model.lstm_forecaster(SETTINGS, "cpu", built, seeds=lstm_model.SEEDS)
    tm.assert_frame_equal(
        lstm_model.predict_seeds(final, learning, built),
        forecaster(learning, truth, learning),
    )


def test_a_final_model_missing_a_seed_will_not_load(rows, truth, built, tmp_path):
    final = lstm_model.fit_final(rows, truth, built, SETTINGS, "cpu")
    lstm_model.save_final(final, pd.DataFrame([SETTINGS]), tmp_path)
    assert list(lstm_model.load_final(tmp_path)) == list(lstm_model.SEEDS)

    (tmp_path / "seed44.pt").unlink()
    with pytest.raises(FileNotFoundError):
        lstm_model.load_final(tmp_path)


def test_scoring_accepts_the_forecasts(rows, truth, built, fitted):
    scored = score(lstm_model.predict_lstm(fitted, rows, built), truth, report=False)
    assert scored["error_km"].notna().any()
