"""The LSTM, a neural network that reads a storm's track one fix at a time.

**What it is.** A neural network is a long chain of multiplications and small bends,
controlled by thousands of numbers called weights. Training shows it a batch of examples,
measures how wrong it was, and nudges every weight slightly in the direction that would
have made it less wrong. One pass through all the examples is an *epoch*.

An LSTM (long short-term memory) is the kind built for sequences. It reads one fix at a
time and carries a small set of numbers forward, its memory. At each fix it decides what
to keep, what to forget and what to add. So nobody has to choose "motion over the last 12 h"
in advance, as for XGBoost. It can learn its own summary, and can see further back.

**The network.** Each sequence goes through one or two LSTM layers, and the memory at the
sequence's last real fix, t0, is its summary of the track. The fixed inputs (time of year and
distance to land) are joined on at that point: they don't change along a sequence, so feeding
them in at every step would only repeat them. One small head then turns all of that into six
numbers: dlat and dlon at +24, +48 and +72 h.

**Why one head for all three leads.** XGBoost fits six separate models, and the +48 h
model doesn't know what the +24 h model said, so a forecast's points can disagree with
each other. Here all six numbers come from one summary of the storm, and the same weights
have to serve every lead. A situation that pushes the +24 h point north naturally pushes
+48 and +72 h too, and that can give a smoother track. It's a tendency, not a guarantee.
It also means the long leads, with their fewer cases, can borrow from the many +24 h cases.

**Short sequences.** Sequences are padded at the end to fit one array. The padding never
matters: an LSTM only reads forward, so the memory at the real last fix is already fixed
before the padding starts, and that's the one the network uses.

**Scaled targets.** Like the inputs, targets are scaled, one mean and spread per lead and
component, fitted on the learning rows. Otherwise +72 h, with moves three times as large,
would have nine times the squared error and would drown out the other leads. This is the
same reasoning as the relative-error rule, applied to training.

**Missing leads.** Some cases have truth at +24 h but not +72 h, because the storm ended.
Dropping those would throw away real +24 h answers, so the loss uses a mask, a grid of 1s
and 0s saying which answers exist. Each error is multiplied by its mask value, which sets
a missing lead's error to exactly 0, so it can't push any weight. ``masked_loss`` then
averages over the answers that exist, lead by lead, and the leads count equally.

**Stopping.** Training for too long makes a network memorize the storms it has seen, so
its error on new storms starts to rise again. Early stopping watches a set of storms it
doesn't learn from, and keeps the weights from the epoch that did best there.

The held-out block can't be that set, for the reason in ``xgb_model.py``: if a block's
own answers chose the stopping point, they'd have shaped the forecast being scored. So
about 15% of the learning storms are set aside instead, chosen by storm and with a fixed
seed. This is allowed because those storms' answers are already part of what the fold may
learn from. Nothing from the held-out block is involved, so its forecasts stay honest.

XGBoost didn't do this, because holding storms back would leave each of its folds and six
models with a different tree count, and no single setting to refit with at the end. An
LSTM doesn't have that problem: what's carried to the final model is the rule (stop on 15%
of the storms, with this patience), not a number of epochs. It does cost those storms as
training data, which all the LSTM's numbers include.

**Repeatable.** Seeds are fixed, so the same data on the same device gives the same model.
A GPU and a CPU can give slightly different models, as for XGBoost.

**Where it runs.** Training uses a GPU if there is one. Forecasting always runs on the
CPU, since it's quick and a saved model then gives the same forecasts anywhere.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from cyclone import config, cv
from cyclone.folds import FOLD_NAMES, development
from cyclone.samples import LEAD_HOURS, with_truth_at
from cyclone.scoring import FORECAST_COLUMNS
from cyclone.sequences import (
    FIXED_COLUMNS,
    STEP_COLUMNS,
    Scaler,
    Sequences,
    fit_scaler,
    scale,
    select,
)
from cyclone.xgb_model import SEED, SEEDS

# LSTMs on a GPU are only repeatable with this set, and it has to happen before the first
# GPU call, which importing torch doesn't make. It does nothing on a CPU.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:2")

MODEL_NAME = "lstm"
# The same network given only 24 h of history, to tell whether any gain comes from the
# longer history or from the model. It's a check, not a candidate.
HISTORY_24H_NAME = "lstm_24h"

N_COMPONENTS = 2  # dlat and dlon

N_COMBINATIONS = 12
SEARCH_SPACE = {
    # How many numbers the memory holds. More can capture more, and memorize more noise.
    "hidden_size": [16, 32, 64],
    # A second layer reads the first layer's output, so it can pick up longer patterns.
    "num_layers": [1, 2],
    # The share of numbers switched off at random in each training step, so that the
    # network can't rely on any one of them. It's a strong guard against memorizing.
    "dropout": [0.0, 0.2, 0.4],
    # How big each nudge to the weights is. Too big and training is unstable, too small
    # and it stops too soon to have learned much.
    "learning_rate": [0.001, 0.003],
    # A gentle pull on every weight towards 0, which favours simpler patterns. AdamW applies
    # it in proportion to the learning rate, which is why these numbers look large.
    "weight_decay": [0.0, 0.01, 0.1],
}

# Examples per nudge. Small enough for many nudges per epoch, and large enough that each
# one isn't steered by a few cases.
BATCH_SIZE = 64
# An upper limit that stopping should reach well before.
MAX_EPOCHS = 200
# Epochs without a new best before giving up. The inner error is noisy from epoch to
# epoch, so stopping at the first rise would stop far too early.
PATIENCE = 20
# An unusual batch can make one nudge huge, which LSTMs are prone to. This caps its size.
MAX_GRADIENT_NORM = 1.0

# The share of a fold's learning storms kept back for early stopping, and the seed that
# picks them. The seed stays the same for every model seed, so what differs between the 5
# members is only the starting weights, the shuffling and the dropout.
INNER_SHARE = 0.15
INNER_SEED = 42


class TrackLSTM(nn.Module):
    """LSTM layers over the sequence, then the fixed inputs, then one output head."""

    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=len(STEP_COLUMNS),
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # This one works between layers, so it has no meaning with a single layer.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # The dropout here works with one layer as well, so the setting always means something.
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size + len(FIXED_COLUMNS), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(LEAD_HOURS) * N_COMPONENTS),
        )

    def forward(
        self,
        steps: torch.Tensor,
        lengths: torch.Tensor,
        fixed: torch.Tensor,
    ) -> torch.Tensor:
        """Return scaled changes in position, shape (forecasts, leads, components).

        Args:
            steps: shape (forecasts, steps, STEP_COLUMNS), scaled, with padding at the end.
            lengths: each sequence's real number of fixes.
            fixed: shape (forecasts, FIXED_COLUMNS), scaled.
        """
        memory, _ = self.lstm(steps)
        at_t0 = memory[torch.arange(len(lengths)), lengths - 1]
        output = self.head(torch.cat([at_t0, fixed], dim=1))
        return output.view(-1, len(LEAD_HOURS), N_COMPONENTS)


@dataclass(frozen=True)
class TargetScaler:
    """The mean and spread of the change in position, per lead and component."""

    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class FittedLSTM:
    """One fitted network with everything needed to use it.

    The scalers belong with the weights. A network only understands inputs scaled the way
    its own training rows were, so on test day they must come from here.
    """

    network: TrackLSTM
    scaler: Scaler
    target_scaler: TargetScaler
    settings: dict
    seed: int
    best_epoch: int


def pick_device() -> str:
    """Return "cuda" if PyTorch can really run an LSTM on a GPU here, otherwise "cpu".

    It tries a tiny LSTM, because seeing a GPU isn't the same as being able to use it.
    A too-old GPU raises a RuntimeError. A CPU-only PyTorch raises an AssertionError.
    """
    if not torch.cuda.is_available():
        return "cpu"
    try:
        nn.LSTM(1, 1).to("cuda")(torch.zeros(1, 1, 1, device="cuda"))
    except (RuntimeError, AssertionError):
        return "cpu"
    return "cuda"


def _fix_randomness(seed: int) -> None:
    """Make the starting weights and dropout repeat, and rule out unrepeatable GPU shortcuts."""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def sample_settings() -> list[dict]:
    """Return the combinations the search will try, always the same ones.

    Twelve draws aren't many, and fully random ones gave hidden size 16 eight times and 32
    only once, which would barely test that setting. So each setting's values are repeated
    evenly and shuffled separately: every value is tried equally often, and the pairings
    are still random. It's a balanced version of the random sampling in ``xgb_model.py``.
    """
    rng = np.random.default_rng(SEED)
    columns = {
        name: rng.permutation(np.resize(values, N_COMBINATIONS)).tolist()
        for name, values in SEARCH_SPACE.items()
    }
    return [{name: column[number] for name, column in columns.items()} for number in range(N_COMBINATIONS)]


def masked_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the mean squared error over the answers that exist, each lead counting equally.

    A blank target would make the loss blank, so it's replaced with 0 before subtracting.
    The value doesn't matter, since the mask then sets that error to exactly 0, which
    gives it no pull on any weight.

    A case missing +72 h still teaches the network about +24 and +48 h. There are fewer
    cases at the long leads, so averaging each lead by itself keeps them from counting
    for less, as they don't in the score that picks the primary model.

    Args:
        predicted: shape (forecasts, leads, components).
        target: the same shape, blank where the storm has no fix at that lead.
    """
    mask = (~torch.isnan(target)).to(predicted.dtype)
    squared = (predicted - torch.nan_to_num(target)) ** 2 * mask
    answers = mask.sum(dim=(0, 2))
    present = answers > 0
    return (squared.sum(dim=(0, 2))[present] / answers[present]).mean()


def _targets(rows: pd.DataFrame, truth: pd.DataFrame) -> np.ndarray:
    """Return the change in position for each row, blank where a lead has no truth."""
    targets = np.full((len(rows), len(LEAD_HOURS), N_COMPONENTS), np.nan)
    numbered = rows.assign(position=np.arange(len(rows)))
    for lead_index, lead_h in enumerate(LEAD_HOURS):
        cases = with_truth_at(numbered, truth, lead_h)
        positions = cases["position"].to_numpy()
        targets[positions, lead_index, 0] = (cases["truth_lat"] - cases["lat"]).to_numpy()
        targets[positions, lead_index, 1] = (cases["truth_lon"] - cases["lon"]).to_numpy()
    return targets


def inner_storms(storms) -> set[str]:
    """Return the storms kept back for early stopping, always the same ones for the same input.

    Storms are chosen whole, because rows from one storm are nearly identical six hours
    apart. Choosing rows would put almost the same case on both sides, and the inner
    error would keep falling long after the network had started memorizing.

    Args:
        storms: the storm IDs of the rows learned from. Repeats and order don't matter.

    Returns:
        About INNER_SHARE of the storms, and at least one.
    """
    unique = np.sort(pd.unique(np.asarray(storms)))
    count = max(1, round(INNER_SHARE * len(unique)))
    return set(np.random.default_rng(INNER_SEED).choice(unique, count, replace=False).tolist())


def _tensors(
    sequences: Sequences,
    scaler: Scaler,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return scaled steps, lengths and fixed inputs, ready for the network."""
    steps, fixed = scale(sequences, scaler)
    return (
        torch.as_tensor(steps, dtype=torch.float32, device=device),
        torch.as_tensor(sequences.lengths, dtype=torch.long, device=device),
        torch.as_tensor(fixed, dtype=torch.float32, device=device),
    )


def fit_lstm(
    rows: pd.DataFrame,
    truth: pd.DataFrame,
    sequences: Sequences,
    settings: dict,
    device: str,
    seed: int = SEED,
) -> FittedLSTM:
    """Fit one network on exactly the rows given, stopping early on some of their storms.

    Both scalers are fitted here, on these rows only, so a fold's held-out rows never
    reach them. Rows with no truth at any lead still help the scaler, since they're real
    inputs, but they're left out of training with nothing to learn from.

    Args:
        rows: the start times to learn from.
        truth: the table from ``samples.make_truth``.
        sequences: the table of inputs. It may hold other rows, which are never read here.
        settings: one combination from ``sample_settings``.
        device: "cpu" or "cuda", from ``pick_device``.
        seed: fixes the random parts of training.

    Returns:
        The fitted network, with the weights from its best epoch, and its scalers.
    """
    _fix_randomness(seed)

    chosen = select(sequences, rows)
    scaler = fit_scaler(chosen)
    steps, lengths, fixed = _tensors(chosen, scaler, device)

    change = _targets(rows, truth)
    target_scaler = TargetScaler(
        mean=np.nanmean(change, axis=0),
        std=np.nanstd(change, axis=0),
    )
    targets = torch.as_tensor(
        (change - target_scaler.mean) / target_scaler.std, dtype=torch.float32, device=device
    )

    has_answer = ~np.isnan(change).all(axis=(1, 2))
    storms = rows["SID"].to_numpy()
    in_inner = np.isin(storms, list(inner_storms(storms[has_answer])))
    fit_on = torch.as_tensor(np.flatnonzero(has_answer & ~in_inner), device=device)
    stop_on = torch.as_tensor(np.flatnonzero(has_answer & in_inner), device=device)

    network = TrackLSTM(
        settings["hidden_size"], settings["num_layers"], settings["dropout"]
    ).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    shuffler = torch.Generator().manual_seed(seed)

    best_error = float("inf")
    best_weights = None
    best_epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        network.train()
        for batch in torch.randperm(len(fit_on), generator=shuffler).split(BATCH_SIZE):
            chosen_rows = fit_on[batch.to(device)]
            optimizer.zero_grad()
            predicted = network(steps[chosen_rows], lengths[chosen_rows], fixed[chosen_rows])
            masked_loss(predicted, targets[chosen_rows]).backward()
            nn.utils.clip_grad_norm_(network.parameters(), MAX_GRADIENT_NORM)
            optimizer.step()

        network.eval()
        with torch.no_grad():
            predicted = network(steps[stop_on], lengths[stop_on], fixed[stop_on])
            error = masked_loss(predicted, targets[stop_on]).item()

        if error < best_error:
            best_error, best_epoch = error, epoch
            best_weights = {name: value.cpu().clone() for name, value in network.state_dict().items()}
        elif epoch - best_epoch >= PATIENCE:
            break

    network.load_state_dict(best_weights)
    return FittedLSTM(network.cpu().eval(), scaler, target_scaler, dict(settings), seed, best_epoch)


def predict_lstm(
    fitted: FittedLSTM,
    rows: pd.DataFrame,
    sequences: Sequences,
    name: str = MODEL_NAME,
) -> pd.DataFrame:
    """Return one network's forecasts for every row.

    Args:
        fitted: a result of ``fit_lstm`` or ``load_fitted``.
        rows: the start times to forecast. Pass only the split being scored.
        sequences: the table of inputs, holding at least these rows.
        name: the name to give these forecasts in the model column.

    Returns:
        Forecasts in the format ``scoring.score`` takes, at each lead time.
    """
    steps, lengths, fixed = _tensors(select(sequences, rows), fitted.scaler, "cpu")
    with torch.no_grad():
        scaled = fitted.network(steps, lengths, fixed).numpy().astype(float)
    change = scaled * fitted.target_scaler.std + fitted.target_scaler.mean

    forecasts = []
    for lead_index, lead_h in enumerate(LEAD_HOURS):
        table = rows[["SID", "init_time"]].assign(
            model=name,
            lead_h=lead_h,
            lat=rows["lat"].to_numpy() + change[:, lead_index, 0],
            lon=rows["lon"].to_numpy() + change[:, lead_index, 1],
        )
        forecasts.append(table[FORECAST_COLUMNS])
    return pd.concat(forecasts, ignore_index=True)


def _seed_path(directory: Path, seed: int) -> Path:
    return Path(directory) / f"seed{seed}.pt"


def save_fitted(fitted: FittedLSTM, path: str | Path) -> None:
    """Write one fitted network and its scalers to one file.

    Everything is stored as tensors, lists and plain numbers, so it can be read back with
    PyTorch's safe loader, which refuses to run code hidden in a file. The file is replaced
    in one step, so a session dying mid-write can't leave half of it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    unfinished = path.with_suffix(".tmp")
    torch.save(
        {
            "weights": fitted.network.state_dict(),
            "scaler": {name: value.tolist() for name, value in vars(fitted.scaler).items()},
            "target_scaler": {
                name: value.tolist() for name, value in vars(fitted.target_scaler).items()
            },
            "settings": fitted.settings,
            "seed": fitted.seed,
            "best_epoch": fitted.best_epoch,
        },
        unfinished,
    )
    unfinished.replace(path)


def load_fitted(path: str | Path) -> FittedLSTM:
    """Read one fitted network back, with its scalers, ready to forecast."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    settings = saved["settings"]
    network = TrackLSTM(settings["hidden_size"], settings["num_layers"], settings["dropout"])
    network.load_state_dict(saved["weights"])
    return FittedLSTM(
        network=network.eval(),
        scaler=Scaler(**{name: np.array(value) for name, value in saved["scaler"].items()}),
        target_scaler=TargetScaler(
            **{name: np.array(value) for name, value in saved["target_scaler"].items()}
        ),
        settings=settings,
        seed=saved["seed"],
        best_epoch=saved["best_epoch"],
    )


def fit_seeds(
    rows: pd.DataFrame,
    truth: pd.DataFrame,
    sequences: Sequences,
    settings: dict,
    device: str,
    seeds: tuple[int, ...] = SEEDS,
    directory: str | Path | None = None,
    resume: bool = False,
) -> dict[int, FittedLSTM]:
    """Fit one network per seed, on the same rows and settings.

    With ``directory`` set, each network is written as soon as it's fitted, so a dropped
    session loses one fit at most. Resuming has to be asked for, as in
    ``cv.search_settings``. A saved network with different settings isn't reused.

    Args:
        rows: the start times to learn from.
        truth: the table from ``samples.make_truth``.
        sequences: the table of inputs.
        settings: one combination from ``sample_settings``.
        device: "cpu" or "cuda", from ``pick_device``.
        seeds: the seeds to fit.
        directory: a folder to write each network to, in the data directory.
        resume: use the networks already in ``directory``.

    Returns:
        {seed: fitted network}, to pass to ``predict_seeds``.
    """
    fitted = {}
    for seed in seeds:
        path = _seed_path(directory, seed) if directory else None
        if resume and path and path.exists():
            saved = load_fitted(path)
            if saved.settings == settings:
                fitted[seed] = saved
                continue
            print(f"  {path} was fitted with other settings, so it's being refitted")
        fitted[seed] = fit_lstm(rows, truth, sequences, settings, device, seed)
        if path:
            save_fitted(fitted[seed], path)
    return fitted


def predict_seeds(
    fitted: dict[int, FittedLSTM],
    rows: pd.DataFrame,
    sequences: Sequences,
    name: str = MODEL_NAME,
) -> pd.DataFrame:
    """Return the average of each seed's forecast positions.

    Training has several random parts: the starting weights, the order of the examples
    and the dropout. Averaging positions, rather than picking a good seed, gives that
    luck nowhere to hide. Latitude and longitude are averaged separately, as in
    ``consensus.consensus``.

    Args:
        fitted: the result of ``fit_seeds``, ``fit_final`` or ``load_final``.
        rows: the start times to forecast. Pass only the split being scored.
        sequences: the table of inputs, holding at least these rows.
        name: the name to give these forecasts in the model column.

    Returns:
        Forecasts in the format ``scoring.score`` takes, at each lead time.
    """
    # Each run lists the same cases in the same order, so they can be averaged by position.
    runs = [predict_lstm(model, rows, sequences, name) for model in fitted.values()]
    return runs[0].assign(
        lat=np.mean([run["lat"] for run in runs], axis=0),
        lon=np.mean([run["lon"] for run in runs], axis=0),
    )


def _left_out_block(learning: pd.DataFrame) -> str:
    """Return the name of the block missing from a fold's learning rows."""
    return next(name for name in FOLD_NAMES if name not in set(learning["fold"]))


def lstm_forecaster(
    settings: dict,
    device: str,
    sequences: Sequences,
    seeds: tuple[int, ...] = (SEED,),
    name: str = MODEL_NAME,
    directory: str | Path | None = None,
    resume: bool = False,
) -> cv.Forecaster:
    """Return the LSTM with these settings, in the form ``cv.out_of_fold`` takes.

    The sequences are built once, before the folds, because they hold nothing after t0
    and need no fitting. The scalers do need fitting, so that happens inside each fold,
    on that fold's learning rows.

    One seed is the default because that's what the search fits. Pass ``SEEDS`` to measure
    a model.

    Args:
        settings: one combination from ``sample_settings``.
        device: "cpu" or "cuda", from ``pick_device``.
        sequences: the table of inputs, for every row the folds will use.
        seeds: the seeds to fit and average.
        name: the name to give these forecasts in the model column.
        directory: a folder to save networks in, with one folder per block inside it.
        resume: use the networks already saved there.
    """

    def forecast(learning: pd.DataFrame, truth: pd.DataFrame, held_out: pd.DataFrame) -> pd.DataFrame:
        block_dir = Path(directory) / _left_out_block(learning) if directory else None
        fitted = fit_seeds(learning, truth, sequences, settings, device, seeds, block_dir, resume)
        return predict_seeds(fitted, held_out, sequences, name)

    return forecast


def search_cv(
    rows: pd.DataFrame,
    truth: pd.DataFrame,
    sequences: Sequences,
    device: str | None = None,
    checkpoint: str | Path | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    """Try each combination under cross-validation and rank them, best first.

    ``cv.search_settings`` does the work, and says how combinations are judged and saved.
    It's the same rule as for XGBoost, on one seed. The winner is partly the luckiest of
    12, so its score leans slightly optimistic.

    Args:
        rows: the table from ``cv.model_rows``.
        truth: the table from ``samples.make_truth``.
        sequences: the table of inputs for those rows.
        device: "cpu" or "cuda". Leave it out to use ``pick_device``.
        checkpoint: a CSV file to write progress to, in the data directory.
        resume: skip the combinations already in the checkpoint file.

    Returns:
        Every combination tried with its error at each lead time, best first.
    """
    device = device or pick_device()
    print(f"Searching {N_COMBINATIONS} combinations on {device}, 5 folds each")
    return cv.search_settings(
        rows,
        truth,
        lambda settings: lstm_forecaster(settings, device, sequences),
        sample_settings(),
        checkpoint,
        resume,
    )


def best_settings(search_table: pd.DataFrame) -> dict:
    """Return the winning settings from a search table, ready to fit with.

    A table row holds every value as a float, so each is turned back into the type the
    search space gives it. A network refuses 32.0 as its hidden size.
    """
    winner = search_table.iloc[0]
    return {name: type(values[0])(winner[name]) for name, values in SEARCH_SPACE.items()}


def fit_final(
    rows: pd.DataFrame,
    truth: pd.DataFrame,
    sequences: Sequences,
    settings: dict,
    device: str,
    directory: str | Path | None = None,
    resume: bool = False,
) -> dict[int, FittedLSTM]:
    """Refit chosen settings with every seed, on every development season.

    It uses the same steps as a fold, handed all five blocks, so it's the model that was
    measured: the scalers come from all of these rows, and early stopping watches about
    15% of these storms. So no test storm plays any part.

    The filter is inside this function, so passing the full table can't reach test.

    Args:
        rows: a table with a "fold" column, such as the one from ``cv.model_rows``.
        truth: the table from ``samples.make_truth``.
        sequences: the table of inputs.
        settings: the chosen settings, from ``best_settings``.
        device: "cpu" or "cuda", from ``pick_device``.
        directory: a folder to write each network to as it's fitted.
        resume: use the networks already in ``directory``.

    Returns:
        {seed: fitted network}, to pass to ``predict_seeds``.
    """
    return fit_seeds(
        development(rows), truth, sequences, settings, device, SEEDS, directory, resume
    )


def save_final(
    fitted: dict[int, FittedLSTM],
    search_table: pd.DataFrame,
    directory: str | Path,
) -> None:
    """Write every seed's network and the search results.

    Args:
        fitted: the result of ``fit_final``.
        search_table: the table from ``search_cv``, kept as a record of what was tried.
        directory: where to write.
    """
    directory = Path(directory)
    for seed, model in fitted.items():
        save_fitted(model, _seed_path(directory, seed))
    search_table.to_csv(directory / "search_results.csv", index=False)
    print(f"Saved {len(fitted)} networks to {directory}")


def load_final(directory: str | Path) -> dict[int, FittedLSTM]:
    """Read every seed's network back. A missing seed stops the run.

    A final forecast from four seeds would look normal and quietly not be the model that
    was measured, so all of them have to be there.

    Args:
        directory: where they were written.

    Returns:
        {seed: fitted network}, to pass to ``predict_seeds``.
    """
    return {seed: load_fitted(_seed_path(directory, seed)) for seed in SEEDS}
