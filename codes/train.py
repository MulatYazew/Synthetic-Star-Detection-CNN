"""
Training pipeline — Synthetic-Star U-Net on Apple M4 (MPS).

Trains the U-Net with three loss functions (BCE, Dice, BCE+Dice), selects the
best model by validation IoU, and saves it to models/synthetic_model.pt.
Plots and CSVs → results/training/, checkpoints → results/checkpoints/,
comparisons → results/performance_study/, reports → results/evaluation/.

Usage (from project root):
    python codes/train.py

Run the num_workers benchmark once to find the optimal DataLoader setting:
    python codes/train.py --bench-workers

Key design choices
──────────────────
  - MPS device (Apple Metal GPU) via torch.backends.mps: ~5–6× faster
    than CPU for this model size.
  - base_filters=32 recommended default (7.8 M params); pass --full-model
    for base_filters=64 (31 M params).
  - Batch size 64 for good MPS occupancy.
  - DataLoader num_workers=0: for in-memory arrays on macOS the "spawn"
    context adds ~0.5 s overhead per epoch; inline loading is faster.
  - No autocast / mixed precision: MPS fp16/bf16 crashes with BCELoss.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from codes.dataset import (generate_dataset, generate_and_save_flat,
                                stratified_split_indices, prepare_dataset)
    from codes.model import UNetStarFinder, binary_iou
    from codes.losses import get_loss
    from codes.config import (
        MODELS_DIR, DATASET_DIR,
        TRAINING_RESULTS_DIR, CHECKPOINTS_DIR, PERFORMANCE_STUDY_DIR,
        EVALUATION_RESULTS_DIR,
        SYNTHETIC_MODEL_PATH, CACHE_PATH,
        IMAGE_SIZE, N_TRAIN, N_VAL, BATCH_SIZE, BASE_FILTERS, EPOCHS, LR, SEED,
        N_SAMPLES, FORCE_REGENERATE,
        EARLY_STOP_PATIENCE, LR_REDUCE_PATIENCE, LR_REDUCE_FACTOR, LR_MIN,
        LOSS_NAMES, LOSS_DISPLAY,
    )
except ImportError:
    from dataset import (generate_dataset, generate_and_save_flat,
                         stratified_split_indices, prepare_dataset)
    from model import UNetStarFinder, binary_iou
    from losses import get_loss
    from config import (
        MODELS_DIR, DATASET_DIR,
        TRAINING_RESULTS_DIR, CHECKPOINTS_DIR, PERFORMANCE_STUDY_DIR,
        EVALUATION_RESULTS_DIR,
        SYNTHETIC_MODEL_PATH, CACHE_PATH,
        IMAGE_SIZE, N_TRAIN, N_VAL, BATCH_SIZE, BASE_FILTERS, EPOCHS, LR, SEED,
        N_SAMPLES, FORCE_REGENERATE,
        EARLY_STOP_PATIENCE, LR_REDUCE_PATIENCE, LR_REDUCE_FACTOR, LR_MIN,
        LOSS_NAMES, LOSS_DISPLAY,
    )


# ─────────────────────────────────────────────────────────────────── #
# Device selection
# ─────────────────────────────────────────────────────────────────── #

def get_device() -> torch.device:
    """Return MPS if available (Apple Silicon GPU), else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────── #
# Dataset helpers
# ─────────────────────────────────────────────────────────────────── #

def load_or_generate(n_train: int, n_val: int, image_size: int,
                     cache: bool = True,
                     force_regenerate: bool = False) -> tuple[np.ndarray, ...]:
    """
    Return (X_train, Y_train, X_val, Y_val).

    1. Load the stratified split from the .npz cache when shapes match
       and force_regenerate is False (fastest path).
    2. Load or generate the flat dataset via prepare_dataset(), then split.
       prepare_dataset() reuses an existing dataset unless force_regenerate
       is True, in which case it overwrites with a freshly seeded dataset.
    """
    if force_regenerate and CACHE_PATH.exists():
        CACHE_PATH.unlink()
        print("  Split cache cleared (force_regenerate=True).")

    if cache and not force_regenerate and CACHE_PATH.exists():
        t0 = time.perf_counter()
        d = np.load(CACHE_PATH)
        if ("X_train" in d and "X_val" in d and
                d["X_train"].shape == (n_train, image_size, image_size, 1) and
                d["X_val"].shape   == (n_val,   image_size, image_size, 1)):
            X_tr, Y_tr = d["X_train"], d["Y_train"]
            X_v,  Y_v  = d["X_val"],   d["Y_val"]
            print(f"  Loaded split from cache in {time.perf_counter()-t0:.2f}s")
            return X_tr, Y_tr, X_v, Y_v
        print("  Cache shape mismatch — reloading from flat files …")

    t0 = time.perf_counter()
    X_all, Y_all, meta = prepare_dataset(
        dataset_root=DATASET_DIR,
        n_samples=N_SAMPLES,
        image_size=image_size,
        seed=SEED,
        force_regenerate=force_regenerate,
    )
    n_stars_all = np.array([r["n_stars"] for r in meta])
    n_test = max(0, len(X_all) - n_train - n_val)
    train_idx, val_idx, _ = stratified_split_indices(
        n_stars_all, n_train, n_val, n_test, seed=SEED)
    X_tr, Y_tr = X_all[train_idx], Y_all[train_idx]
    X_v,  Y_v  = X_all[val_idx],   Y_all[val_idx]
    print(f"  Stratified split in {time.perf_counter()-t0:.2f}s")

    if cache:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE_PATH,
                            X_train=X_tr, Y_train=Y_tr,
                            X_val=X_v,    Y_val=Y_v)
        print(f"  Split cached → {CACHE_PATH}")

    return X_tr, Y_tr, X_v, Y_v


def make_loader(X: np.ndarray, Y: np.ndarray,
                batch_size: int,
                shuffle: bool,
                num_workers: int = 0) -> DataLoader:
    """
    Build a DataLoader from NumPy arrays.

    pin_memory is left False: it is a CUDA-only optimisation and has no
    effect (or causes crashes) on MPS.
    num_workers=0 is faster than >0 for in-memory arrays on macOS because
    the "spawn" multiprocessing start method adds ~0.5 s overhead per epoch.
    Run --bench-workers to verify the optimal value on your machine.
    """
    xt = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()  # NHWC → NCHW
    yt = torch.from_numpy(Y).permute(0, 3, 1, 2).contiguous()
    dataset = TensorDataset(xt, yt)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )


# ─────────────────────────────────────────────────────────────────── #
# num_workers benchmark
# ─────────────────────────────────────────────────────────────────── #

def bench_num_workers(X: np.ndarray, Y: np.ndarray,
                      batch_size: int = 64) -> None:
    """
    Time one full pass through the dataset for num_workers in {0,2,4,6,8}.
    Recommended value for MacBook Air M4 (10 CPU cores): 0 or 2.
    """
    print("\nnum_workers benchmark (one full pass, batch_size={}):".format(batch_size))
    for nw in [0, 2, 4, 6, 8]:
        loader = make_loader(X, Y, batch_size=batch_size,
                             shuffle=False, num_workers=nw)
        for _ in loader:   # warmup pass
            pass
        t0 = time.perf_counter()
        n_batches = 0
        for _ in loader:
            n_batches += 1
        elapsed = time.perf_counter() - t0
        print(f"  num_workers={nw}: {elapsed:.3f}s  "
              f"({len(X)/elapsed:.0f} samp/s)  "
              f"{n_batches} batches")
    print()


# ─────────────────────────────────────────────────────────────────── #
# Training helpers
# ─────────────────────────────────────────────────────────────────── #

class EarlyStopper:
    def __init__(self, patience: int, mode: str = "max"):
        self.patience   = patience
        self.mode       = mode
        self.best       = -float("inf") if mode == "max" else float("inf")
        self.counter    = 0
        self.best_epoch = 0

    def step(self, value: float, epoch: int) -> bool:
        """Return True when training should stop."""
        improved = (value > self.best) if self.mode == "max" else (value < self.best)
        if improved:
            self.best       = value
            self.counter    = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
        return self.counter >= self.patience


class ReduceLROnPlateau:
    def __init__(self, optimizer: torch.optim.Optimizer,
                 factor: float = 0.5, patience: int = 5,
                 min_lr: float = 1e-6):
        self.opt      = optimizer
        self.factor   = factor
        self.patience = patience
        self.min_lr   = min_lr
        self.best     = float("inf")
        self.counter  = 0

    def step(self, val_loss: float) -> None:
        if val_loss < self.best:
            self.best    = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                for pg in self.opt.param_groups:
                    pg["lr"] = max(pg["lr"] * self.factor, self.min_lr)
                self.counter = 0


def run_epoch(model: nn.Module,
              loader: DataLoader,
              optimizer: torch.optim.Optimizer | None,
              device: torch.device,
              loss_fn,
              profile: bool = False) -> tuple[float, float, dict]:
    """
    Run one training or validation epoch.

    loss_fn signature: loss_fn(y_true, y_pred) → scalar tensor.
    Returns (mean_loss, mean_iou, timing_dict).
    timing_dict keys: data_s, forward_s, backward_s, optim_s
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss  = 0.0
    total_iou   = 0.0
    n_batches   = 0
    timing      = {"data_s": 0.0, "forward_s": 0.0,
                   "backward_s": 0.0, "optim_s": 0.0}

    t_data_start = time.perf_counter()

    for xb, yb in loader:
        if profile:
            if device.type == "mps":
                torch.mps.synchronize()
            timing["data_s"] += time.perf_counter() - t_data_start

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        t_fwd = time.perf_counter()
        pred  = model(xb)
        loss  = loss_fn(yb, pred)   # (y_true, y_pred) convention from losses.py

        if profile:
            if device.type == "mps":
                torch.mps.synchronize()
            timing["forward_s"] += time.perf_counter() - t_fwd

        if is_train:
            t_bwd = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if profile:
                if device.type == "mps":
                    torch.mps.synchronize()
                timing["backward_s"] += time.perf_counter() - t_bwd

            t_opt = time.perf_counter()
            optimizer.step()

            if profile:
                if device.type == "mps":
                    torch.mps.synchronize()
                timing["optim_s"] += time.perf_counter() - t_opt

        with torch.no_grad():
            iou = binary_iou(pred, yb)

        total_loss += loss.item()
        total_iou  += iou.item()
        n_batches  += 1
        t_data_start = time.perf_counter()

    return total_loss / n_batches, total_iou / n_batches, timing


def _save_training_csv(history: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_iou",
                         "val_loss", "val_iou", "epoch_time_s"])
        for ep, vals in enumerate(zip(
                history["train_loss"], history["train_iou"],
                history["val_loss"],   history["val_iou"],
                history["epoch_time_s"]), 1):
            writer.writerow([ep] + [round(v, 6) for v in vals])
    print(f"  Training metrics → {path}")


# ─────────────────────────────────────────────────────────────────── #
# Single-loss training run
# ─────────────────────────────────────────────────────────────────── #

def train_one_loss(
    loss_name:       str,
    train_loader:    DataLoader,
    val_loader:      DataLoader,
    device:          torch.device,
    n_train:         int,
    n_val:           int,
    base_filters:    int   = BASE_FILTERS,
    epochs:          int   = EPOCHS,
    lr:              float = LR,
    checkpoint_path: Path  = None,
    verbose:         bool  = True,
) -> tuple[dict, float]:
    """
    Train a fresh U-Net for one loss function.

    Returns (history_dict, best_val_iou).
    If checkpoint_path is given, the best model state is saved there.
    """
    loss_fn = get_loss(loss_name)
    label   = LOSS_DISPLAY.get(loss_name, loss_name)

    model     = UNetStarFinder(base_filters=base_filters, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_sched  = ReduceLROnPlateau(optimizer, factor=LR_REDUCE_FACTOR,
                                   patience=LR_REDUCE_PATIENCE, min_lr=LR_MIN)
    stopper   = EarlyStopper(patience=EARLY_STOP_PATIENCE, mode="max")

    history: dict[str, list] = {
        "train_loss": [], "train_iou": [],
        "val_loss":   [], "val_iou":   [],
        "epoch_time_s": [],
    }
    best_val_iou = -float("inf")
    best_state   = None

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Loss: {label:<14} | params: {model.param_count()/1e6:.2f} M "
              f"| device: {device}")
        print(f"{'─'*60}")

    epoch_bar = tqdm(range(1, epochs + 1), desc=f"[{label}]",
                     unit="epoch", disable=not verbose)
    for epoch in epoch_bar:
        t_epoch = time.perf_counter()
        profile = (epoch == 1)

        tr_loss, tr_iou, tr_timing = run_epoch(
            model, train_loader, optimizer, device, loss_fn, profile=profile)
        val_loss, val_iou, _ = run_epoch(
            model, val_loader, None, device, loss_fn)

        epoch_s = time.perf_counter() - t_epoch
        lr_now  = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["train_iou"].append(tr_iou)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["epoch_time_s"].append(epoch_s)

        epoch_bar.set_postfix(
            loss=f"{tr_loss:.4f}", iou=f"{tr_iou:.4f}",
            val_iou=f"{val_iou:.4f}", lr=f"{lr_now:.2e}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_state   = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)

        lr_sched.step(val_loss)

        if verbose:
            throughput = (n_train + n_val) / epoch_s
            tqdm.write(
                f"  [{label}] Ep {epoch:3d}/{epochs}  "
                f"loss {tr_loss:.4f}  iou {tr_iou:.4f}  "
                f"val_loss {val_loss:.4f}  val_iou {val_iou:.4f}  "
                f"lr {lr_now:.2e}  {epoch_s:.1f}s  ({throughput:.0f} samp/s)"
            )
            if profile:
                t = tr_timing
                tqdm.write(
                    f"    Profile:  "
                    f"data {t['data_s']:.3f}s  "
                    f"fwd {t['forward_s']:.3f}s  "
                    f"bwd {t['backward_s']:.3f}s  "
                    f"opt {t['optim_s']:.3f}s"
                )

        if stopper.step(val_iou, epoch):
            if verbose:
                tqdm.write(f"  [{label}] Early stop at epoch {epoch}. "
                           f"Best val IoU = {best_val_iou:.4f} "
                           f"(epoch {stopper.best_epoch})")
            break

    if verbose:
        print(f"  [{label}] ✓ Best val IoU: {best_val_iou:.4f}")

    return history, best_val_iou


# ─────────────────────────────────────────────────────────────────── #
# Plotting helpers
# ─────────────────────────────────────────────────────────────────── #

def plot_history(hist: dict, loss_name: str, save_dir: Path) -> None:
    """Save individual training/validation history plot for one loss function."""
    label = LOSS_DISPLAY.get(loss_name, loss_name)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(hist["train_loss"], label="train")
    axes[0].plot(hist["val_loss"],   label="validation")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss — {label}")
    axes[0].legend()

    axes[1].plot(hist["train_iou"], label="train")
    axes[1].plot(hist["val_iou"],   label="validation")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("IoU")
    axes[1].set_title(f"IoU — {label}")
    axes[1].legend()

    plt.suptitle(f"Synthetic U-Net ({label}) — training history", fontsize=12)
    plt.tight_layout()
    out = save_dir / f"{loss_name}_training_history.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  History plot → {out}")


def plot_loss_comparison(all_histories: dict[str, dict], save_dir: Path) -> None:
    """Save a comparison plot of validation IoU and validation loss for all 3 losses."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for loss_name, hist in all_histories.items():
        label = LOSS_DISPLAY.get(loss_name, loss_name)
        axes[0].plot(hist["val_iou"],  label=label)
        axes[1].plot(hist["val_loss"], label=label)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation IoU")
    axes[0].set_title("Validation IoU — loss function comparison")
    axes[0].legend()

    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation Loss")
    axes[1].set_title("Validation Loss — loss function comparison")
    axes[1].legend()

    plt.suptitle("Synthetic Pipeline — Loss Function Comparison", fontsize=12)
    plt.tight_layout()
    out = save_dir / "loss_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Comparison plot → {out}")


def save_comparison_report(
    results_per_loss: dict[str, tuple],
    save_dir: Path,
    best_loss_name: str,
    model_save_path: Path,
) -> None:
    """Save CSV table and text summary of the loss function comparison."""
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / "loss_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["loss_fn", "display_name", "best_val_iou", "epochs_trained"])
        for loss_name, (hist, best_iou) in results_per_loss.items():
            writer.writerow([
                loss_name,
                LOSS_DISPLAY.get(loss_name, loss_name),
                round(best_iou, 6),
                len(hist["val_iou"]),
            ])
    print(f"  Comparison CSV → {csv_path}")

    txt_path = save_dir / "best_model_report.txt"
    lines = [
        "=== Synthetic Pipeline — Loss Function Comparison ===",
        "",
        f"{'Loss Function':<22}{'Best Val IoU':>14}{'Epochs':>10}",
        "─" * 46,
    ]
    for loss_name, (hist, best_iou) in results_per_loss.items():
        label  = LOSS_DISPLAY.get(loss_name, loss_name)
        marker = "  ← SELECTED" if loss_name == best_loss_name else ""
        lines.append(
            f"{label:<22}{best_iou:>14.4f}{len(hist['val_iou']):>10}{marker}"
        )
    best_iou_val = results_per_loss[best_loss_name][1]
    lines += [
        "─" * 46,
        "",
        f"Best loss function : {LOSS_DISPLAY.get(best_loss_name, best_loss_name)}",
        f"Best val IoU       : {best_iou_val:.4f}",
        f"Model saved to     : {model_save_path}",
    ]
    txt_path.write_text("\n".join(lines))

    print(f"  Report → {txt_path}")
    print()
    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────────── #
# Main training orchestrator
# ─────────────────────────────────────────────────────────────────── #

def train_synthetic(
    n_train:          int   = N_TRAIN,
    n_val:            int   = N_VAL,
    image_size:       int   = IMAGE_SIZE,
    batch_size:       int   = BATCH_SIZE,
    epochs:           int   = EPOCHS,
    base_filters:     int   = BASE_FILTERS,
    lr:               float = LR,
    model_path:       Path  = SYNTHETIC_MODEL_PATH,
    num_workers:      int   = 0,
    use_cache:        bool  = True,
    loss_names:       list  = None,
    verbose:          bool  = True,
    force_regenerate: bool  = FORCE_REGENERATE,
) -> dict:
    """
    Train U-Net on synthetic data for each loss function, select the best
    by validation IoU, and save it to model_path.

    Returns dict mapping loss_name → (history, best_val_iou).
    Plots/CSVs → results/training/, checkpoints → results/checkpoints/,
    comparisons → results/performance_study/, reports → results/evaluation/.
    """
    if loss_names is None:
        loss_names = LOSS_NAMES

    for d in [MODELS_DIR, TRAINING_RESULTS_DIR, CHECKPOINTS_DIR,
              PERFORMANCE_STUDY_DIR, EVALUATION_RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    device = get_device()
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Device      : {device}")
        print(f"  Base filters: {base_filters}")
        print(f"  Batch size  : {batch_size}")
        print(f"  Epochs      : {epochs}")
        print(f"  Loss fns    : {loss_names}")
        print(f"  Training    : {TRAINING_RESULTS_DIR}")
        print(f"  Checkpoints : {CHECKPOINTS_DIR}")
        print(f"  Perf. study : {PERFORMANCE_STUDY_DIR}")
        print(f"  Evaluation  : {EVALUATION_RESULTS_DIR}")
        print(f"{'='*60}\n")

    # ── Dataset ──────────────────────────────────────────────────── #
    if verbose:
        print("[1/3] Dataset")
    X_tr, Y_tr, X_v, Y_v = load_or_generate(
        n_train, n_val, image_size, use_cache, force_regenerate)
    if verbose:
        print(f"  X_train: {X_tr.shape}  Y_train: {Y_tr.shape}")
        print(f"  X_val  : {X_v.shape}   Y_val  : {Y_v.shape}")
        print(f"  Star pixel fraction (train): {Y_tr.mean():.4f}")

    train_loader = make_loader(X_tr, Y_tr, batch_size, shuffle=True,
                               num_workers=num_workers)
    val_loader   = make_loader(X_v,  Y_v,  batch_size, shuffle=False,
                               num_workers=num_workers)

    # ── Multi-loss training ──────────────────────────────────────── #
    if verbose:
        print(f"\n[2/3] Training {len(loss_names)} model(s)")

    results: dict[str, tuple] = {}
    for loss_name in loss_names:
        checkpoint = CHECKPOINTS_DIR / f"{loss_name}_best.pt"
        hist, best_iou = train_one_loss(
            loss_name       = loss_name,
            train_loader    = train_loader,
            val_loader      = val_loader,
            device          = device,
            n_train         = n_train,
            n_val           = n_val,
            base_filters    = base_filters,
            epochs          = epochs,
            lr              = lr,
            checkpoint_path = checkpoint,
            verbose         = verbose,
        )
        results[loss_name] = (hist, best_iou)
        _save_training_csv(hist,
                           TRAINING_RESULTS_DIR / f"{loss_name}_training.csv")
        plot_history(hist, loss_name, TRAINING_RESULTS_DIR)

    # ── Best model selection ─────────────────────────────────────── #
    best_loss = max(results, key=lambda k: results[k][1])

    if verbose:
        print(f"\n[3/3] Results")

    src = CHECKPOINTS_DIR / f"{best_loss}_best.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, model_path)
    print(f"  Best model ({LOSS_DISPLAY.get(best_loss, best_loss)}) saved → {model_path}")

    all_histories = {k: v[0] for k, v in results.items()}
    plot_loss_comparison(all_histories, PERFORMANCE_STUDY_DIR)
    save_comparison_report(results, EVALUATION_RESULTS_DIR, best_loss, model_path)

    return results


# ─────────────────────────────────────────────────────────────────── #
# CLI
# ─────────────────────────────────────────────────────────────────── #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthetic star U-Net — multi-loss training pipeline")
    p.add_argument("--bench-workers", action="store_true",
                   help="Benchmark num_workers values (0,2,4,6,8) then exit")
    p.add_argument("--full-model", action="store_true",
                   help="Use base_filters=64 (31 M params) instead of 32 (7.8 M)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs",     type=int, default=EPOCHS)
    p.add_argument("--lr",         type=float, default=LR)
    p.add_argument("--no-cache",    action="store_true",
                   help="Ignore the split cache (re-split from flat files each run)")
    p.add_argument("--force-regen", action="store_true",
                   help="Overwrite existing dataset and generate a fresh one "
                        "(overrides FORCE_REGENERATE in config.py)")
    p.add_argument("--loss", nargs="+", default=None,
                   choices=LOSS_NAMES,
                   help="Loss functions to train (default: all three)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.bench_workers:
        print("Generating dataset for benchmark …")
        X, Y = generate_dataset(N_TRAIN, IMAGE_SIZE, seed=SEED)
        bench_num_workers(X, Y, batch_size=args.batch_size)
        sys.exit(0)

    train_synthetic(
        batch_size        = args.batch_size,
        epochs            = args.epochs,
        base_filters      = 64 if args.full_model else BASE_FILTERS,
        lr                = args.lr,
        use_cache         = not args.no_cache,
        loss_names        = args.loss,
        force_regenerate  = args.force_regen,
    )
