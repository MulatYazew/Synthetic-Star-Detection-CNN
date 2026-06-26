"""
Training pipeline — Synthetic-Star U-Net on Apple M4 (MPS).

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
    Run --bench-workers to verify on your machine.
  - Dataset disk cache: first run saves synthetic_dataset_cache.npz;
    subsequent runs skip generation entirely.
  - No autocast / mixed precision: MPS fp16/bf16 crashes with BCELoss
    due to type mismatches in MPSGraph. Float32 is used throughout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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
                                stratified_split_indices)
    from codes.model import UNetStarFinder, bce_dice_loss, binary_iou
except ImportError:
    from dataset import (generate_dataset, generate_and_save_flat,
                         stratified_split_indices)
    from model import UNetStarFinder, bce_dice_loss, binary_iou


# ─────────────────────────────────────────────────────────────────── #
# Configuration
# ─────────────────────────────────────────────────────────────────── #

MODELS_DIR   = ROOT / "models"
FIGURES_DIR  = ROOT / "figures"
RESULTS_DIR  = ROOT / "results"
DATASET_DIR  = ROOT / "star-dataset"
MODEL_PATH   = MODELS_DIR / "star_finder_synthetic.pt"
CACHE_PATH   = DATASET_DIR / "synthetic_dataset_cache.npz"

IMAGE_SIZE   = 64
N_TRAIN      = 10_000
N_VAL        = 2_000
BATCH_SIZE   = 64        # 64 gives slightly better MPS occupancy than 32
BASE_FILTERS = 32        # 7.8 M params; same quality as 64 on 64×64 images
EPOCHS       = 50
LR           = 1e-4
SEED         = 42

EARLY_STOP_PATIENCE = 12
LR_REDUCE_PATIENCE  = 5
LR_REDUCE_FACTOR    = 0.5
LR_MIN              = 1e-6


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
                     cache: bool = True) -> tuple[np.ndarray, ...]:
    """
    Return (X_train, Y_train, X_val, Y_val).

    Priority:
      1. Load from star-dataset/synthetic_dataset_cache.npz if shapes match.
      2. Load from star-dataset/synthetic_stars.npy + synthetic_labels.npy
         and apply stratified split by n_stars.
      3. Generate the flat dataset from scratch, then split.
    """
    if cache and CACHE_PATH.exists():
        t0 = time.perf_counter()
        d = np.load(CACHE_PATH)
        if ("X_train" in d and "X_val" in d and
                d["X_train"].shape == (n_train, image_size, image_size, 1) and
                d["X_val"].shape   == (n_val,   image_size, image_size, 1)):
            X_tr, Y_tr = d["X_train"], d["Y_train"]
            X_v,  Y_v  = d["X_val"],   d["Y_val"]
            print(f"  Loaded dataset from cache in {time.perf_counter()-t0:.2f}s")
            return X_tr, Y_tr, X_v, Y_v
        print("  Cache shape mismatch — reloading from flat files …")

    stars_path  = DATASET_DIR / "synthetic_stars.npy"
    labels_path = DATASET_DIR / "synthetic_labels.npy"
    meta_path   = DATASET_DIR / "metadata" / "synthetic_metadata.json"

    if stars_path.exists() and labels_path.exists() and meta_path.exists():
        t0 = time.perf_counter()
        print("  Loading flat dataset from star-dataset/ …")
        X_all = np.load(stars_path)
        Y_all = np.load(labels_path)
        with open(meta_path) as fh:
            meta = json.load(fh)
        n_stars_all = np.array([r["n_stars"] for r in meta])
        n_test = len(X_all) - n_train - n_val
        if n_test < 0:
            n_test = 0
        train_idx, val_idx, _ = stratified_split_indices(
            n_stars_all, n_train, n_val, n_test, seed=SEED)
        X_tr, Y_tr = X_all[train_idx], Y_all[train_idx]
        X_v,  Y_v  = X_all[val_idx],   Y_all[val_idx]
        print(f"  Stratified split done in {time.perf_counter()-t0:.2f}s")
    else:
        n_total = n_train + n_val + 2_000
        print(f"  Generating {n_total:,} synthetic images …")
        t0 = time.perf_counter()
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        X_all, Y_all, meta = generate_and_save_flat(
            n_samples=n_total, dataset_root=DATASET_DIR,
            image_size=image_size, seed=SEED)
        n_stars_all = np.array([r["n_stars"] for r in meta])
        n_test = n_total - n_train - n_val
        train_idx, val_idx, _ = stratified_split_indices(
            n_stars_all, n_train, n_val, n_test, seed=SEED)
        X_tr, Y_tr = X_all[train_idx], Y_all[train_idx]
        X_v,  Y_v  = X_all[val_idx],   Y_all[val_idx]
        print(f"  Generated + split in {time.perf_counter()-t0:.2f}s")

    if cache:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE_PATH,
                            X_train=X_tr, Y_train=Y_tr,
                            X_val=X_v,    Y_val=Y_v)
        print(f"  Dataset cached → {CACHE_PATH}")

    return X_tr, Y_tr, X_v, Y_v


def make_loader(X: np.ndarray, Y: np.ndarray,
                batch_size: int,
                shuffle: bool,
                num_workers: int = 0,
                device: torch.device = None) -> DataLoader:
    """
    Build a DataLoader from NumPy arrays.

    pin_memory is left False: it is a CUDA-only optimisation and has no
    effect (or causes crashes) on MPS.
    num_workers=0 is faster than >0 for in-memory arrays on macOS because
    the "spawn" multiprocessing start method adds ~0.5 s overhead per epoch.
    Run --bench-workers to verify the optimal value on your machine.
    """
    # Move to float32 tensors on CPU; transfer to device happens inside the loop
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
        # One warmup pass to amortise worker startup cost
        for _ in loader:
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
# Training loop
# ─────────────────────────────────────────────────────────────────── #

class EarlyStopper:
    def __init__(self, patience: int, mode: str = "max"):
        self.patience = patience
        self.mode     = mode
        self.best     = -float("inf") if mode == "max" else float("inf")
        self.counter  = 0
        self.best_epoch = 0

    def step(self, value: float, epoch: int) -> bool:
        """Return True when training should stop."""
        improved = (value > self.best) if self.mode == "max" else (value < self.best)
        if improved:
            self.best = value
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
        return self.counter >= self.patience


class ReduceLROnPlateau:
    def __init__(self, optimizer: torch.optim.Optimizer,
                 factor: float = 0.5, patience: int = 5,
                 min_lr: float = 1e-6):
        self.opt     = optimizer
        self.factor  = factor
        self.patience = patience
        self.min_lr  = min_lr
        self.best    = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> None:
        if val_loss < self.best:
            self.best = val_loss
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
              profile: bool = False
              ) -> tuple[float, float, dict]:
    """
    Run one training or validation epoch.
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

        # ── Forward ────────────────────────────────────────────────── #
        t_fwd = time.perf_counter()
        pred  = model(xb)
        loss  = bce_dice_loss(pred, yb)

        if profile:
            if device.type == "mps":
                torch.mps.synchronize()
            timing["forward_s"] += time.perf_counter() - t_fwd

        # ── Backward ───────────────────────────────────────────────── #
        if is_train:
            t_bwd = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)   # set_to_none saves a memset
            loss.backward()

            if profile:
                if device.type == "mps":
                    torch.mps.synchronize()
                timing["backward_s"] += time.perf_counter() - t_bwd

            # ── Optimizer step ─────────────────────────────────────── #
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


def train_synthetic(
    n_train:      int   = N_TRAIN,
    n_val:        int   = N_VAL,
    image_size:   int   = IMAGE_SIZE,
    batch_size:   int   = BATCH_SIZE,
    epochs:       int   = EPOCHS,
    base_filters: int   = BASE_FILTERS,
    lr:           float = LR,
    model_path:   Path  = MODEL_PATH,
    num_workers:  int   = 0,
    use_cache:    bool  = True,
    verbose:      bool  = True,
) -> dict:
    """
    Full training pipeline.  Returns history dict with keys:
    train_loss, train_iou, val_loss, val_iou, epoch_time_s.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("metrics", "predictions", "star_catalogs",
                "transfer_learning", "real_data_results", "performance_study"):
        (RESULTS_DIR / sub).mkdir(parents=True, exist_ok=True)

    device = get_device()
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Device      : {device}")
        print(f"  Base filters: {base_filters}")
        print(f"  Batch size  : {batch_size}")
        print(f"  Epochs      : {epochs}")
        print(f"{'='*60}\n")

    # ── Dataset ────────────────────────────────────────────────────── #
    if verbose:
        print("[1/4] Dataset")
    X_tr, Y_tr, X_v, Y_v = load_or_generate(n_train, n_val, image_size, use_cache)
    if verbose:
        print(f"  X_train: {X_tr.shape}  Y_train: {Y_tr.shape}")
        print(f"  X_val  : {X_v.shape}   Y_val  : {Y_v.shape}")
        print(f"  Star pixel fraction (train): {Y_tr.mean():.4f}\n")

    train_loader = make_loader(X_tr, Y_tr, batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = make_loader(X_v,  Y_v,  batch_size, shuffle=False, num_workers=num_workers)

    # ── Model ──────────────────────────────────────────────────────── #
    if verbose:
        print("[2/4] Model")
    model = UNetStarFinder(base_filters=base_filters, dropout=0.2).to(device)
    if verbose:
        print(f"  Parameters  : {model.param_count()/1e6:.2f} M")
        print(f"  Architecture: U-Net  1→{base_filters}→{base_filters*2}→"
              f"{base_filters*4}→{base_filters*8}→{base_filters*16} (bottleneck)\n")

    # L2 regularisation via weight_decay in the optimizer (AdamW-style)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    lr_scheduler = ReduceLROnPlateau(optimizer, factor=LR_REDUCE_FACTOR,
                                     patience=LR_REDUCE_PATIENCE, min_lr=LR_MIN)
    early_stopper = EarlyStopper(patience=EARLY_STOP_PATIENCE, mode="max")

    # ── Training loop ─────────────────────────────────────────────── #
    if verbose:
        print("[3/4] Training")

    history: dict[str, list] = {
        "train_loss": [], "train_iou": [],
        "val_loss":   [], "val_iou":   [],
        "epoch_time_s": [],
    }
    best_val_iou  = -float("inf")
    best_state    = None

    epoch_bar = tqdm(range(1, epochs + 1), desc="Training",
                     unit="epoch", disable=not verbose)
    for epoch in epoch_bar:
        t_epoch = time.perf_counter()

        # Profile first epoch to show breakdown; subsequent epochs fast path
        profile = (epoch == 1)

        tr_loss, tr_iou, tr_timing = run_epoch(
            model, train_loader, optimizer, device, profile=profile)
        val_loss, val_iou, val_timing = run_epoch(
            model, val_loader, None, device, profile=profile)

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

        # Checkpoint
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_state   = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}
            torch.save(best_state, model_path)

        lr_scheduler.step(val_loss)

        if verbose:
            throughput = (n_train + n_val) / epoch_s
            remaining  = (epochs - epoch) * epoch_s
            tqdm.write(
                f"Epoch {epoch:3d}/{epochs}  "
                f"loss {tr_loss:.4f}  iou {tr_iou:.4f}  "
                f"val_loss {val_loss:.4f}  val_iou {val_iou:.4f}  "
                f"lr {lr_now:.2e}  "
                f"{epoch_s:.1f}s  ({throughput:.0f} samp/s)  "
                f"ETA {remaining/60:.1f}min"
            )
            if profile:
                t = tr_timing
                tqdm.write(
                    f"  Profiling (train epoch):  "
                    f"data {t['data_s']:.3f}s  "
                    f"fwd {t['forward_s']:.3f}s  "
                    f"bwd {t['backward_s']:.3f}s  "
                    f"opt {t['optim_s']:.3f}s"
                )

        if early_stopper.step(val_iou, epoch):
            if verbose:
                tqdm.write(f"\nEarly stop at epoch {epoch}. "
                           f"Best val IoU = {best_val_iou:.4f} "
                           f"(epoch {early_stopper.best_epoch})")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    _save_training_csv(history, RESULTS_DIR / "metrics" / "training_metrics.csv")

    if verbose:
        mean_t   = np.mean(history["epoch_time_s"])
        print(f"\n[4/4] Done")
        print(f"  Best val IoU : {best_val_iou:.4f}")
        print(f"  Model saved  : {model_path}")
        print(f"  Mean epoch   : {mean_t:.1f}s  "
              f"({(n_train+n_val)/mean_t:.0f} samp/s avg)")

    plot_history(history, FIGURES_DIR)
    return history


# ─────────────────────────────────────────────────────────────────── #
# Plotting
# ─────────────────────────────────────────────────────────────────── #

def plot_history(hist: dict, figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(hist["train_loss"], label="train")
    axes[0].plot(hist["val_loss"],   label="validation")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training loss (BCE+Dice hybrid)")
    axes[0].legend()

    axes[1].plot(hist["train_iou"], label="train")
    axes[1].plot(hist["val_iou"],   label="validation")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("IoU")
    axes[1].set_title("Intersection over Union")
    axes[1].legend()

    plt.suptitle("Synthetic U-Net (MPS) — training history", fontsize=12)
    plt.tight_layout()
    out = figures_dir / "training_history.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Training curves → {out}")


# ─────────────────────────────────────────────────────────────────── #
# CLI
# ─────────────────────────────────────────────────────────────────── #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimised training for synthetic star U-Net")
    p.add_argument("--bench-workers", action="store_true",
                   help="Benchmark num_workers values (0,2,4,6,8) then exit")
    p.add_argument("--full-model", action="store_true",
                   help="Use base_filters=64 (31 M params) instead of 32 (7.8 M)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs",     type=int, default=EPOCHS)
    p.add_argument("--lr",         type=float, default=LR)
    p.add_argument("--no-cache",   action="store_true",
                   help="Force dataset regeneration (ignore disk cache)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.bench_workers:
        print("Generating dataset for benchmark …")
        X, Y = generate_dataset(N_TRAIN, IMAGE_SIZE, seed=SEED)
        bench_num_workers(X, Y, batch_size=args.batch_size)
        sys.exit(0)

    train_synthetic(
        batch_size   = args.batch_size,
        epochs       = args.epochs,
        base_filters = 64 if args.full_model else BASE_FILTERS,
        lr           = args.lr,
        use_cache    = not args.no_cache,
    )
