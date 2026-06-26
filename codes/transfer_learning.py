"""
Transfer-learning pipeline for real astronomical data (MPS).

Strategy
--------
Phase 1  – Freeze encoder (enc1–enc4 + bottleneck).
           Train only the decoder on real labelled patches.
           LR = 1e-4, up to 30 epochs, early-stop on val IoU.

Phase 2  – Unfreeze all layers; fine-tune end-to-end at LR = 1e-5
           to prevent catastrophic forgetting of synthetic features.

Also provides `train_scratch_real` which trains a fresh U-Net on real
data only, for the three-way comparison in Task 4.

Usage (from project root):
    python codes/transfer_learning.py
"""

import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    from codes.dataset       import load_real_patches
    from codes.model import (UNetStarFinder, freeze_encoder, unfreeze_all,
                              bce_dice_loss, binary_iou)
except ImportError:
    from dataset import load_real_patches
    from model   import (UNetStarFinder, freeze_encoder, unfreeze_all,
                          bce_dice_loss, binary_iou)


# ─────────────────────────────────────────────────────────────────── #
# Config
# ─────────────────────────────────────────────────────────────────── #

BASE_DIR      = Path(__file__).resolve().parent.parent
MODELS_DIR    = BASE_DIR / "models"
FIGURES_DIR   = BASE_DIR / "figures"
RESULTS_DIR   = BASE_DIR / "results"
DATA_DIR      = BASE_DIR / "star-dataset"

SYNTHETIC_MODEL = MODELS_DIR / "star_finder_synthetic.pt"
TL_MODEL        = MODELS_DIR / "star_finder_tl.pt"
SCRATCH_MODEL   = MODELS_DIR / "star_finder_scratch.pt"

REAL_STARS_PATH  = DATA_DIR / "real_stars.npy"
REAL_LABELS_PATH = DATA_DIR / "real_stars_labels.npy"

PATCH_SIZE    = 64
BATCH_SIZE    = 16
PHASE1_EPOCHS = 30
PHASE2_EPOCHS = 20


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────── #
# Inner training loop
# ─────────────────────────────────────────────────────────────────── #

def _train_loop(model, X_tr, Y_tr, X_vl, Y_vl,
                lr, epochs, batch_size, device,
                patience=8, save_path=None, verbose=1):
    """Train/val loop with EarlyStopping + ReduceLROnPlateau. Returns history."""
    xt = torch.from_numpy(X_tr).permute(0, 3, 1, 2)
    yt = torch.from_numpy(Y_tr).permute(0, 3, 1, 2)
    xv = torch.from_numpy(X_vl).permute(0, 3, 1, 2)
    yv = torch.from_numpy(Y_vl).permute(0, 3, 1, 2)
    tr_loader = DataLoader(TensorDataset(xt, yt), batch_size=batch_size,
                           shuffle=True, num_workers=0)
    vl_loader = DataLoader(TensorDataset(xv, yv), batch_size=batch_size,
                           shuffle=False, num_workers=0)

    opt  = torch.optim.Adam(
               filter(lambda p: p.requires_grad, model.parameters()),
               lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min', factor=0.5, patience=4, min_lr=1e-7)

    hist = {'loss': [], 'iou': [], 'val_loss': [], 'val_iou': []}
    best_iou, best_state, no_improve = -1.0, None, 0

    epoch_bar = tqdm(range(1, epochs + 1), desc="Fine-tuning",
                     unit="epoch", disable=not verbose)
    for ep in epoch_bar:
        model.train()
        tl_acc, ti_acc = [], []
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred   = model(xb)
            loss   = bce_dice_loss(pred, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tl_acc.append(loss.item()); ti_acc.append(binary_iou(pred, yb).item())

        model.eval()
        vl_acc, vi_acc = [], []
        with torch.no_grad():
            for xb, yb in vl_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred   = model(xb)
                vl_acc.append(bce_dice_loss(pred, yb).item())
                vi_acc.append(binary_iou(pred, yb).item())

        tl = float(np.mean(tl_acc)); ti = float(np.mean(ti_acc))
        vl = float(np.mean(vl_acc)); vi = float(np.mean(vi_acc))
        hist['loss'].append(tl);     hist['iou'].append(ti)
        hist['val_loss'].append(vl); hist['val_iou'].append(vi)
        sched.step(vl)

        epoch_bar.set_postfix(iou=f"{ti:.4f}", val_iou=f"{vi:.4f}",
                              lr=f"{opt.param_groups[0]['lr']:.1e}")

        if vi > best_iou:
            best_iou   = vi
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            if save_path:
                torch.save(best_state, save_path)
        else:
            no_improve += 1

        if verbose:
            tqdm.write(f"  Ep {ep:3d}/{epochs}  loss {tl:.4f}  iou {ti:.4f}  "
                       f"val_loss {vl:.4f}  val_iou {vi:.4f}  "
                       f"lr {opt.param_groups[0]['lr']:.1e}")

        if no_improve >= patience:
            if verbose:
                tqdm.write(f"  Early stop at epoch {ep}  (best val_iou={best_iou:.4f})")
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return hist, best_iou


# ─────────────────────────────────────────────────────────────────── #
# Transfer-learning fine-tuning
# ─────────────────────────────────────────────────────────────────── #

def transfer_learn(base_model_path: Path  = SYNTHETIC_MODEL,
                   tl_model_path: Path    = TL_MODEL,
                   patch_size: int        = PATCH_SIZE,
                   max_patches: int       = None,
                   val_split: float       = 0.15,
                   phase1_epochs: int     = PHASE1_EPOCHS,
                   phase2_epochs: int     = PHASE2_EPOCHS,
                   verbose: int           = 1) -> dict:
    """
    Two-phase transfer learning on real labelled patches.

    Returns dict with keys 'phase1' and 'phase2', each containing a
    history dict.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "transfer_learning").mkdir(parents=True, exist_ok=True)
    device = get_device()

    if not REAL_STARS_PATH.exists():
        raise FileNotFoundError(
            f"Real data not found: {REAL_STARS_PATH}\n"
            "Place real_stars.npy and real_stars_labels.npy in star-dataset/.")

    if verbose:
        print("Loading real labelled patches …")
    X_real, Y_real = load_real_patches(
        str(REAL_STARS_PATH), str(REAL_LABELS_PATH),
        patch_size=patch_size, stride=patch_size,
        max_patches=max_patches, seed=0)

    if verbose:
        print(f"  {len(X_real)} patches  {X_real.shape}")

    n_val   = max(1, int(len(X_real) * val_split))
    n_train = len(X_real) - n_val
    X_tr, Y_tr = X_real[:n_train], Y_real[:n_train]
    X_vl, Y_vl = X_real[n_train:], Y_real[n_train:]

    if verbose:
        print(f"Loading base model: {base_model_path}")
    model = UNetStarFinder(base_filters=64, dropout=0.2).to(device)
    model.load_state_dict(torch.load(base_model_path, map_location=device))
    model = freeze_encoder(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = model.param_count()
    if verbose:
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # ── Phase 1: decoder only ─────────────────────────────────────── #
    if verbose:
        print("\n[Phase 1] Fine-tuning decoder …")
    hist1, best1 = _train_loop(
        model, X_tr, Y_tr, X_vl, Y_vl,
        lr=1e-4, epochs=phase1_epochs, batch_size=BATCH_SIZE,
        device=device, patience=8, save_path=tl_model_path, verbose=verbose)
    if verbose:
        print(f"Phase 1 best val IoU: {best1:.4f}")

    # ── Phase 2: full model fine-tuning ──────────────────────────── #
    model = unfreeze_all(model)
    if verbose:
        print("\n[Phase 2] Full model fine-tuning (LR=1e-5) …")
    hist2, best2 = _train_loop(
        model, X_tr, Y_tr, X_vl, Y_vl,
        lr=1e-5, epochs=phase2_epochs, batch_size=BATCH_SIZE,
        device=device, patience=8, save_path=tl_model_path, verbose=verbose)

    torch.save(model.state_dict(), tl_model_path)
    if verbose:
        print(f"\nTransfer-learned model saved → {tl_model_path}")
        print(f"Phase 2 best val IoU: {best2:.4f}")

    histories = {"phase1": hist1, "phase2": hist2}
    _plot_tl_history(histories, FIGURES_DIR)
    _save_tl_csv(histories, RESULTS_DIR / "transfer_learning" / "tl_history.csv")
    return histories


# ─────────────────────────────────────────────────────────────────── #
# Train from scratch on real data
# ─────────────────────────────────────────────────────────────────── #

def train_scratch_real(scratch_model_path: Path = SCRATCH_MODEL,
                       patch_size: int = PATCH_SIZE,
                       max_patches: int = None,
                       val_split: float = 0.15,
                       epochs: int = 60,
                       verbose: int = 1) -> dict:
    """Train a brand-new U-Net from scratch using only the real labelled data."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    if not REAL_STARS_PATH.exists():
        raise FileNotFoundError(f"Real data not found: {REAL_STARS_PATH}")

    if verbose:
        print("Loading real labelled patches …")
    X_real, Y_real = load_real_patches(
        str(REAL_STARS_PATH), str(REAL_LABELS_PATH),
        patch_size=patch_size, stride=patch_size,
        max_patches=max_patches, seed=0)

    n_val   = max(1, int(len(X_real) * val_split))
    n_train = len(X_real) - n_val
    X_tr, Y_tr = X_real[:n_train], Y_real[:n_train]
    X_vl, Y_vl = X_real[n_train:], Y_real[n_train:]

    model = UNetStarFinder(base_filters=64, dropout=0.2).to(device)

    if verbose:
        print("\n[Scratch] Training on real data only …")
    hist, best = _train_loop(
        model, X_tr, Y_tr, X_vl, Y_vl,
        lr=1e-4, epochs=epochs, batch_size=BATCH_SIZE,
        device=device, patience=12, save_path=scratch_model_path, verbose=verbose)

    torch.save(model.state_dict(), scratch_model_path)
    if verbose:
        print(f"Scratch model saved → {scratch_model_path}")
        print(f"Best val IoU: {best:.4f}")

    return hist


# ─────────────────────────────────────────────────────────────────── #
# Plotting
# ─────────────────────────────────────────────────────────────────── #

def _save_tl_csv(histories: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "epoch", "loss", "iou", "val_loss", "val_iou"])
        for phase, hist in histories.items():
            for ep, (tl, ti, vl, vi) in enumerate(zip(
                    hist["loss"], hist["iou"],
                    hist["val_loss"], hist["val_iou"]), 1):
                writer.writerow([phase, ep,
                                 round(tl, 6), round(ti, 6),
                                 round(vl, 6), round(vi, 6)])
    print(f"  TL metrics → {path}")


def _plot_tl_history(histories: dict, figures_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax_row, (phase_name, hist) in zip(axes, histories.items()):
        ax_row[0].plot(hist["loss"],     label="train")
        ax_row[0].plot(hist["val_loss"], label="val")
        ax_row[0].set_title(f"{phase_name} — Loss")
        ax_row[0].set_xlabel("Epoch"); ax_row[0].legend()

        ax_row[1].plot(hist["iou"],     label="train")
        ax_row[1].plot(hist["val_iou"], label="val")
        ax_row[1].set_title(f"{phase_name} — IoU")
        ax_row[1].set_xlabel("Epoch"); ax_row[1].legend()

    plt.suptitle("Transfer Learning (MPS) — training history", fontsize=12)
    plt.tight_layout()
    out = figures_dir / "transfer_learning_history.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"TL curves saved → {out}")


if __name__ == "__main__":
    transfer_learn()
