"""
Transfer-learning and from-scratch training pipelines for real astronomical data.

For each pipeline (TL and scratch) trains with three loss functions
(BCE, Dice, BCE+Dice) and selects the best model by validation IoU.

Strategy (Transfer Learning)
----------------------------
Phase 1: Freeze encoder (enc1–enc4 + bottleneck).
         Train only the decoder on real labelled patches.
         LR = 1e-4, up to 30 epochs, early-stop on val IoU.

Phase 2: Unfreeze all; fine-tune end-to-end at LR = 1e-5
         to prevent catastrophic forgetting of synthetic features.

Usage (from project root):
    python codes/transfer_learning.py
"""

import csv
import os
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    from codes.dataset import load_real_patches
    from codes.model import (UNetStarFinder, freeze_encoder, unfreeze_all,
                              binary_iou)
    from codes.losses import get_loss
    from codes.config import (
        MODELS_DIR, TL_RESULTS_DIR, REAL_RESULTS_DIR,
        SYNTHETIC_MODEL_PATH, TL_MODEL_PATH, SCRATCH_MODEL_PATH,
        REAL_STARS_PATH, REAL_LABELS_PATH,
        LOSS_NAMES, LOSS_DISPLAY,
    )
except ImportError:
    from dataset import load_real_patches
    from model import (UNetStarFinder, freeze_encoder, unfreeze_all,
                       binary_iou)
    from losses import get_loss
    from config import (
        MODELS_DIR, TL_RESULTS_DIR, REAL_RESULTS_DIR,
        SYNTHETIC_MODEL_PATH, TL_MODEL_PATH, SCRATCH_MODEL_PATH,
        REAL_STARS_PATH, REAL_LABELS_PATH,
        LOSS_NAMES, LOSS_DISPLAY,
    )


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
                lr, epochs, batch_size, device, loss_fn,
                patience=8, save_path=None, label="", verbose=1):
    """Train/val loop with EarlyStopping + ReduceLROnPlateau. Returns (history, best_val_iou)."""
    xt = torch.from_numpy(X_tr).permute(0, 3, 1, 2)
    yt = torch.from_numpy(Y_tr).permute(0, 3, 1, 2)
    xv = torch.from_numpy(X_vl).permute(0, 3, 1, 2)
    yv = torch.from_numpy(Y_vl).permute(0, 3, 1, 2)
    tr_loader = DataLoader(TensorDataset(xt, yt), batch_size=batch_size,
                           shuffle=True, num_workers=0)
    vl_loader = DataLoader(TensorDataset(xv, yv), batch_size=batch_size,
                           shuffle=False, num_workers=0)

    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=4, min_lr=1e-7)

    hist = {"loss": [], "iou": [], "val_loss": [], "val_iou": []}
    best_iou, best_state, no_improve = -1.0, None, 0

    epoch_bar = tqdm(range(1, epochs + 1), desc=label or "Training",
                     unit="epoch", disable=not verbose)
    for ep in epoch_bar:
        model.train()
        tl_acc, ti_acc = [], []
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred   = model(xb)
            loss   = loss_fn(yb, pred)   # (y_true, y_pred) convention
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tl_acc.append(loss.item())
            ti_acc.append(binary_iou(pred, yb).item())

        model.eval()
        vl_acc, vi_acc = [], []
        with torch.no_grad():
            for xb, yb in vl_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred   = model(xb)
                vl_acc.append(loss_fn(yb, pred).item())
                vi_acc.append(binary_iou(pred, yb).item())

        tl = float(np.mean(tl_acc)); ti = float(np.mean(ti_acc))
        vl = float(np.mean(vl_acc)); vi = float(np.mean(vi_acc))
        hist["loss"].append(tl);     hist["iou"].append(ti)
        hist["val_loss"].append(vl); hist["val_iou"].append(vi)
        sched.step(vl)

        epoch_bar.set_postfix(iou=f"{ti:.4f}", val_iou=f"{vi:.4f}",
                              lr=f"{opt.param_groups[0]['lr']:.1e}")

        if vi > best_iou:
            best_iou   = vi
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
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
# Transfer learning — single loss function
# ─────────────────────────────────────────────────────────────────── #

def _tl_one_loss(loss_name, X_tr, Y_tr, X_vl, Y_vl, device,
                 base_model_path,
                 phase1_epochs=PHASE1_EPOCHS,
                 phase2_epochs=PHASE2_EPOCHS,
                 checkpoint_path=None, verbose=1):
    """Two-phase TL for one loss function. Returns (histories_dict, best_val_iou)."""
    loss_fn = get_loss(loss_name)
    label   = LOSS_DISPLAY.get(loss_name, loss_name)

    model = UNetStarFinder(base_filters=64, dropout=0.2).to(device)
    model.load_state_dict(torch.load(base_model_path, map_location=device,
                                     weights_only=True))
    model = freeze_encoder(model)

    if verbose:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = model.param_count()
        print(f"\n[{label}] Phase 1 — decoder fine-tuning  "
              f"({trainable:,}/{total:,} trainable params)")

    hist1, best1 = _train_loop(
        model, X_tr, Y_tr, X_vl, Y_vl,
        lr=1e-4, epochs=phase1_epochs, batch_size=BATCH_SIZE,
        device=device, loss_fn=loss_fn,
        patience=8, save_path=checkpoint_path,
        label=f"[{label}] Ph1", verbose=verbose)

    model = unfreeze_all(model)
    if verbose:
        print(f"\n[{label}] Phase 2 — full fine-tuning (LR=1e-5)")

    hist2, best2 = _train_loop(
        model, X_tr, Y_tr, X_vl, Y_vl,
        lr=1e-5, epochs=phase2_epochs, batch_size=BATCH_SIZE,
        device=device, loss_fn=loss_fn,
        patience=8, save_path=checkpoint_path,
        label=f"[{label}] Ph2", verbose=verbose)

    best_iou = max(best1, best2)
    if verbose:
        print(f"[{label}] Phase 1 best: {best1:.4f} | Phase 2 best: {best2:.4f} "
              f"| Overall best: {best_iou:.4f}")

    return {"phase1": hist1, "phase2": hist2}, best_iou


# ─────────────────────────────────────────────────────────────────── #
# Transfer-learning orchestrator
# ─────────────────────────────────────────────────────────────────── #

def transfer_learn(base_model_path: Path = None,
                   tl_model_path: Path   = TL_MODEL_PATH,
                   patch_size: int       = PATCH_SIZE,
                   max_patches: int      = None,
                   val_split: float      = 0.15,
                   phase1_epochs: int    = PHASE1_EPOCHS,
                   phase2_epochs: int    = PHASE2_EPOCHS,
                   loss_names: list      = None,
                   verbose: int          = 1) -> dict:
    """
    Two-phase TL on real patches for each loss function in loss_names.
    Best model by val IoU is saved to tl_model_path.
    All outputs go to results/transfer_learning/.

    Returns dict mapping loss_name → (histories, best_val_iou).
    """
    if base_model_path is None:
        if SYNTHETIC_MODEL_PATH.exists():
            base_model_path = SYNTHETIC_MODEL_PATH
        else:
            # Fall back to legacy name for backward compatibility
            legacy = MODELS_DIR / "star_finder_synthetic.pt"
            if legacy.exists():
                base_model_path = legacy
            else:
                raise FileNotFoundError(
                    f"Base synthetic model not found.\n"
                    f"Expected: {SYNTHETIC_MODEL_PATH}\n"
                    f"Run python codes/train.py first.")

    if loss_names is None:
        loss_names = LOSS_NAMES

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    if not REAL_STARS_PATH.exists():
        raise FileNotFoundError(
            f"Real data not found: {REAL_STARS_PATH}\n"
            "Place real_stars.npy and real_stars_labels.npy in star-dataset/.")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Transfer Learning Pipeline")
        print(f"  Base model  : {base_model_path}")
        print(f"  Loss fns    : {loss_names}")
        print(f"  Output dir  : {TL_RESULTS_DIR}")
        print(f"{'='*60}")
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

    results: dict[str, tuple] = {}
    for loss_name in loss_names:
        checkpoint = TL_RESULTS_DIR / f"{loss_name}_best.pt"
        histories, best_iou = _tl_one_loss(
            loss_name       = loss_name,
            X_tr=X_tr, Y_tr=Y_tr, X_vl=X_vl, Y_vl=Y_vl,
            device          = device,
            base_model_path = base_model_path,
            phase1_epochs   = phase1_epochs,
            phase2_epochs   = phase2_epochs,
            checkpoint_path = checkpoint,
            verbose         = verbose,
        )
        results[loss_name] = (histories, best_iou)
        _save_tl_csv(histories,
                     TL_RESULTS_DIR / f"{loss_name}_tl_history.csv",
                     loss_name=loss_name)

    best_loss = max(results, key=lambda k: results[k][1])
    src = TL_RESULTS_DIR / f"{best_loss}_best.pt"
    tl_model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, tl_model_path)

    if verbose:
        print(f"\nTransfer-learning comparison:")
        for ln, (_, biou) in results.items():
            marker = "  ← BEST" if ln == best_loss else ""
            print(f"  {LOSS_DISPLAY.get(ln, ln):<22} val IoU = {biou:.4f}{marker}")
        print(f"\nBest TL model saved → {tl_model_path}")

    plot_tl_comparison(results, TL_RESULTS_DIR)
    _save_comparison_report(
        results, TL_RESULTS_DIR, best_loss, tl_model_path,
        title="Transfer Learning Pipeline")
    return results


# ─────────────────────────────────────────────────────────────────── #
# Scratch — single loss function
# ─────────────────────────────────────────────────────────────────── #

def _scratch_one_loss(loss_name, X_tr, Y_tr, X_vl, Y_vl,
                      device, epochs=60,
                      checkpoint_path=None, verbose=1):
    """Train one fresh U-Net for one loss function. Returns (history, best_val_iou)."""
    loss_fn = get_loss(loss_name)
    label   = LOSS_DISPLAY.get(loss_name, loss_name)

    model = UNetStarFinder(base_filters=64, dropout=0.2).to(device)
    if verbose:
        print(f"\n[{label}] Training from scratch …")

    hist, best_iou = _train_loop(
        model, X_tr, Y_tr, X_vl, Y_vl,
        lr=1e-4, epochs=epochs, batch_size=BATCH_SIZE,
        device=device, loss_fn=loss_fn,
        patience=12, save_path=checkpoint_path,
        label=f"[{label}]", verbose=verbose)

    if verbose:
        print(f"[{label}] Best val IoU: {best_iou:.4f}")
    return hist, best_iou


# ─────────────────────────────────────────────────────────────────── #
# From-scratch orchestrator
# ─────────────────────────────────────────────────────────────────── #

def train_scratch_real(scratch_model_path: Path = SCRATCH_MODEL_PATH,
                       patch_size: int          = PATCH_SIZE,
                       max_patches: int         = None,
                       val_split: float         = 0.15,
                       epochs: int              = 60,
                       loss_names: list         = None,
                       verbose: int             = 1) -> dict:
    """
    Train U-Net from scratch on real data for each loss function.
    Best model by val IoU is saved to scratch_model_path.
    All outputs go to results/real/.

    Returns dict mapping loss_name → (history, best_val_iou).
    """
    if loss_names is None:
        loss_names = LOSS_NAMES

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    if not REAL_STARS_PATH.exists():
        raise FileNotFoundError(f"Real data not found: {REAL_STARS_PATH}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Scratch Training Pipeline (real data)")
        print(f"  Loss fns   : {loss_names}")
        print(f"  Output dir : {REAL_RESULTS_DIR}")
        print(f"{'='*60}")
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

    results: dict[str, tuple] = {}
    for loss_name in loss_names:
        checkpoint = REAL_RESULTS_DIR / f"{loss_name}_best.pt"
        hist, best_iou = _scratch_one_loss(
            loss_name       = loss_name,
            X_tr=X_tr, Y_tr=Y_tr, X_vl=X_vl, Y_vl=Y_vl,
            device          = device,
            epochs          = epochs,
            checkpoint_path = checkpoint,
            verbose         = verbose,
        )
        results[loss_name] = (hist, best_iou)
        _save_scratch_csv(hist,
                          REAL_RESULTS_DIR / f"{loss_name}_scratch_history.csv",
                          loss_name=loss_name)

    best_loss = max(results, key=lambda k: results[k][1])
    src = REAL_RESULTS_DIR / f"{best_loss}_best.pt"
    scratch_model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, scratch_model_path)

    if verbose:
        print(f"\nScratch training comparison:")
        for ln, (_, biou) in results.items():
            marker = "  ← BEST" if ln == best_loss else ""
            print(f"  {LOSS_DISPLAY.get(ln, ln):<22} val IoU = {biou:.4f}{marker}")
        print(f"\nBest scratch model saved → {scratch_model_path}")

    plot_scratch_comparison(results, REAL_RESULTS_DIR)
    _save_comparison_report(
        results, REAL_RESULTS_DIR, best_loss, scratch_model_path,
        title="Real-Data Scratch Pipeline")
    return results


# ─────────────────────────────────────────────────────────────────── #
# Plotting helpers
# ─────────────────────────────────────────────────────────────────── #

def plot_tl_comparison(results: dict, save_dir: Path) -> None:
    """Comparison plot of val IoU and val loss across all TL runs."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for loss_name, (histories, _) in results.items():
        label = LOSS_DISPLAY.get(loss_name, loss_name)
        val_iou  = (histories["phase1"]["val_iou"]
                    + histories["phase2"]["val_iou"])
        val_loss = (histories["phase1"]["val_loss"]
                    + histories["phase2"]["val_loss"])
        n_ph1 = len(histories["phase1"]["val_iou"])

        axes[0].plot(val_iou,  label=label)
        axes[1].plot(val_loss, label=label)
        # Dashed vertical line marks the phase 1 → phase 2 boundary
        if n_ph1 < len(val_iou):
            for ax in axes:
                ax.axvline(x=n_ph1, color="gray", linestyle="--", linewidth=0.8,
                           alpha=0.6)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation IoU")
    axes[0].set_title("TL Validation IoU (dashed = phase boundary)")
    axes[0].legend()

    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation Loss")
    axes[1].set_title("TL Validation Loss")
    axes[1].legend()

    plt.suptitle("Transfer Learning — Loss Function Comparison", fontsize=12)
    plt.tight_layout()
    out = save_dir / "loss_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  TL comparison plot → {out}")


def plot_scratch_comparison(results: dict, save_dir: Path) -> None:
    """Comparison plot of val IoU and val loss across all scratch runs."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for loss_name, (hist, _) in results.items():
        label = LOSS_DISPLAY.get(loss_name, loss_name)
        axes[0].plot(hist["val_iou"],  label=label)
        axes[1].plot(hist["val_loss"], label=label)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation IoU")
    axes[0].set_title("Scratch Validation IoU — loss function comparison")
    axes[0].legend()

    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation Loss")
    axes[1].set_title("Scratch Validation Loss — loss function comparison")
    axes[1].legend()

    plt.suptitle("Real-Data Scratch Training — Loss Function Comparison", fontsize=12)
    plt.tight_layout()
    out = save_dir / "loss_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Scratch comparison plot → {out}")


# ─────────────────────────────────────────────────────────────────── #
# CSV / text report helpers
# ─────────────────────────────────────────────────────────────────── #

def _save_tl_csv(histories: dict, path: Path, loss_name: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["loss_fn", "phase", "epoch", "loss", "iou",
                         "val_loss", "val_iou"])
        for phase, hist in histories.items():
            for ep, (tl, ti, vl, vi) in enumerate(zip(
                    hist["loss"], hist["iou"],
                    hist["val_loss"], hist["val_iou"]), 1):
                writer.writerow([loss_name, phase, ep,
                                 round(tl, 6), round(ti, 6),
                                 round(vl, 6), round(vi, 6)])
    print(f"  TL history CSV → {path}")


def _save_scratch_csv(hist: dict, path: Path, loss_name: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["loss_fn", "epoch", "loss", "iou", "val_loss", "val_iou"])
        for ep, (tl, ti, vl, vi) in enumerate(zip(
                hist["loss"], hist["iou"],
                hist["val_loss"], hist["val_iou"]), 1):
            writer.writerow([loss_name, ep,
                             round(tl, 6), round(ti, 6),
                             round(vl, 6), round(vi, 6)])
    print(f"  Scratch history CSV → {path}")


def _save_comparison_report(results, save_dir, best_loss, model_path, title=""):
    """Write loss_comparison.csv and best_model_report.txt."""
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / "loss_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["loss_fn", "display_name", "best_val_iou"])
        for ln, (_, biou) in results.items():
            writer.writerow([ln, LOSS_DISPLAY.get(ln, ln), round(biou, 6)])
    print(f"  Comparison CSV → {csv_path}")

    txt_path = save_dir / "best_model_report.txt"
    lines = [
        f"=== {title} — Loss Function Comparison ===" if title
        else "=== Loss Function Comparison ===",
        "",
        f"{'Loss Function':<22}{'Best Val IoU':>14}",
        "─" * 36,
    ]
    for ln, (_, biou) in results.items():
        marker = "  ← SELECTED" if ln == best_loss else ""
        lines.append(f"{LOSS_DISPLAY.get(ln, ln):<22}{biou:>14.4f}{marker}")
    best_iou = results[best_loss][1]
    lines += [
        "─" * 36,
        "",
        f"Best loss function : {LOSS_DISPLAY.get(best_loss, best_loss)}",
        f"Best val IoU       : {best_iou:.4f}",
        f"Model saved to     : {model_path}",
    ]
    txt_path.write_text("\n".join(lines))
    print(f"  Report → {txt_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    transfer_learn()
    train_scratch_real()
