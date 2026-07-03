"""
Evaluation utilities for the stellar-segmentation U-Net.

Pixel-level metrics
-------------------
  IoU (Jaccard index)   TP / (TP + FP + FN)
  Pixel accuracy        (TP + TN) / N
  Precision             TP / (TP + FP)
  Recall                TP / (TP + FN)
  F1 / Dice coefficient 2·Precision·Recall / (Precision + Recall)

Object-level metrics
--------------------
  Star count  — connected components in predicted mask (min_area ≥ 2)
  Centroid, area, mean intensity from skimage.measure.regionprops
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.measure import label as skimage_label, regionprops
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────── #
# Pixel-level metrics
# ─────────────────────────────────────────────────────────────────── #

def pixel_iou(y_true: np.ndarray,
              y_pred: np.ndarray,
              threshold: float = 0.5) -> float:
    """Binary IoU = TP / (TP + FP + FN).  Inputs can be (H,W) or (H,W,1)."""
    t = (y_true.squeeze() > 0.5).astype(bool)
    p = (y_pred.squeeze() > threshold).astype(bool)
    inter = np.logical_and(t, p).sum()
    union = np.logical_or(t, p).sum()
    return float(inter) / float(union + 1e-7)


def pixel_metrics(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  threshold: float = 0.5) -> dict:
    """
    Return a dict with all pixel-level metrics at a single threshold.

    Keys: iou, pixel_accuracy, precision, recall, f1, dice
    """
    t = (y_true.squeeze() > 0.5)
    p = (y_pred.squeeze() > threshold)

    tp = float(np.logical_and(t,  p).sum())
    fp = float(np.logical_and(~t, p).sum())
    fn = float(np.logical_and(t, ~p).sum())
    tn = float(np.logical_and(~t, ~p).sum())
    n  = tp + fp + fn + tn

    iou       = tp / (tp + fp + fn + 1e-7)
    acc       = (tp + tn) / (n + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall    = tp / (tp + fn + 1e-7)
    f1        = 2.0 * precision * recall / (precision + recall + 1e-7)

    return {
        "iou":            iou,
        "pixel_accuracy": acc,
        "precision":      precision,
        "recall":         recall,
        "f1":             f1,
        "dice":           f1,       # Dice = F1 for binary case
    }


def comprehensive_metrics(y_true_batch: np.ndarray,
                           y_pred_batch: np.ndarray,
                           threshold: float = 0.5) -> dict:
    """
    Aggregate pixel metrics over a batch, plus star-count statistics.

    Parameters
    ----------
    y_true_batch, y_pred_batch : (N, H, W, 1) arrays

    Returns
    -------
    dict with keys mean_iou, mean_pixel_accuracy, mean_precision,
    mean_recall, mean_f1, mean_dice, count_mae, count_exact_rate
    """
    keys = ["iou", "pixel_accuracy", "precision", "recall", "f1", "dice"]
    accum = {k: [] for k in keys}

    n_true_list, n_pred_list = [], []
    for i in tqdm(range(len(y_true_batch)), desc="Computing metrics", leave=False):
        m = pixel_metrics(y_true_batch[i], y_pred_batch[i], threshold)
        for k in keys:
            accum[k].append(m[k])
        n_true_list.append(count_stars(y_true_batch[i], 0.5))
        n_pred_list.append(count_stars(y_pred_batch[i], threshold))

    n_true = np.array(n_true_list)
    n_pred = np.array(n_pred_list)

    result = {f"mean_{k}": float(np.mean(accum[k])) for k in keys}
    result["std_iou"]           = float(np.std(accum["iou"]))
    result["count_mae"]         = float(np.abs(n_true - n_pred).mean())
    result["count_exact_rate"]  = float((n_true == n_pred).mean())
    result["n_true"]            = n_true
    result["n_pred"]            = n_pred
    return result


def threshold_sweep(y_true_batch: np.ndarray,
                    y_pred_batch: np.ndarray,
                    thresholds: np.ndarray = None) -> dict:
    """
    Evaluate pixel metrics at every threshold in `thresholds`.

    Returns dict mapping metric name → list of values (one per threshold).
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    keys = ["iou", "precision", "recall", "f1"]
    curves = {k: [] for k in keys}

    for thr in tqdm(thresholds, desc="Threshold sweep", unit="thr", leave=False):
        batch_vals = {k: [] for k in keys}
        for i in range(len(y_true_batch)):
            m = pixel_metrics(y_true_batch[i], y_pred_batch[i], thr)
            for k in keys:
                batch_vals[k].append(m[k])
        for k in keys:
            curves[k].append(float(np.mean(batch_vals[k])))

    curves["threshold"] = list(thresholds)
    return curves


# ─────────────────────────────────────────────────────────────────── #
# Object-level (star count) metrics
# ─────────────────────────────────────────────────────────────────── #

def count_stars(mask: np.ndarray,
                threshold: float = 0.5,
                min_area: int = 2) -> int:
    """
    Count connected components in a binary mask.
    Single-pixel detections (area < min_area) are rejected to
    reduce noise in the star count.
    """
    binary  = (mask.squeeze() > threshold).astype(np.uint8)
    labeled = skimage_label(binary, connectivity=2)
    regions = regionprops(labeled)
    return sum(1 for r in regions if r.area >= min_area)


def get_star_regions(mask: np.ndarray,
                     image: np.ndarray = None,
                     threshold: float = 0.5,
                     min_area: int = 2) -> list:
    """
    Return a list of skimage RegionProperties for each detected star.

    Parameters
    ----------
    mask  : probability map or binary mask  (H, W) or (H, W, 1)
    image : intensity image for mean_intensity; uses mask if None

    Each region exposes: .centroid, .area, .bbox,
    .mean_intensity (if image given), .equivalent_diameter.
    """
    binary  = (mask.squeeze() > threshold).astype(np.uint8)
    labeled = skimage_label(binary, connectivity=2)
    intens  = image.squeeze() if image is not None else None
    regions = regionprops(labeled, intensity_image=intens)
    return [r for r in regions if r.area >= min_area]


def star_count_accuracy(y_true_batch: np.ndarray,
                        y_pred_batch: np.ndarray,
                        threshold: float = 0.5,
                        min_area: int = 2) -> dict:
    """Per-batch star-count mean absolute error and exact-match rate."""
    n_t = np.array([count_stars(y_true_batch[i], 0.5,       min_area)
                    for i in range(len(y_true_batch))])
    n_p = np.array([count_stars(y_pred_batch[i], threshold, min_area)
                    for i in range(len(y_pred_batch))])
    return {
        "count_mae":        float(np.abs(n_t - n_p).mean()),
        "count_exact_rate": float((n_t == n_p).mean()),
        "n_true":           n_t,
        "n_pred":           n_p,
    }


# ─────────────────────────────────────────────────────────────────── #
# Batch evaluation
# ─────────────────────────────────────────────────────────────────── #

def evaluate_model(model,
                   X: np.ndarray,
                   Y: np.ndarray,
                   threshold: float = 0.5,
                   batch_size: int = 64,
                   min_area: int = 2) -> tuple:
    """
    Run full evaluation: pixel + object-level metrics.
    Returns (metrics_dict, Y_pred).
    """
    import torch
    device = next(model.parameters()).device
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).permute(0, 3, 1, 2).to(device)
            preds.append(model(xb).permute(0, 2, 3, 1).cpu().numpy())
    Y_pred = np.concatenate(preds, axis=0)

    metrics = comprehensive_metrics(Y, Y_pred, threshold)
    return metrics, Y_pred


# ─────────────────────────────────────────────────────────────────── #
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────── #

def plot_predictions(X: np.ndarray,
                     Y_true: np.ndarray,
                     Y_pred: np.ndarray,
                     n_examples: int = 6,
                     threshold: float = 0.5,
                     title_prefix: str = "",
                     save_path: str = None) -> plt.Figure:
    """
    Grid: input | true mask | predicted mask | overlay (n_examples rows).
    Lime contour = ground truth; red contour = prediction.
    """
    n_examples = min(n_examples, len(X))
    fig, axes  = plt.subplots(n_examples, 4,
                               figsize=(14, 3.2 * n_examples))
    if n_examples == 1:
        axes = axes[np.newaxis]

    for col, title in enumerate(
            ["Input image", "True mask", "Predicted mask", "Overlay"]):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    for i in range(n_examples):
        img  = X[i].squeeze()
        true = Y_true[i].squeeze()
        pred = (Y_pred[i].squeeze() > threshold).astype(float)

        axes[i, 0].imshow(img,  cmap="gray", origin="upper")
        axes[i, 1].imshow(true, cmap="gray", origin="upper")
        axes[i, 2].imshow(pred, cmap="gray", origin="upper")

        axes[i, 3].imshow(img, cmap="gray", origin="upper")
        axes[i, 3].contour(true, levels=[0.5], colors="lime",  linewidths=1)
        axes[i, 3].contour(pred, levels=[0.5], colors="red",   linewidths=1)

        iou_v = pixel_iou(true, pred, threshold)
        n_t   = count_stars(true)
        n_p   = count_stars(pred)
        axes[i, 0].set_ylabel(
            f"IoU={iou_v:.2f}\ntrue={n_t} pred={n_p}", fontsize=8)

        for ax in axes[i]:
            ax.axis("off")

    if title_prefix:
        fig.suptitle(f"{title_prefix} — lime=truth, red=prediction",
                     fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_large_field(image: np.ndarray,
                     pred_mask: np.ndarray,
                     true_mask: np.ndarray = None,
                     threshold: float = 0.5,
                     title: str = "Stellar field",
                     save_path: str = None) -> plt.Figure:
    """
    Display a large image with its predicted mask (and optionally true mask).
    Uses Astropy ZScale for contrast.
    """
    from astropy.visualization import ZScaleInterval
    img  = image.squeeze()
    pred = (pred_mask.squeeze() > threshold).astype(float)

    ncols = 3 if true_mask is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 7))

    iv = ZScaleInterval()
    vmin, vmax = iv.get_limits(img)

    axes[0].imshow(img,  cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("Input image")

    axes[1].imshow(pred, cmap="gray", origin="lower")
    n_det = count_stars(pred, threshold)
    axes[1].set_title(f"Predicted mask  ({n_det} stars)")

    if true_mask is not None:
        axes[2].imshow(true_mask.squeeze(), cmap="gray", origin="lower")
        axes[2].set_title("True mask")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def print_metrics(metrics: dict, label: str = "") -> None:
    hdr = f"── {label} ──" if label else "── Metrics ──"
    print(f"\n{hdr}")
    scalar_keys = [k for k, v in metrics.items()
                   if isinstance(v, (float, int)) and not k.startswith("n_")]
    for k in scalar_keys:
        print(f"  {k:<28s}: {metrics[k]:.4f}")
    print()


# ─────────────────────────────────────────────────────────────────── #
# Experiment tracking — CSV export
# ─────────────────────────────────────────────────────────────────── #

def save_metrics_to_csv(metrics: dict,
                         path: "str | Path",
                         label: str = "",
                         append: bool = False) -> None:
    """
    Save scalar metrics from a dict to a CSV file.

    append=True appends a new row to an existing file (for sweep logs).
    append=False (default) overwrites / creates a fresh file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scalar = {k: v for k, v in metrics.items()
              if isinstance(v, (float, int)) and not k.startswith("n_")}
    mode         = "a" if append and path.exists() else "w"
    write_header = not (append and path.exists())
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label"] + list(scalar.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow({"label": label,
                         **{k: round(v, 6) for k, v in scalar.items()}})


def save_threshold_sweep_to_csv(curves: dict, path: "str | Path") -> None:
    """Save threshold sweep results (from threshold_sweep()) to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = [k for k in curves if k != "threshold"]
    thresholds  = curves["threshold"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold"] + metric_keys)
        writer.writeheader()
        for i, thr in enumerate(thresholds):
            row = {"threshold": round(float(thr), 4)}
            row.update({k: round(float(curves[k][i]), 6) for k in metric_keys})
            writer.writerow(row)
