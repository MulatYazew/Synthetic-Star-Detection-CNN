"""
Loss functions for stellar segmentation.

Three formulations are provided and compared in the notebook:

1. Binary Cross-Entropy (BCE)
   L_BCE = -[y log(p) + (1-y) log(1-p)]
   Standard pixel-wise likelihood loss. Sensitive to class imbalance:
   background pixels dominate (~97% of pixels for sparse stars).

2. Dice Loss
   L_Dice = 1 - (2 |Y ∩ P| + ε) / (|Y| + |P| + ε)
   Directly optimises the F1/Dice overlap coefficient. Robust to class
   imbalance because it normalises by the size of the detected region.

3. BCE + Dice Hybrid (α=0.5)
   L_hybrid = α·L_BCE + (1-α)·L_Dice
   Combines pixel-wise calibration from BCE with overlap-maximisation
   from Dice. Empirically the best-performing loss for sparse star
   segmentation: BCE ensures well-calibrated probabilities while Dice
   guards against the background-dominance failure mode.
"""

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────── #
# Dice coefficient (similarity metric, ∈ [0,1])
# ─────────────────────────────────────────────────────────────────── #

def dice_coefficient(y_true: torch.Tensor,
                     y_pred: torch.Tensor,
                     smooth: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice coefficient averaged over the batch.

    Parameters
    ----------
    y_true  : ground-truth binary mask  (B, H, W, 1) or (B, 1, H, W)
    y_pred  : predicted probability map  same shape as y_true
    smooth  : Laplace smoothing to avoid 0/0

    Returns
    -------
    Scalar tensor ∈ [0, 1]; higher is better.
    """
    p = y_pred.reshape(y_pred.size(0), -1)
    t = y_true.reshape(y_true.size(0), -1)
    intersection = (p * t).sum(dim=1)
    denom        = p.sum(dim=1) + t.sum(dim=1)
    return ((2.0 * intersection + smooth) / (denom + smooth)).mean()


# ─────────────────────────────────────────────────────────────────── #
# Loss functions
# ─────────────────────────────────────────────────────────────────── #

def bce_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Pixel-wise binary cross-entropy, averaged over all elements."""
    return F.binary_cross_entropy(y_pred, y_true)


def dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """1 − soft Dice coefficient."""
    return 1.0 - dice_coefficient(y_true, y_pred)


def bce_dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor,
                  alpha: float = 0.5) -> torch.Tensor:
    """
    Convex combination of BCE and Dice.
    α=0.5 weights both contributions equally.
    """
    return alpha * bce_loss(y_true, y_pred) + (1.0 - alpha) * dice_loss(y_true, y_pred)


# ─────────────────────────────────────────────────────────────────── #
# Registry helpers  (used by model.py to select the loss by name)
# ─────────────────────────────────────────────────────────────────── #

LOSS_REGISTRY = {
    "bce":      bce_loss,
    "dice":     dice_loss,
    "bce_dice": bce_dice_loss,
}


def get_loss(name: str):
    """Return loss function by string key ('bce', 'dice', or 'bce_dice')."""
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'.  Choose from: {list(LOSS_REGISTRY)}"
        )
    return LOSS_REGISTRY[name]
