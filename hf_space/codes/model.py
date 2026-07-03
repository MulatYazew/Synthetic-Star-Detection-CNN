"""
U-Net for pixel-wise stellar segmentation — MPS-ready.

Architecture
  Encoder    :  1→64→128→256→512   (ConvBlock + MaxPool)
  Bottleneck :  512→1024
  Decoder    :  1024→512→256→128→64 (ConvTranspose + skip-cat + ConvBlock)
  Output     :  64→1  sigmoid

Design notes
  - Fixed channel dimensions → MPS can fuse ops
  - L2 regularisation via optimizer weight_decay (AdamW-style)
  - nn.Dropout2d for spatial dropout
  - BatchNorm placed before ReLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────── #
# Building blocks
# ─────────────────────────────────────────────────────────────────── #

class ConvBlock(nn.Module):
    """Two (Conv2D → BN → ReLU) units followed by spatial dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.drop  = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return self.drop(x)


# ─────────────────────────────────────────────────────────────────── #
# Losses
# ─────────────────────────────────────────────────────────────────── #

def dice_loss(pred: torch.Tensor, target: torch.Tensor,
              smooth: float = 1e-6) -> torch.Tensor:
    """1 − soft Dice coefficient, averaged over the batch."""
    p = pred.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    inter = (p * t).sum(dim=1)
    denom = p.sum(dim=1) + t.sum(dim=1)
    return 1.0 - ((2.0 * inter + smooth) / (denom + smooth)).mean()


def bce_dice_loss(pred: torch.Tensor, target: torch.Tensor,
                  alpha: float = 0.5) -> torch.Tensor:
    """Hybrid BCE + Dice loss (α = 0.5)."""
    bce  = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return alpha * bce + (1.0 - alpha) * dice


def bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pixel-wise binary cross-entropy averaged over all elements."""
    return F.binary_cross_entropy(pred, target)


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor,
                     smooth: float = 1e-6) -> torch.Tensor:
    """Soft Dice coefficient ∈ [0, 1], averaged over the batch (higher = better)."""
    p = pred.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    inter = (p * t).sum(dim=1)
    denom = p.sum(dim=1) + t.sum(dim=1)
    return ((2.0 * inter + smooth) / (denom + smooth)).mean()


def binary_iou(pred: torch.Tensor, target: torch.Tensor,
               threshold: float = 0.5) -> torch.Tensor:
    """Binary IoU metric (both classes averaged)."""
    p = (pred  >= threshold).float()
    t = (target >= threshold).float()
    inter = (p * t).sum()
    union = (p + t - p * t).sum()
    return inter / (union + 1e-6)


# ─────────────────────────────────────────────────────────────────── #
# Full U-Net
# ─────────────────────────────────────────────────────────────────── #

class UNetStarFinder(nn.Module):
    """
    U-Net for stellar segmentation.

    Parameters
    ----------
    base_filters : int
        Number of filters at the shallowest encoder level.
        Doubled at each depth: 64 → 128 → 256 → 512 → bottleneck 1024.
        Use 32 for a lighter model (7.8 M params) or 64 for the full model
        (31 M params). 32 is recommended for 64×64 inputs on M4.
    dropout : float
        SpatialDropout2D rate after each conv block.
    """

    def __init__(self, base_filters: int = 64, dropout: float = 0.2):
        super().__init__()
        f = base_filters

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = ConvBlock(1,    f,    dropout)
        self.enc2 = ConvBlock(f,    f*2,  dropout)
        self.enc3 = ConvBlock(f*2,  f*4,  dropout)
        self.enc4 = ConvBlock(f*4,  f*8,  dropout)

        # Bottleneck
        self.bottleneck = ConvBlock(f*8, f*16, dropout)

        # Decoder — transposed convolutions for upsampling
        self.up4 = nn.ConvTranspose2d(f*16, f*8,  2, stride=2)
        self.up3 = nn.ConvTranspose2d(f*8,  f*4,  2, stride=2)
        self.up2 = nn.ConvTranspose2d(f*4,  f*2,  2, stride=2)
        self.up1 = nn.ConvTranspose2d(f*2,  f,    2, stride=2)

        # Decoder conv blocks; input channels = upsampled + skip = doubled
        self.dec4 = ConvBlock(f*16, f*8,  dropout)
        self.dec3 = ConvBlock(f*8,  f*4,  dropout)
        self.dec2 = ConvBlock(f*4,  f*2,  dropout)
        self.dec1 = ConvBlock(f*2,  f,    dropout)

        # 1×1 output projection
        self.output_conv = nn.Conv2d(f, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        # Bottleneck
        b = self.bottleneck(self.pool(s4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b),  s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))

        return torch.sigmoid(self.output_conv(d1))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────── #
# Factory helpers (mirrors model.py API)
# ─────────────────────────────────────────────────────────────────── #

def build_unet_torch(base_filters: int = 64,
                     dropout: float = 0.2) -> UNetStarFinder:
    """Return an uninitialised UNetStarFinder (call .to(device) yourself)."""
    return UNetStarFinder(base_filters=base_filters, dropout=dropout)


# ─────────────────────────────────────────────────────────────────── #
# Transfer-learning helpers (mirrors model.py freeze_encoder / unfreeze_all)
# ─────────────────────────────────────────────────────────────────── #

def freeze_encoder(model: UNetStarFinder) -> UNetStarFinder:
    """
    Freeze encoder blocks (enc1–enc4) and bottleneck in-place.
    Only decoder + output_conv weights remain trainable.
    Pass the frozen model's trainable parameters to the optimizer:
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), ...)
    """
    for name, param in model.named_parameters():
        if any(part in name for part in ("enc1", "enc2", "enc3", "enc4", "bottleneck")):
            param.requires_grad = False
    return model


def unfreeze_all(model: UNetStarFinder) -> UNetStarFinder:
    """Unfreeze every parameter for end-to-end fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    return model
