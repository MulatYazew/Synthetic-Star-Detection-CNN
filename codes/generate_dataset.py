"""
Generate the full synthetic dataset at once and save to star-dataset/.

Outputs:
    star-dataset/synthetic_stars.npy          — all images (N, 64, 64, 1)
    star-dataset/synthetic_labels.npy         — all masks  (N, 64, 64, 1)
    star-dataset/metadata/synthetic_metadata.csv / .json
    star-dataset/visual_samples/sample_NNNNN_{image|mask}.png

The dataset is NOT pre-split.  Load synthetic_stars.npy + synthetic_labels.npy
and use stratified_split_indices() (in codes/dataset.py) to obtain
train / validation / test subsets — as the notebook does.

Usage (from project root):
    python codes/generate_dataset.py
    python codes/generate_dataset.py --n-total 20000
    python codes/generate_dataset.py --no-png --n-previews 0
    python codes/generate_dataset.py --force      # overwrite existing files
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from codes.dataset import generate_and_save_flat

DATASET_ROOT   = ROOT / "star-dataset"
DEFAULT_N      = 14_000   # 10 000 train + 2 000 val + 2 000 test
IMAGE_SIZE     = 64


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate and save the flat synthetic stellar-field dataset")
    p.add_argument("--n-total",    type=int,  default=DEFAULT_N,
                   help=f"Total number of samples (default {DEFAULT_N})")
    p.add_argument("--image-size", type=int,  default=IMAGE_SIZE,
                   help=f"Patch side length in pixels (default {IMAGE_SIZE})")
    p.add_argument("--seed",       type=int,  default=42,
                   help="Random seed (default 42)")
    p.add_argument("--no-png",     action="store_true",
                   help="Skip PNG preview generation")
    p.add_argument("--n-previews", type=int,  default=20,
                   help="Number of PNG previews to save (default 20)")
    p.add_argument("--force",      action="store_true",
                   help="Regenerate even if synthetic_stars.npy already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    stars_path = DATASET_ROOT / "synthetic_stars.npy"

    print("=" * 60)
    print("  Synthetic Star Dataset Generator")
    print(f"  Output root : {DATASET_ROOT}")
    print(f"  Image size  : {args.image_size} × {args.image_size}")
    print(f"  Total       : {args.n_total:,} samples  (seed={args.seed})")
    print("=" * 60)

    if not args.force and stars_path.exists():
        print(f"\n[skip] {stars_path.name} already exists. Use --force to regenerate.")
        return

    t0 = time.perf_counter()
    generate_and_save_flat(
        n_samples        = args.n_total,
        dataset_root     = DATASET_ROOT,
        image_size       = args.image_size,
        seed             = args.seed,
        save_png_previews= not args.no_png,
        n_png_previews   = args.n_previews,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s  ({args.n_total / elapsed:.0f} samples/s)")
    print(f"Root: {DATASET_ROOT}")


if __name__ == "__main__":
    main()
