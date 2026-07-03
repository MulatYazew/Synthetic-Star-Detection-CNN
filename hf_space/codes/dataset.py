"""
Synthetic stellar field generation and real data loading.

Physics model:
  - Star PSF:  Σ(x,y) = A0/(2πσ²) · exp(-(x²+y²)/(2σ²))
  - FWHM      = √(2ln2) · σ ≈ 2.355σ
  - S/N       = A0 / (2πσ²)  →  A0 = (S/N) · 2πσ²
  - Label pixels within 3σ of the star centre as star (binary mask = 1)
  - Buffer zone between overlapping stars kept in mask generation
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #

def sigma_from_fwhm(fwhm: float) -> float:
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def flux_from_snr(snr: float, sigma: float) -> float:
    return snr * 2.0 * np.pi * sigma ** 2


def place_stars(image_size: int,
                 n_stars: int,
                 sigma: float,
                 snr: float,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Returns (clean_field, mask, star_positions).
    clean_field contains the noiseless PSF-convolved star flux.
    mask is 1 inside 3σ of each star (with buffer gaps at midpoints).
    """
    field = np.zeros((image_size, image_size), dtype=np.float32)
    mask  = np.zeros((image_size, image_size), dtype=np.float32)

    A0 = flux_from_snr(snr, sigma)

    # Random positions (keep stars inside a margin so they are not clipped)
    margin = max(1, int(3 * sigma))
    lo, hi = margin, image_size - margin
    if lo >= hi:
        lo, hi = 0, image_size

    positions = []
    for _ in range(n_stars):
        cx = int(rng.integers(lo, hi))
        cy = int(rng.integers(lo, hi))
        positions.append((cy, cx))          # (row, col)
        field[cy, cx] += A0                 # delta function → will be convolved

    # Convolve the entire field with the PSF at once (efficient for many stars)
    clean_field = gaussian_filter(field, sigma=sigma)

    # Build the label mask: 3σ disc around every star, with buffer between stars
    r3 = 3 * sigma
    yy, xx = np.mgrid[0:image_size, 0:image_size]

    dist2_all = []
    for (cy, cx) in positions:
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        dist2_all.append(dist2)

    for i, (cy, cx) in enumerate(positions):
        inside_3sigma = dist2_all[i] <= r3 ** 2
        # Buffer: a pixel is labelled as star i only if it is closer to star i
        # than to any other star (Voronoi partitioning within the 3σ disc)
        closest = np.ones((image_size, image_size), dtype=bool)
        for j, _ in enumerate(positions):
            if j != i:
                closest &= (dist2_all[i] <= dist2_all[j])
        mask[inside_3sigma & closest] = 1.0

    return clean_field, mask, positions


def generate_synthetic_field(image_size: int = 64,
                              n_stars: int = 3,
                              fwhm: float = None,
                              snr: float = None,
                              rng: np.random.Generator = None,
                              return_metadata: bool = False):
    """
    Generate one (image, mask) pair of synthetic stars with Gaussian PSF.

    Parameters
    ----------
    image_size      : side length of the square image (pixels)
    n_stars         : number of stars to place
    fwhm            : PSF FWHM in pixels; sampled from [8, 32] if None
    snr             : peak signal-to-noise ratio; log-uniform in [2, 10000] if None
    rng             : numpy random Generator (created if None)
    return_metadata : if True, also return a metadata dict

    Returns
    -------
    image : (image_size, image_size, 1) float32 array, normalised to [-1, 1]
    mask  : (image_size, image_size, 1) float32 binary array
    meta  : dict with n_stars/fwhm/sigma/snr/flux/star_coords (only if return_metadata=True)
    """
    if rng is None:
        rng = np.random.default_rng()

    if fwhm is None:
        fwhm = rng.uniform(8.0, 32.0)
    if snr is None:
        snr = float(np.exp(rng.uniform(np.log(2.0), np.log(10_000.0))))

    sigma = sigma_from_fwhm(fwhm)
    clean_field, mask, positions = place_stars(image_size, n_stars, sigma, snr, rng)

    # Add Gaussian white noise with variance = 1
    noise = rng.standard_normal((image_size, image_size)).astype(np.float32)
    image = clean_field + noise

    # Normalise image to [-1, 1] using the global min/max
    img_min = image.min()
    img_max = image.max()
    if img_max > img_min:
        image = 2.0 * (image - img_min) / (img_max - img_min) - 1.0

    if return_metadata:
        meta = {
            "n_stars":     n_stars,
            "fwhm":        round(float(fwhm), 4),
            "sigma":       round(float(sigma), 4),
            "snr":         round(float(snr), 4),
            "flux":        round(float(flux_from_snr(snr, sigma)), 4),
            "star_coords": [[int(cy), int(cx)] for cy, cx in positions],
        }
        return image[..., np.newaxis], mask[..., np.newaxis], meta
    return image[..., np.newaxis], mask[..., np.newaxis]


def generate_dataset(n_samples: int = 2000,
                     image_size: int = 64,
                     min_stars: int = 1,
                     max_stars: int = 5,
                     fwhm_range: tuple = (8.0, 32.0),
                     snr_range: tuple = (2.0, 10_000.0),
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a full training / validation dataset.

    Returns
    -------
    X : (n_samples, image_size, image_size, 1) float32
    Y : (n_samples, image_size, image_size, 1) float32 binary
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n_samples, image_size, image_size, 1), dtype=np.float32)
    Y = np.empty((n_samples, image_size, image_size, 1), dtype=np.float32)

    for i in tqdm(range(n_samples), desc="Generating dataset", unit="sample"):
        n_stars = int(rng.integers(min_stars, max_stars + 1))
        fwhm = rng.uniform(*fwhm_range)
        snr  = float(np.exp(rng.uniform(np.log(snr_range[0]), np.log(snr_range[1]))))
        X[i], Y[i] = generate_synthetic_field(image_size, n_stars, fwhm, snr, rng)

    return X, Y


# --------------------------------------------------------------------------- #
# Real data loading
# --------------------------------------------------------------------------- #

def load_real_patches(real_stars_path: str,
                      real_labels_path: str,
                      patch_size: int = 64,
                      stride: int = 64,
                      max_patches: int = None,
                      seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Cut a real astronomical image into (patch_size × patch_size) patches.

    Parameters
    ----------
    real_stars_path  : path to real_stars.npy  (2-D flux image)
    real_labels_path : path to real_stars_labels.npy  (2-D binary mask)
    patch_size       : size of each square patch
    stride           : step between patches
    max_patches      : subsample this many patches at random (all if None)
    seed             : random seed for subsampling

    Returns
    -------
    X : (N, patch_size, patch_size, 1) float32, normalised to [-1, 1]
    Y : (N, patch_size, patch_size, 1) float32 binary
    """
    flux   = np.load(real_stars_path).astype(np.float32)
    labels = np.load(real_labels_path).astype(np.float32)

    H, W = flux.shape
    patches_x, patches_y = [], []

    row_starts = range(0, H - patch_size + 1, stride)
    col_starts = range(0, W - patch_size + 1, stride)
    for r in tqdm(row_starts, desc="Extracting patches", unit="row"):
        for c in col_starts:
            px = flux[r:r+patch_size, c:c+patch_size]
            py = labels[r:r+patch_size, c:c+patch_size]

            # Normalise each patch independently
            pmin, pmax = px.min(), px.max()
            if pmax > pmin:
                px = 2.0 * (px - pmin) / (pmax - pmin) - 1.0

            patches_x.append(px[..., np.newaxis])
            patches_y.append(py[..., np.newaxis])

    X = np.stack(patches_x).astype(np.float32)
    Y = np.stack(patches_y).astype(np.float32)

    if max_patches is not None and max_patches < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_patches, replace=False)
        X, Y = X[idx], Y[idx]

    return X, Y


def load_fits_image(fits_path: str, hdu_index: int = 2,
                    row_start: int = 0, row_end: int = 1024,
                    col_start: int = 0, col_end: int = 1024) -> np.ndarray:
    """
    Load a tile from a FITS/FITS.fz file and normalise to [-1, 1].
    Returns (H, W, 1) float32 array.
    """
    from astropy.io import fits as astropy_fits
    with astropy_fits.open(fits_path) as hdul:
        data = hdul[hdu_index].data[row_start:row_end, col_start:col_end]

    data = data.astype(np.float32)
    dmin, dmax = data.min(), data.max()
    if dmax > dmin:
        data = 2.0 * (data - dmin) / (dmax - dmin) - 1.0

    return data[..., np.newaxis]


# --------------------------------------------------------------------------- #
# Disk persistence helpers
# --------------------------------------------------------------------------- #

def _save_png(array: np.ndarray, path: Path) -> None:
    """Save (H,W) or (H,W,1) float32 array as a grayscale PNG via matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arr = array.squeeze()
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    plt.imsave(str(path), arr, cmap="gray", vmin=0, vmax=1)
    plt.close("all")


def stratified_split_indices(
    labels: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (train_idx, val_idx, test_idx) that preserve the class distribution
    of ``labels`` (e.g. n_stars per image) across all three splits.

    Samples within each label group are shuffled, then distributed
    proportionally.  Indices inside each split are shuffled before returning.
    """
    rng = np.random.default_rng(seed)
    n_total = len(labels)
    if n_train + n_val + n_test > n_total:
        raise ValueError(
            f"n_train+n_val+n_test ({n_train+n_val+n_test}) > n_total ({n_total})")

    p_train = n_train / n_total
    p_val   = n_val   / n_total

    train_parts: list[np.ndarray] = []
    val_parts:   list[np.ndarray] = []
    test_parts:  list[np.ndarray] = []

    for lbl in np.unique(labels):
        group = np.where(labels == lbl)[0]
        rng.shuffle(group)
        n = len(group)
        n_tr = int(round(n * p_train))
        n_vl = int(round(n * p_val))
        n_te = n - n_tr - n_vl
        train_parts.append(group[:n_tr])
        val_parts.append(group[n_tr: n_tr + n_vl])
        test_parts.append(group[n_tr + n_vl: n_tr + n_vl + n_te])

    train_idx = np.concatenate(train_parts)
    val_idx   = np.concatenate(val_parts)
    test_idx  = np.concatenate(test_parts)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx[:n_train], val_idx[:n_val], test_idx[:n_test]


def generate_and_save_flat(
    n_samples: int = 14_000,
    dataset_root: "Path | str" = None,
    image_size: int = 64,
    min_stars: int = 1,
    max_stars: int = 5,
    fwhm_range: tuple = (8.0, 32.0),
    snr_range: tuple = (2.0, 10_000.0),
    seed: int = 42,
    save_png_previews: bool = True,
    n_png_previews: int = 20,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Generate the full dataset at once and persist to ``dataset_root``:

        synthetic_stars.npy          — (n_samples, H, W, 1) float32 images
        synthetic_labels.npy         — (n_samples, H, W, 1) float32 masks
        metadata/synthetic_metadata.csv / .json
        visual_samples/sample_NNNNN_{image|mask}.png  (first n_png_previews)

    Returns
    -------
    X       : (n_samples, image_size, image_size, 1) float32
    Y       : (n_samples, image_size, image_size, 1) float32
    records : list of per-sample metadata dicts
    """
    dataset_root = Path(dataset_root)
    metadata_dir = dataset_root / "metadata"
    visual_dir   = dataset_root / "visual_samples"

    for d in [metadata_dir, visual_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    records: list[dict] = []

    X = np.empty((n_samples, image_size, image_size, 1), dtype=np.float32)
    Y = np.empty((n_samples, image_size, image_size, 1), dtype=np.float32)

    for i in tqdm(range(n_samples), desc="Generating dataset", unit="sample"):
        n_stars = int(rng.integers(min_stars, max_stars + 1))
        fwhm    = float(rng.uniform(*fwhm_range))
        snr     = float(np.exp(rng.uniform(np.log(snr_range[0]),
                                           np.log(snr_range[1]))))

        img, msk, meta = generate_synthetic_field(
            image_size, n_stars, fwhm, snr, rng, return_metadata=True)
        X[i] = img
        Y[i] = msk
        meta["sample_id"] = i
        records.append(meta)

        if save_png_previews and i < n_png_previews:
            _save_png(img, visual_dir / f"sample_{i:05d}_image.png")
            _save_png(msk, visual_dir / f"sample_{i:05d}_mask.png")

    # ── Save flat arrays ──────────────────────────────────────────────── #
    np.save(dataset_root / "synthetic_stars.npy",  X)
    np.save(dataset_root / "synthetic_labels.npy", Y)

    # ── CSV ──────────────────────────────────────────────────────────── #
    csv_path   = metadata_dir / "synthetic_metadata.csv"
    fieldnames = ["sample_id", "n_stars", "fwhm", "sigma", "snr", "flux", "star_coords"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = {k: rec[k] for k in fieldnames}
            row["star_coords"] = json.dumps(rec["star_coords"])
            writer.writerow(row)

    # ── JSON ─────────────────────────────────────────────────────────── #
    json_path = metadata_dir / "synthetic_metadata.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"  Saved {n_samples:,} samples  →  {dataset_root}/")
    print(f"  Images   →  synthetic_stars.npy")
    print(f"  Labels   →  synthetic_labels.npy")
    print(f"  Metadata →  {csv_path.name}  {json_path.name}")
    return X, Y, records


# --------------------------------------------------------------------------- #
# Dataset preparation  (check-or-generate)
# --------------------------------------------------------------------------- #

def dataset_is_complete(dataset_root: "Path | str", n_samples: int) -> bool:
    """Return True if a complete synthetic dataset of n_samples exists in dataset_root."""
    root   = Path(dataset_root)
    stars  = root / "synthetic_stars.npy"
    labels = root / "synthetic_labels.npy"
    meta   = root / "metadata" / "synthetic_metadata.json"
    if not (stars.exists() and labels.exists() and meta.exists()):
        return False
    try:
        arr = np.load(stars, mmap_mode="r")
        return int(arr.shape[0]) == n_samples
    except Exception:
        return False


def prepare_dataset(
    dataset_root: "Path | str",
    n_samples: int = 14_000,
    image_size: int = 64,
    min_stars: int = 1,
    max_stars: int = 5,
    fwhm_range: tuple = (8.0, 32.0),
    snr_range: tuple = (2.0, 10_000.0),
    seed: int = 42,
    force_regenerate: bool = False,
    save_png_previews: bool = True,
    n_png_previews: int = 20,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Return (X, Y, records) for the full synthetic dataset.

    Loads from disk when a complete dataset already exists and
    force_regenerate is False.  Generates and saves a fresh dataset
    otherwise.  Python, NumPy, and PyTorch random seeds are all fixed to
    ``seed`` before generation so every new dataset is reproducible.
    """
    import random

    dataset_root = Path(dataset_root)

    if not force_regenerate and dataset_is_complete(dataset_root, n_samples):
        print("Existing synthetic dataset found.")
        print("Skipping dataset generation.")
        print("Loading dataset for training...")
        X = np.load(dataset_root / "synthetic_stars.npy")
        Y = np.load(dataset_root / "synthetic_labels.npy")
        with open(dataset_root / "metadata" / "synthetic_metadata.json") as fh:
            records = json.load(fh)
        return X, Y, records

    if force_regenerate:
        print("FORCE_REGENERATE=True — generating a fresh dataset.")
    else:
        print(f"No complete dataset found. Generating {n_samples:,} images ...")

    # Fix all random seeds before generation for full reproducibility
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass

    dataset_root.mkdir(parents=True, exist_ok=True)
    return generate_and_save_flat(
        n_samples=n_samples,
        dataset_root=dataset_root,
        image_size=image_size,
        min_stars=min_stars,
        max_stars=max_stars,
        fwhm_range=fwhm_range,
        snr_range=snr_range,
        seed=seed,
        save_png_previews=save_png_previews,
        n_png_previews=n_png_previews,
    )
