"""Centralized path and hyperparameter configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Top-level directories ──────────────────────────────────────────── #
DATASET_DIR = ROOT / "star-dataset"
MODELS_DIR  = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# ── Results subdirectories ─────────────────────────────────────────── #
TRAINING_RESULTS_DIR   = RESULTS_DIR / "training"        # training logs & curves
CHECKPOINTS_DIR        = RESULTS_DIR / "checkpoints"     # per-experiment .pt files
EVALUATION_RESULTS_DIR = RESULTS_DIR / "evaluation"      # metrics, reports, qual. figures
PERFORMANCE_STUDY_DIR  = RESULTS_DIR / "performance_study"  # comparisons & benchmarks
FIGURES_DIR            = RESULTS_DIR / "figures"         # notebook-generated figures
REAL_RESULTS_DIR       = RESULTS_DIR / "real"
TL_RESULTS_DIR         = RESULTS_DIR / "transfer_learning"

# ── Dataset paths ──────────────────────────────────────────────────── #
CACHE_PATH       = DATASET_DIR / "synthetic_dataset_cache.npz"
SYNTHETIC_STARS  = DATASET_DIR / "synthetic_stars.npy"
SYNTHETIC_LABELS = DATASET_DIR / "synthetic_labels.npy"
SYNTHETIC_META   = DATASET_DIR / "metadata" / "synthetic_metadata.json"
REAL_STARS_PATH  = DATASET_DIR / "real_stars.npy"
REAL_LABELS_PATH = DATASET_DIR / "real_stars_labels.npy"

# ── Final model paths ──────────────────────────────────────────────── #
SYNTHETIC_MODEL_PATH = MODELS_DIR / "synthetic_model.pt"
TL_MODEL_PATH        = MODELS_DIR / "transfer_learning_model.pt"
SCRATCH_MODEL_PATH   = MODELS_DIR / "real_scratch_model.pt"

# ── Training hyperparameters ───────────────────────────────────────── #
IMAGE_SIZE   = 64
N_TRAIN      = 10_000
N_VAL        = 2_000
BATCH_SIZE   = 64
BASE_FILTERS = 32
EPOCHS       = 50
LR           = 1e-4
SEED         = 42

N_SAMPLES        = 14_000   # total images in the flat synthetic dataset
FORCE_REGENERATE = False    # True → regenerate dataset even if one exists

EARLY_STOP_PATIENCE = 12
LR_REDUCE_PATIENCE  = 5
LR_REDUCE_FACTOR    = 0.5
LR_MIN              = 1e-6

# Loss functions to compare in every pipeline
LOSS_NAMES   = ["bce", "dice", "bce_dice"]
LOSS_DISPLAY = {"bce": "BCE", "dice": "Dice", "bce_dice": "BCE + Dice"}
