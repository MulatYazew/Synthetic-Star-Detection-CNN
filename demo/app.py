"""
Streamlit demo — Synthetic Star Detection CNN
=============================================

Launch:
    streamlit run demo/app.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import streamlit as st
import torch

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Star Finder — CNN Demo",
    page_icon="⭐",
    layout="wide",
)

from codes.dataset       import generate_synthetic_field
from codes.model import UNetStarFinder
from codes.evaluate      import count_stars, pixel_iou

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ── Model loading ─────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(ROOT, "models")

def _load_model(path: str):
    if not os.path.exists(path):
        return None
    m = UNetStarFinder(base_filters=64, dropout=0.2).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m

@st.cache_resource(show_spinner="Loading models …")
def load_all_models():
    return {
        "Synthetic":        _load_model(
            os.path.join(MODELS_DIR, "star_finder_synthetic.pt")),
        "Transfer-learned": _load_model(
            os.path.join(MODELS_DIR, "star_finder_tl.pt")),
        "Scratch (real)":   _load_model(
            os.path.join(MODELS_DIR, "star_finder_scratch.pt")),
    }

models    = load_all_models()
available = {k: v for k, v in models.items() if v is not None}

# ── Helpers ───────────────────────────────────────────────────────────────────

def normalise(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        return (2.0 * (arr - lo) / (hi - lo) - 1.0).astype(np.float32)
    return arr.astype(np.float32)


def torch_infer(model, image_2d: np.ndarray) -> np.ndarray:
    """Run model on one (H,W) image; return (H,W) probability map."""
    inp = normalise(image_2d)[np.newaxis, np.newaxis]   # (1,1,H,W)
    x   = torch.from_numpy(inp).to(device)
    with torch.no_grad():
        out = model(x)
    return out.squeeze().cpu().numpy()


def predict_and_figure(image_2d: np.ndarray,
                       true_mask_2d: np.ndarray | None,
                       selected_models: dict,
                       threshold: float) -> plt.Figure:
    """Build a summary figure for one image across multiple models."""
    n_models = len(selected_models)
    ncols    = 2 + n_models
    has_truth = true_mask_2d is not None

    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.5),
                             constrained_layout=True)

    inp = normalise(image_2d)
    axes[0].imshow(inp, cmap="gray", origin="upper")
    axes[0].set_title("Input image", fontsize=11)
    axes[0].axis("off")

    if has_truth:
        axes[1].imshow(true_mask_2d, cmap="gray", origin="upper")
        axes[1].set_title("True mask", fontsize=11)
        axes[1].axis("off")
        pred_start = 2
    else:
        axes[1].axis("off")
        pred_start = 1

    for col_i, (name, mdl) in enumerate(selected_models.items()):
        ax       = axes[pred_start + col_i]
        raw      = torch_infer(mdl, image_2d)
        bin_mask = (raw > threshold).astype(float)
        n_pred   = count_stars(bin_mask, threshold)

        ax.imshow(bin_mask, cmap="gray", origin="upper")

        if has_truth:
            iou_val = pixel_iou(true_mask_2d, raw, threshold)
            ax.set_title(f"{name}\nIoU={iou_val:.3f}  stars={n_pred}", fontsize=10)
        else:
            ax.set_title(f"{name}\nstars detected={n_pred}", fontsize=10)
        ax.axis("off")

    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⭐ Star Finder CNN")
st.sidebar.markdown(
    "A U-Net trained to segment stars in astronomical images.  \n"
    "*(Lectures 12 & 13 — Signal & Imaging Acquisition and Modelling)*"
)
st.sidebar.caption(f"Device: `{device}`  |  torch {torch.__version__}")

if not available:
    st.sidebar.error(
        "No trained model found in `models/`.  \n"
        "Run `python codes/train.py` first."
    )
    st.stop()

selected_names = st.sidebar.multiselect(
    "Models to compare",
    options=list(available.keys()),
    default=[list(available.keys())[0]],
)
selected_models = {k: available[k] for k in selected_names if k in available}

threshold = st.sidebar.slider(
    "Detection threshold",
    min_value=0.1, max_value=0.95,
    value=0.5, step=0.05,
)

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_syn, tab_real, tab_about = st.tabs([
    "🔭 Synthetic field",
    "📡 Upload real image",
    "ℹ️  About",
])

# ============================================================
# Tab 1 — Synthetic
# ============================================================
with tab_syn:
    st.header("Generate a synthetic stellar field")
    st.markdown(
        "Stars are modelled as Gaussian PSFs: "
        r"$\Sigma(x,y)=\frac{A_0}{2\pi\sigma^2}e^{-(x^2+y^2)/(2\sigma^2)}$, "
        r" with $A_0=(S/N)\cdot 2\pi\sigma^2$."
    )

    col_sl1, col_sl2, col_sl3, col_sl4 = st.columns(4)
    with col_sl1:
        image_size = st.select_slider(
            "Image size (px)", options=[64, 128, 256, 512], value=256)
    with col_sl2:
        n_stars = st.slider("Number of stars", 1, 10, 4)
    with col_sl3:
        fwhm = st.slider("FWHM (px)", 4.0, 40.0, 12.0, step=1.0)
    with col_sl4:
        snr_log = st.slider(
            "S/N (log scale)", 0.3, 4.0, 1.5, step=0.1,
            format="10^%.1f")
        snr = 10.0 ** snr_log

    seed = st.number_input("Random seed", value=42, step=1)

    if st.button("Generate & predict", type="primary"):
        rng = np.random.default_rng(int(seed))
        img_arr, mask_arr = generate_synthetic_field(
            image_size=image_size, n_stars=n_stars,
            fwhm=fwhm, snr=snr, rng=rng)

        if not selected_models:
            st.warning("Select at least one model in the sidebar.")
        else:
            with st.spinner("Running inference …"):
                fig = predict_and_figure(
                    img_arr.squeeze(), mask_arr.squeeze(),
                    selected_models, threshold)
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Metrics")
            rows = []
            for name, mdl in selected_models.items():
                raw = torch_infer(mdl, img_arr.squeeze())
                iou = pixel_iou(mask_arr.squeeze(), raw, threshold)
                n_p = count_stars(raw, threshold)
                n_t = count_stars(mask_arr.squeeze())
                rows.append({"Model": name, "IoU": f"{iou:.4f}",
                             "Stars true": n_t, "Stars pred": n_p})
            st.table(rows)


# ============================================================
# Tab 2 — Upload real image
# ============================================================
with tab_real:
    st.header("Upload a real astronomical image")
    st.markdown(
        "Supported formats: **PNG / TIFF / NPY / FITS** (`.fits`, `.fits.fz`).  \n"
        "For FITS files the app reads the first science extension."
    )

    uploaded = st.file_uploader(
        "Choose an image file",
        type=["png", "tif", "tiff", "npy", "fits", "fz"],
    )

    if uploaded is not None:
        ext = uploaded.name.lower().split(".")[-1]

        try:
            if ext == "npy":
                raw_img = np.load(uploaded).astype(np.float32)

            elif ext in ("fits", "fz"):
                import tempfile
                from astropy.io import fits as astropy_fits
                with tempfile.NamedTemporaryFile(suffix=".fits") as tmp:
                    tmp.write(uploaded.read())
                    tmp.flush()
                    with astropy_fits.open(tmp.name) as hdul:
                        raw_img = None
                        for hdu in hdul:
                            if hdu.data is not None and hdu.data.ndim >= 2:
                                raw_img = hdu.data.astype(np.float32)
                                if raw_img.ndim > 2:
                                    raw_img = raw_img[0]
                                break
                    if raw_img is None:
                        st.error("No 2-D image HDU found in FITS file.")
                        st.stop()

            else:
                from PIL import Image
                pil_img = Image.open(uploaded).convert("L")
                raw_img = np.array(pil_img, dtype=np.float32)

        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        if raw_img.ndim > 2:
            raw_img = raw_img.squeeze()

        H, W = raw_img.shape
        H16, W16 = (H // 16) * 16, (W // 16) * 16
        if H16 != H or W16 != W:
            st.info(
                f"Image cropped from {H}×{W} to {H16}×{W16} "
                f"to satisfy U-Net divisibility-by-16 requirement."
            )
            raw_img = raw_img[:H16, :W16]

        st.subheader("Preview")
        fig_prev, ax_prev = plt.subplots(figsize=(6, 5))
        ax_prev.imshow(raw_img, cmap="gray", origin="lower")
        ax_prev.axis("off")
        ax_prev.set_title(f"Uploaded: {uploaded.name}  ({H16}×{W16})")
        st.pyplot(fig_prev)
        plt.close(fig_prev)

        if not selected_models:
            st.warning("Select at least one model in the sidebar.")
        elif st.button("Run star detection", type="primary"):
            with st.spinner("Running inference …"):
                fig = predict_and_figure(
                    raw_img, None, selected_models, threshold)
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Detected star counts")
            rows = []
            for name, mdl in selected_models.items():
                raw = torch_infer(mdl, raw_img)
                n_p = count_stars(raw, threshold)
                rows.append({"Model": name, "Stars detected": n_p})
            st.table(rows)


# ============================================================
# Tab 3 — About
# ============================================================
with tab_about:
    st.header("About this project")
    st.markdown("""
### Photometry AI: Identification of Stars in an Image

**Course:** Signal and Imaging Acquisition and Modelling in Environment
**Lectures 12 & 13** — Federico De Guio · Alejandro Benitez-Llambay

---

#### Architecture: U-Net (Fully Convolutional Network)

```
Input (H × W × 1)
  ↓
Encoder: 4× [Conv3×3 → BN → ReLU] × 2  +  MaxPool2×2
  Filters: 64 → 128 → 256 → 512
  ↓
Bottleneck: [Conv3×3 → BN → ReLU] × 2  (1024 filters)
  ↓
Decoder: 4× ConvTranspose(2×2) → Concat(skip) → [Conv3×3 → BN → ReLU] × 2
  ↓
Output: Conv1×1 → Sigmoid
```

Fully convolutional — trained on 64×64 patches, applies to any image divisible by 16.

---

#### Training strategy

| Model | Training data | Notes |
|---|---|---|
| Synthetic | 10 000 generated fields | FWHM ∈ [8,32] px, S/N ∈ [2, 10 000] |
| Transfer-learned | Real labelled patches | Encoder frozen → decoder fine-tuned, then full fine-tune |
| Scratch (real) | Real labelled patches | Full model trained from random initialisation |

---

#### Key metric: Binary IoU

$$\\text{IoU} = \\frac{\\text{TP}}{\\text{TP} + \\text{FP} + \\text{FN}}$$

Pixel labels: **1 = star** (within 3σ of the star centre), **0 = background**.

---

#### File layout

```
codes/
  dataset.py           ← synthetic generation + real data loading
  model.py             ← U-Net (MPS)
  train.py             ← optimised training script
  transfer_learning.py ← TL fine-tuning pipeline
  evaluate.py          ← metrics and visualisation helpers
notebooks/
  synthetic_star_detection.ipynb  ← complete analysis notebook
demo/
  app.py               ← this Streamlit app
models/                ← saved model state-dicts (.pt)
figures/               ← training curves and prediction plots
star-dataset/          ← real FITS + labelled npy files
```
""")
