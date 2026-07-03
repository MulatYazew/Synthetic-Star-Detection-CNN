---
title: StarNet
emoji: ⭐
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
pinned: false
---

# StarNet — U-Net Stellar Segmentation Demo

Interactive demo of a U-Net trained to detect stars in astronomical images
(synthetic Gaussian-PSF fields and real FITS/JPG/PNG/NPY uploads).

This folder is a self-contained deployment target for Hugging Face Spaces —
it does not depend on anything outside itself. The full training pipeline,
notebook, and dataset live in the parent repository.

## Configuration

Model checkpoints are not bundled in this Space (kept small on purpose).
They're downloaded at runtime from a separate Hugging Face Hub **model**
repo. Set these under the Space's Settings → Variables and secrets:

| Name              | Required | Description                                              |
|-------------------|----------|------------------------------------------------------------|
| `HF_MODEL_REPO_ID` | yes      | e.g. `your-username/starnet-checkpoints`                  |
| `HF_TOKEN`         | only if the model repo is private | HF access token with read access |

The model repo must contain:
- `star_finder_synthetic.pt`
- `star_finder_tl.pt`
- `star_finder_scratch.pt`

## Deploying

1. Create a Hugging Face model repo and upload the three `.pt` checkpoints
   (from the parent repo's `models/` folder) to it — via the Hub web UI or
   `huggingface_hub.upload_file`.
2. Create a new Space (SDK: Streamlit) on huggingface.co.
3. Push this folder's contents to the Space's git remote.
4. Set `HF_MODEL_REPO_ID` (and `HF_TOKEN` if needed) in the Space's Settings.

See the parent repository's README for the full deployment walkthrough.
