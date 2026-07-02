# Matches the project's actual runtime (.venv reports Python 3.12.13); no
# runtime.txt / .python-version exists to pin this otherwise.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# requirements.txt lists a bare "torch" with no build constraint. On Linux,
# PyPI's default "torch" wheel bundles CUDA runtime libraries (~2GB+) that
# are useless on Render (no GPU) and blow up build time/image size. Installing
# the CPU-only wheel first satisfies that requirement before the second
# install even looks at it.
#
# No apt-get system packages (e.g. libgl1) are installed here: this repo does
# not use OpenCV or torchvision, and Pillow's PyPI wheels bundle their own
# image codecs (libjpeg/libpng/zlib), so nothing in requirements.txt needs
# them. matplotlib already runs headless via matplotlib.use("Agg") in
# demo/app.py. Re-check this if you add opencv-python, torchvision, or
# similar packages later.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# models/*.pt are stored via Git LFS (see .gitattributes) — this Dockerfile
# has no .git access (excluded by .dockerignore) so it cannot fetch LFS
# content itself. Render's git checkout, which runs before this build
# context is assembled, is what must resolve the LFS pointers into real
# weight files. This check fails the BUILD loudly with an actionable message
# if that didn't happen, instead of shipping pointer stubs that crash
# torch.load() at runtime with a cryptic error.
RUN for f in models/*.pt; do \
        if head -c 200 "$f" | grep -q "git-lfs.github.com"; then \
            echo "ERROR: $f is a Git LFS pointer file, not real model weights." >&2; \
            echo "Render did not resolve LFS content during checkout for this service." >&2; \
            echo "Verify Git LFS is enabled for this service in the Render dashboard" >&2; \
            echo "(Settings > Build & Deploy), then trigger a fresh deploy." >&2; \
            exit 1; \
        fi; \
    done

# Documentation only — Render assigns the actual port via $PORT at runtime
# and ignores EXPOSE.
EXPOSE 8501

# Shell form (not exec form) so $PORT is expanded by the shell at container
# start; Render injects $PORT as an environment variable, it is not known at
# build time.
CMD streamlit run demo/app.py --server.port=$PORT --server.address=0.0.0.0
