#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd -- "$(dirname -- "$0")" && pwd -P)"
PIXARC_ROOT="$(cd -- "$SCRIPT_ROOT/../../../.." && pwd -P)"
JIT_PYTHON="${PIXEL_REMAINDER_JIT_TEST_PYTHON:-/root/miniconda3/envs/jit/bin/python}"
PIXELGEN_TEST_PYTHON="${PIXEL_REMAINDER_PIXELGEN_TEST_PYTHON:-$JIT_PYTHON}"
PIXELGEN_RUNTIME_PYTHON="${PIXEL_REMAINDER_PIXELGEN_RUNTIME_PYTHON:-/root/miniconda3/envs/pixelgen/bin/python}"
for PYTHON_BIN in "$JIT_PYTHON" "$PIXELGEN_TEST_PYTHON" "$PIXELGEN_RUNTIME_PYTHON"; do
  [[ -x "$PYTHON_BIN" ]] || { echo "Python executable unavailable: $PYTHON_BIN" >&2; exit 2; }
done

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  "$JIT_PYTHON" -m pytest -q -p no:cacheprovider \
  "$PIXARC_ROOT/JiT/methods/pixel-remainder-taylor/tests"

# Deliberately a new interpreter: JiT and PixelGen expose same-named packages.
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  "$PIXELGEN_TEST_PYTHON" -m pytest -q -p no:cacheprovider \
  "$PIXARC_ROOT/PixelGen/methods/pixel-remainder-taylor/tests"

# The frozen PixelGen sidecar binds torch 2.7.1+cu126, so validate it in the
# production model-family environment even when pytest lives elsewhere.
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp \
  "$PIXELGEN_RUNTIME_PYTHON" \
  "$PIXARC_ROOT/PixelGen/methods/pixel-remainder-taylor/tests/validate_real_manifest.py"
