#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd -- "$(dirname -- "$0")" && pwd -P)"
PIXARC_ROOT="$(cd -- "$SCRIPT_ROOT/../../../.." && pwd -P)"
JIT_PYTHON="${PIXEL_REMAINDER_JIT_TEST_PYTHON:-/root/miniconda3/envs/jit/bin/python}"
PIXELGEN_PYTHON="${PIXEL_REMAINDER_PIXELGEN_RUNTIME_PYTHON:-/root/miniconda3/envs/pixelgen/bin/python}"
CONFIG_SOURCE="${PIXEL_REMAINDER_CONFIG_SOURCE:-$PIXARC_ROOT}"
if [[ ! -f "$CONFIG_SOURCE/JiT/checkpoints/JiT-B-16-256/checkpoint-last.pth" ]]; then
  SIBLING_SOURCE="$(dirname -- "$PIXARC_ROOT")/PixARC"
  if [[ -f "$SIBLING_SOURCE/JiT/checkpoints/JiT-B-16-256/checkpoint-last.pth" ]]; then
    CONFIG_SOURCE="$SIBLING_SOURCE"
  fi
fi
[[ -x "$JIT_PYTHON" ]] || { echo "JiT Python unavailable: $JIT_PYTHON" >&2; exit 2; }
[[ -x "$PIXELGEN_PYTHON" ]] || { echo "PixelGen Python unavailable: $PIXELGEN_PYTHON" >&2; exit 2; }

for NAME in \
  jit_b16_256_instrumented_full.yaml \
  jit_b16_256_fixed_i3_k2.yaml \
  jit_b16_256_prt_t0p01_h3.yaml \
  jit_b16_256_prt_t0p02_h3.yaml \
  jit_b16_256_prt_t0p04_h3.yaml
do
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" \
    "$SCRIPT_ROOT/preflight_run.py" --model JiT \
    --input-config "$CONFIG_SOURCE/JiT/methods/pixel-remainder-taylor/configs/$NAME" \
    --manifest "$PIXARC_ROOT/results/JiT/baselines/taylorseer/protocol/manifest_1k.jsonl" \
    --expected-count 1000
done

for NAME in \
  pixelgen_xl_256_instrumented_full.yaml \
  pixelgen_xl_256_fixed_i3_k2.yaml \
  pixelgen_xl_256_prt_t0p01_h3.yaml \
  pixelgen_xl_256_prt_t0p02_h3.yaml \
  pixelgen_xl_256_prt_t0p04_h3.yaml
do
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' "$PIXELGEN_PYTHON" \
    "$SCRIPT_ROOT/preflight_run.py" --model PixelGen \
    --input-config "$CONFIG_SOURCE/PixelGen/methods/pixel-remainder-taylor/configs/$NAME" \
    --manifest "$PIXARC_ROOT/results/PixelGen/baselines/taylorseer/protocol/manifest_1k.jsonl" \
    --expected-count 1000
done
