#!/usr/bin/env bash
set -euo pipefail

if [[ "${DICACHE_GPU_TESTS_ALLOWED:-0}" != "1" ]]; then
  echo "Refusing compile matrix. Set DICACHE_GPU_TESTS_ALLOWED=1 only after the target GPU is allocated." >&2
  exit 2
fi
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != *,* ]] || {
  echo "Refusing compile matrix: CUDA_VISIBLE_DEVICES must name exactly one allocated GPU." >&2
  exit 2
}
DEVICE="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
[[ -n "$DEVICE" ]] || {
  echo "Refusing compile matrix: empty GPU identifier." >&2
  exit 2
}

if ! DEVICE_PIDS="$(nvidia-smi -i "$DEVICE" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
  echo "Refusing compile matrix: nvidia-smi process query failed for GPU $DEVICE." >&2
  exit 3
fi
if [[ -n "$(sed '/^[[:space:]]*$/d' <<<"$DEVICE_PIDS")" ]]; then
  echo "Refusing compile matrix: GPU $DEVICE has an existing compute process." >&2
  exit 3
fi
if ! DEVICE_STATE="$(nvidia-smi -i "$DEVICE" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null)"; then
  echo "Refusing compile matrix: nvidia-smi state query failed for GPU $DEVICE." >&2
  exit 3
fi
if ! awk -F',' 'NR == 1 {for (i=1; i<=2; i++) gsub(/[[:space:]]/, "", $i); valid=(NF == 2 && $1 ~ /^[0-9]+([.][0-9]+)?$/ && $2 ~ /^[0-9]+([.][0-9]+)?$/); busy=($1+0 > 5 || $2+0 > 1024)} END {exit !(NR == 1 && valid && !busy)}' <<<"$DEVICE_STATE"; then
  echo "Refusing compile matrix: GPU $DEVICE is busy or returned invalid telemetry." >&2
  exit 3
fi

BASELINE_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd -P)"
PIXARC_ROOT="$(cd -- "$BASELINE_ROOT/../../.." && pwd -P)"
UPSTREAM_ROOT="$PIXARC_ROOT/third-party/JiT"
export PYTHONPATH="$UPSTREAM_ROOT:$BASELINE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python "$BASELINE_ROOT/scripts/run_compile_matrix.py" "$@"
