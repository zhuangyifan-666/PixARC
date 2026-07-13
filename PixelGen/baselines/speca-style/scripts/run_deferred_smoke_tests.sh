#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --full-config FILE --candidate-config FILE --manifest FILE --output-root DIR" >&2
}

FULL_CONFIG=""
CANDIDATE_CONFIG=""
MANIFEST=""
OUTPUT_ROOT=""
while (($#)); do
  case "$1" in
    --full-config) [[ $# -ge 2 ]] || { usage; exit 2; }; FULL_CONFIG="$2"; shift 2 ;;
    --candidate-config) [[ $# -ge 2 ]] || { usage; exit 2; }; CANDIDATE_CONFIG="$2"; shift 2 ;;
    --manifest) [[ $# -ge 2 ]] || { usage; exit 2; }; MANIFEST="$2"; shift 2 ;;
    --output-root) [[ $# -ge 2 ]] || { usage; exit 2; }; OUTPUT_ROOT="$2"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -f "$FULL_CONFIG" && -f "$CANDIDATE_CONFIG" && -f "$MANIFEST" && -n "$OUTPUT_ROOT" ]] || { usage; exit 2; }
if [[ "${SPECA_GPU_TESTS_ALLOWED:-0}" != "1" ]]; then
  echo "Deferred GPU smoke tests are locked. Set SPECA_GPU_TESTS_ALLOWED=1 only after the target GPU is idle." >&2
  exit 2
fi
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != *,* ]] || {
  echo "Smoke wrapper requires exactly one explicitly visible allocated GPU." >&2
  exit 2
}
DEVICE="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
[[ -n "$DEVICE" ]] || { echo "Refusing smoke test: empty GPU identifier." >&2; exit 2; }
if ! DEVICE_PIDS="$(nvidia-smi -i "$DEVICE" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)" \
  || [[ -n "$(sed '/^[[:space:]]*$/d' <<<"$DEVICE_PIDS")" ]]; then
  echo "Refusing smoke test: GPU $DEVICE has a compute process or could not be inspected." >&2
  exit 3
fi
if ! DEVICE_STATE="$(nvidia-smi -i "$DEVICE" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null)" \
  || ! awk -F',' 'NR == 1 {for(i=1;i<=2;i++) gsub(/[[:space:]]/, "", $i); valid=(NF==2 && $1~/^[0-9]+([.][0-9]+)?$/ && $2~/^[0-9]+([.][0-9]+)?$/); busy=($1+0>5 || $2+0>1024)} END {exit !(NR==1 && valid && !busy)}' <<<"$DEVICE_STATE"; then
  echo "Refusing smoke test: GPU $DEVICE is busy or returned invalid telemetry." >&2
  exit 3
fi

SCRIPT_ROOT="$(cd -- "$(dirname -- "$0")" && pwd -P)"
BASELINE_ROOT="$(cd -- "$SCRIPT_ROOT/.." && pwd -P)"
PIXARC_ROOT="$(cd -- "$BASELINE_ROOT/../../.." && pwd -P)"
MODEL_FAMILY="$(basename -- "$(cd -- "$BASELINE_ROOT/../.." && pwd -P)")"
UPSTREAM_ROOT="$PIXARC_ROOT/third-party/$MODEL_FAMILY"
[[ -f "$SCRIPT_ROOT/generate_shard.py" ]] || { echo "Missing $SCRIPT_ROOT/generate_shard.py" >&2; exit 2; }
if [[ -e "$OUTPUT_ROOT" && -n "$(find "$OUTPUT_ROOT" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing non-empty smoke output root: $OUTPUT_ROOT" >&2
  exit 4
fi
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd -- "$OUTPUT_ROOT" && pwd -P)"
export PYTHONPATH="$UPSTREAM_ROOT:$BASELINE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python "$SCRIPT_ROOT/deferred_run_guard.py" \
  --config "$FULL_CONFIG" --manifest "$MANIFEST" --max-records 8
python "$SCRIPT_ROOT/deferred_run_guard.py" \
  --config "$CANDIDATE_CONFIG" --manifest "$MANIFEST" --max-records 8

SPECA_INVOCATION_ID="smoke-full-$(date +%s%N)" \
python "$SCRIPT_ROOT/generate_shard.py" \
  --config "$FULL_CONFIG" --config-origin-dir "$(cd -- "$(dirname -- "$FULL_CONFIG")" && pwd -P)" \
  --manifest "$MANIFEST" --shard-id 0 --world-size 1 \
  --output-root "$OUTPUT_ROOT/full" --acknowledge-gpu-job
SPECA_INVOCATION_ID="smoke-candidate-$(date +%s%N)" \
python "$SCRIPT_ROOT/generate_shard.py" \
  --config "$CANDIDATE_CONFIG" --config-origin-dir "$(cd -- "$(dirname -- "$CANDIDATE_CONFIG")" && pwd -P)" \
  --manifest "$MANIFEST" --shard-id 0 --world-size 1 \
  --output-root "$OUTPUT_ROOT/candidate" --acknowledge-gpu-job
