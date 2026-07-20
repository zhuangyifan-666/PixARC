#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --model JiT|PixelGen --config FILE --manifest FILE --output-root DIR --expected-count N [--resume]" >&2
}

MODEL=""; CONFIG=""; MANIFEST=""; OUTPUT_ROOT=""; EXPECTED_COUNT=""; RESUME=0
while (($#)); do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --expected-count) EXPECTED_COUNT="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    *) usage; exit 2 ;;
  esac
done
[[ "$MODEL" == "JiT" || "$MODEL" == "PixelGen" ]] || { usage; exit 2; }
[[ -f "$CONFIG" && -f "$MANIFEST" ]] || { usage; exit 2; }
[[ "$EXPECTED_COUNT" =~ ^[1-9][0-9]*$ ]] || { usage; exit 2; }
[[ "${PIXEL_REMAINDER_GPU_RUN_ALLOWED:-0}" == "1" ]] || {
  echo "Refusing GPU work: set PIXEL_REMAINDER_GPU_RUN_ALLOWED=1 after allocating idle GPUs." >&2
  exit 3
}
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "CUDA_VISIBLE_DEVICES must list four allocated GPUs." >&2; exit 3; }
IFS=',' read -r -a GPU_IDS <<<"$CUDA_VISIBLE_DEVICES"
[[ ${#GPU_IDS[@]} -eq 4 ]] || { echo "Exactly four visible GPU identifiers are required." >&2; exit 3; }
for ((I=0; I<4; I++)); do
  GPU_IDS[$I]="${GPU_IDS[$I]//[[:space:]]/}"
  [[ -n "${GPU_IDS[$I]}" ]] || exit 3
  for ((J=0; J<I; J++)); do
    [[ "${GPU_IDS[$I]}" != "${GPU_IDS[$J]}" ]] || { echo "GPU identifiers must be unique." >&2; exit 3; }
  done
  PIDS_ON_GPU="$(nvidia-smi -i "${GPU_IDS[$I]}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || exit 3)"
  [[ -z "${PIDS_ON_GPU//[[:space:]]/}" ]] || { echo "GPU ${GPU_IDS[$I]} already has a compute process." >&2; exit 3; }
done

SCRIPT_ROOT="$(cd -- "$(dirname -- "$0")" && pwd -P)"
PIXARC_ROOT="$(cd -- "$SCRIPT_ROOT/../../../.." && pwd -P)"
PYTHON_BIN="${PIXEL_REMAINDER_PYTHON:-python}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 4; }
if [[ "$MODEL" == "JiT" ]]; then
  GENERATOR="$SCRIPT_ROOT/generate_shard.py"
else
  GENERATOR="$PIXARC_ROOT/PixelGen/methods/pixel-remainder-taylor/scripts/generate_shard.py"
fi
VALIDATOR="$SCRIPT_ROOT/validate_outputs.py"
TIMING_SCRIPT="$SCRIPT_ROOT/launcher_timing.py"
[[ -f "$GENERATOR" && -f "$VALIDATOR" && -f "$TIMING_SCRIPT" ]] || { echo "Required generator/validator/timing script is missing." >&2; exit 4; }

if [[ -e "$OUTPUT_ROOT" && -n "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" && $RESUME -eq 0 ]]; then
  echo "Refusing a non-empty output root without --resume: $OUTPUT_ROOT" >&2
  exit 4
fi
mkdir -p "$OUTPUT_ROOT/logs"
OUTPUT_ROOT="$(cd -- "$OUTPUT_ROOT" && pwd -P)"
LOCK_DIR="$OUTPUT_ROOT/.pixel-remainder-launch.lock"
mkdir "$LOCK_DIR" 2>/dev/null || { echo "Another launcher owns $LOCK_DIR" >&2; exit 4; }
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

CONFIG="$(cd -- "$(dirname -- "$CONFIG")" && pwd -P)/$(basename -- "$CONFIG")"
MANIFEST="$(cd -- "$(dirname -- "$MANIFEST")" && pwd -P)/$(basename -- "$MANIFEST")"
CONFIG_ORIGIN="$(dirname -- "$CONFIG")"
BASELINE_COUNT="$("$PYTHON_BIN" "$TIMING_SCRIPT" count --output-root "$OUTPUT_ROOT" --world-size 4)"
START_NS="$(date +%s%N)"
INVOCATION="$(date -u +%Y%m%dT%H%M%SZ)-$$"
PIDS=()
for RANK in 0 1 2 3; do
  EXTRA=()
  [[ $RESUME -eq 0 ]] || EXTRA+=(--resume)
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$RANK]}" \
  PIXEL_REMAINDER_INVOCATION_ID="$INVOCATION" \
  "$PYTHON_BIN" "$GENERATOR" \
    --config "$CONFIG" --config-origin-dir "$CONFIG_ORIGIN" \
    --manifest "$MANIFEST" --shard-id "$RANK" --world-size 4 \
    --output-root "$OUTPUT_ROOT" --acknowledge-gpu-job "${EXTRA[@]}" \
    >"$OUTPUT_ROOT/logs/rank_${RANK}_${INVOCATION}.log" 2>&1 &
  PIDS+=("$!")
done
STATUS=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || STATUS=1
done
END_NS="$(date +%s%N)"
if [[ $STATUS -eq 0 ]]; then
  "$PYTHON_BIN" "$VALIDATOR" \
    --run-root "$OUTPUT_ROOT" --manifest "$MANIFEST" \
    --expected-count "$EXPECTED_COUNT" --resolution 256 || STATUS=1
fi
"$PYTHON_BIN" "$TIMING_SCRIPT" record \
  --output-root "$OUTPUT_ROOT" --manifest "$MANIFEST" \
  --invocation-id "$INVOCATION" --start-ns "$START_NS" --end-ns "$END_NS" \
  --launcher-status "$STATUS" --baseline-count "$BASELINE_COUNT" --world-size 4
[[ $STATUS -eq 0 ]] || { echo "Launcher or validation failed; inspect $OUTPUT_ROOT/logs" >&2; exit 5; }
ELAPSED="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cumulative_elapsed_seconds"])' "$OUTPUT_ROOT/launcher_timing.json")"
echo "Completed $MODEL: $EXPECTED_COUNT samples in cumulative $ELAPSED seconds"
