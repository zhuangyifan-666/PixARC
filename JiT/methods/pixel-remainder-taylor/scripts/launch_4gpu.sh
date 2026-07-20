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
[[ -f "$CONFIG" && -f "$MANIFEST" && -f "${MANIFEST}.meta.json" ]] || { usage; exit 2; }
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
if [[ "$MODEL" == "JiT" ]]; then
  GENERATOR="$SCRIPT_ROOT/generate_shard.py"
else
  GENERATOR="$PIXARC_ROOT/PixelGen/methods/pixel-remainder-taylor/scripts/generate_shard.py"
fi
VALIDATOR="$SCRIPT_ROOT/validate_outputs.py"
[[ -f "$GENERATOR" && -f "$VALIDATOR" ]] || { echo "Required generator/validator is missing." >&2; exit 4; }

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
START_NS="$(date +%s%N)"
INVOCATION="$(date -u +%Y%m%dT%H%M%SZ)-$$"
PIDS=()
for RANK in 0 1 2 3; do
  EXTRA=()
  [[ $RESUME -eq 0 ]] || EXTRA+=(--resume)
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$RANK]}" \
  PIXEL_REMAINDER_INVOCATION_ID="$INVOCATION" \
  python "$GENERATOR" \
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
[[ $STATUS -eq 0 ]] || { echo "At least one rank failed; inspect $OUTPUT_ROOT/logs" >&2; exit 5; }
END_NS="$(date +%s%N)"
ELAPSED="$(python -c 'import sys; print((int(sys.argv[2])-int(sys.argv[1]))/1e9)' "$START_NS" "$END_NS")"
python "$VALIDATOR" \
  --run-root "$OUTPUT_ROOT" --manifest "$MANIFEST" \
  --expected-count "$EXPECTED_COUNT" --resolution 256
python -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); p.write_text(json.dumps({"elapsed_seconds":float(sys.argv[2]),"invocation_id":sys.argv[3]},indent=2)+"\n")' \
  "$OUTPUT_ROOT/launcher_timing.json" "$ELAPSED" "$INVOCATION"
echo "Completed $MODEL: $EXPECTED_COUNT samples in $ELAPSED seconds"
