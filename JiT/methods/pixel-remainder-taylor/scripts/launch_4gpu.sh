#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --model JiT|PixelGen --config FILE --manifest FILE --output-root DIR --expected-count N [--resume]" >&2
}

MODEL=""; CONFIG=""; MANIFEST=""; OUTPUT_ROOT=""; EXPECTED_COUNT=""; RESUME=0
while (($#)); do
  case "$1" in
    --model) [[ $# -ge 2 ]] || { usage; exit 2; }; MODEL="$2"; shift 2 ;;
    --config) [[ $# -ge 2 ]] || { usage; exit 2; }; CONFIG="$2"; shift 2 ;;
    --manifest) [[ $# -ge 2 ]] || { usage; exit 2; }; MANIFEST="$2"; shift 2 ;;
    --output-root) [[ $# -ge 2 ]] || { usage; exit 2; }; OUTPUT_ROOT="$2"; shift 2 ;;
    --expected-count) [[ $# -ge 2 ]] || { usage; exit 2; }; EXPECTED_COUNT="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    *) usage; exit 2 ;;
  esac
done
[[ "$MODEL" == "JiT" || "$MODEL" == "PixelGen" ]] || { usage; exit 2; }
[[ -f "$CONFIG" && -f "$MANIFEST" && -n "$OUTPUT_ROOT" ]] || { usage; exit 2; }
[[ "$EXPECTED_COUNT" =~ ^[1-9][0-9]*$ ]] || { usage; exit 2; }
[[ "${PIXEL_REMAINDER_GPU_RUN_ALLOWED:-0}" == "1" ]] || {
  echo "Refusing GPU work: set PIXEL_REMAINDER_GPU_RUN_ALLOWED=1 after allocating idle GPUs." >&2
  exit 3
}
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || {
  echo "CUDA_VISIBLE_DEVICES must list exactly four allocated GPUs." >&2
  exit 3
}

SCRIPT_ROOT="$(cd -- "$(dirname -- "$0")" && pwd -P)"
PIXARC_ROOT="$(cd -- "$SCRIPT_ROOT/../../../.." && pwd -P)"
PYTHON_BIN="${PIXEL_REMAINDER_PYTHON:-python}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 4
}
PROTOCOL_CLI="$SCRIPT_ROOT/protocol_cli.py"
SNAPSHOT_SCRIPT="$SCRIPT_ROOT/snapshot_input.py"
TIMING_SCRIPT="$SCRIPT_ROOT/launcher_timing.py"
VALIDATOR="$SCRIPT_ROOT/validate_outputs.py"
if [[ "$MODEL" == "JiT" ]]; then
  GENERATOR="$SCRIPT_ROOT/generate_shard.py"
else
  GENERATOR="$PIXARC_ROOT/PixelGen/methods/pixel-remainder-taylor/scripts/generate_shard.py"
fi
for REQUIRED in "$PROTOCOL_CLI" "$SNAPSHOT_SCRIPT" "$TIMING_SCRIPT" "$VALIDATOR" "$GENERATOR"; do
  [[ -f "$REQUIRED" ]] || { echo "Missing launcher helper: $REQUIRED" >&2; exit 4; }
done

if [[ -n "$(git -C "$PIXARC_ROOT" status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing production launch: executable worktree is not clean." >&2
  exit 4
fi

CONFIG_ORIGIN="$(cd -- "$(dirname -- "$CONFIG")" && pwd -P)"
CONFIG="$CONFIG_ORIGIN/$(basename -- "$CONFIG")"
MANIFEST="$(cd -- "$(dirname -- "$MANIFEST")" && pwd -P)/$(basename -- "$MANIFEST")"
MANIFEST_SIDECAR="$($PYTHON_BIN "$PROTOCOL_CLI" sidecar "$MANIFEST")"
"$PYTHON_BIN" "$PROTOCOL_CLI" validate-count \
  --manifest "$MANIFEST" --expected-count "$EXPECTED_COUNT" >/dev/null

IFS=',' read -r -a GPU_IDS <<<"$CUDA_VISIBLE_DEVICES"
[[ ${#GPU_IDS[@]} -eq 4 ]] || { echo "Exactly four GPU identifiers are required." >&2; exit 3; }
PHYSICAL_GPU_UUIDS=()
for ((I=0; I<4; I++)); do
  GPU_IDS[$I]="${GPU_IDS[$I]//[[:space:]]/}"
  [[ -n "${GPU_IDS[$I]}" ]] || { echo "Empty GPU identifier." >&2; exit 3; }
  for ((J=0; J<I; J++)); do
    [[ "${GPU_IDS[$I]}" != "${GPU_IDS[$J]}" ]] || {
      echo "CUDA_VISIBLE_DEVICES entries must be unique." >&2
      exit 3
    }
  done
  if ! GPU_PIDS="$(nvidia-smi -i "${GPU_IDS[$I]}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
    echo "nvidia-smi process query failed for ${GPU_IDS[$I]}." >&2
    exit 3
  fi
  if ! GPU_STATE="$(nvidia-smi -i "${GPU_IDS[$I]}" --query-gpu=uuid,index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null)"; then
    echo "nvidia-smi state query failed for ${GPU_IDS[$I]}." >&2
    exit 3
  fi
  if ! awk -F',' 'NR == 1 {for (i=1; i<=4; i++) gsub(/[[:space:]]/, "", $i); ok=(NF == 4 && $1 != "" && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+([.][0-9]+)?$/ && $4 ~ /^[0-9]+([.][0-9]+)?$/)} END {exit !(NR == 1 && ok)}' <<<"$GPU_STATE"; then
    echo "Invalid nvidia-smi state for ${GPU_IDS[$I]}." >&2
    exit 3
  fi
  GPU_UUID="$(awk -F',' 'NR == 1 {gsub(/[[:space:]]/, "", $1); print $1}' <<<"$GPU_STATE")"
  for EXISTING_UUID in "${PHYSICAL_GPU_UUIDS[@]}"; do
    [[ "$GPU_UUID" != "$EXISTING_UUID" ]] || {
      echo "CUDA_VISIBLE_DEVICES aliases one physical GPU twice." >&2
      exit 3
    }
  done
  PHYSICAL_GPU_UUIDS+=("$GPU_UUID")
  [[ -z "${GPU_PIDS//[[:space:]]/}" ]] || {
    echo "GPU ${GPU_IDS[$I]} has an active compute process." >&2
    exit 3
  }
  if awk -F',' '$3+0 > 5 || $4+0 > 1024 {busy=1} END {exit !busy}' <<<"$GPU_STATE"; then
    echo "GPU ${GPU_IDS[$I]} exceeds idle utilization/memory thresholds: $GPU_STATE" >&2
    exit 3
  fi
done

OUTPUT_WAS_NONEMPTY=0
if [[ -e "$OUTPUT_ROOT" ]] && [[ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  OUTPUT_WAS_NONEMPTY=1
fi
[[ $OUTPUT_WAS_NONEMPTY -eq 0 || $RESUME -eq 1 ]] || {
  echo "Refusing a non-empty output root without --resume: $OUTPUT_ROOT" >&2
  exit 4
}
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd -- "$OUTPUT_ROOT" && pwd -P)"
case "$OUTPUT_ROOT/" in
  "$PIXARC_ROOT/"*) echo "Generation output must be outside the PixARC checkout." >&2; exit 4 ;;
esac

OUTPUT_LOCK="$OUTPUT_ROOT/.pixel-remainder-launch.lock"
mkdir "$OUTPUT_LOCK" 2>/dev/null || { echo "Another launcher owns $OUTPUT_LOCK" >&2; exit 4; }
GPU_LOCK_ROOT="/tmp/pixarc-pixel-remainder-gpu-locks"
mkdir -p "$GPU_LOCK_ROOT"
GPU_LOCKS=(); PIDS=(); INTERRUPTED=0
cleanup_locks() {
  for LOCK in "${GPU_LOCKS[@]}"; do rmdir "$LOCK" 2>/dev/null || true; done
  rmdir "$OUTPUT_LOCK" 2>/dev/null || true
}
on_signal() {
  trap '' INT TERM HUP
  INTERRUPTED=1
  echo "Launcher interrupted; owned ranks are not signalled and timing will be recorded incomplete." >&2
}
trap on_signal INT TERM HUP
trap cleanup_locks EXIT
for GPU_UUID in "${PHYSICAL_GPU_UUIDS[@]}"; do
  [[ "$GPU_UUID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Unsafe GPU UUID." >&2; exit 4; }
  LOCK="$GPU_LOCK_ROOT/$GPU_UUID.lock"
  mkdir "$LOCK" 2>/dev/null || { echo "GPU coordination lock exists: $LOCK" >&2; exit 4; }
  GPU_LOCKS+=("$LOCK")
done

mkdir -p "$OUTPUT_ROOT"/{samples,metadata,summaries,traces,logs}
ARCHIVED_CONFIG="$OUTPUT_ROOT/config_resolved.yaml"
ARCHIVED_MANIFEST="$OUTPUT_ROOT/input_manifest.jsonl"
ARCHIVED_SIDECAR="$OUTPUT_ROOT/input_manifest.jsonl.meta.json"
"$PYTHON_BIN" "$SNAPSHOT_SCRIPT" "$CONFIG" "$ARCHIVED_CONFIG"
"$PYTHON_BIN" "$SNAPSHOT_SCRIPT" "$MANIFEST" "$ARCHIVED_MANIFEST"
"$PYTHON_BIN" "$SNAPSHOT_SCRIPT" "$MANIFEST_SIDECAR" "$ARCHIVED_SIDECAR"
cmp -s "$CONFIG" "$ARCHIVED_CONFIG" || { echo "Config snapshot mismatch." >&2; exit 5; }
cmp -s "$MANIFEST" "$ARCHIVED_MANIFEST" || { echo "Manifest snapshot mismatch." >&2; exit 5; }
cmp -s "$MANIFEST_SIDECAR" "$ARCHIVED_SIDECAR" || { echo "Sidecar snapshot mismatch." >&2; exit 5; }

BASELINE_COUNT="$($PYTHON_BIN "$TIMING_SCRIPT" count --output-root "$OUTPUT_ROOT" --world-size 4)"
START_NS="$(date +%s%N)"
INVOCATION="$(date -u +%Y%m%dT%H%M%SZ)-$$"
for RANK in 0 1 2 3; do
  [[ $INTERRUPTED -eq 0 ]] || break
  EXTRA=(); [[ $RESUME -eq 0 ]] || EXTRA+=(--resume)
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$RANK]}" \
  PIXEL_REMAINDER_INVOCATION_ID="$INVOCATION" \
  "$PYTHON_BIN" "$GENERATOR" \
    --config "$ARCHIVED_CONFIG" --config-origin-dir "$CONFIG_ORIGIN" \
    --manifest "$ARCHIVED_MANIFEST" --shard-id "$RANK" --world-size 4 \
    --output-root "$OUTPUT_ROOT" --acknowledge-gpu-job "${EXTRA[@]}" \
    >"$OUTPUT_ROOT/logs/rank_${RANK}_${INVOCATION}.log" 2>&1 &
  PIDS+=("$!")
done
STATUS=0
for PID in "${PIDS[@]}"; do wait "$PID" || STATUS=1; done
[[ $INTERRUPTED -eq 0 ]] || STATUS=130
END_NS="$(date +%s%N)"
if [[ $STATUS -eq 0 ]]; then
  "$PYTHON_BIN" "$VALIDATOR" --run-root "$OUTPUT_ROOT" \
    --manifest "$ARCHIVED_MANIFEST" --expected-count "$EXPECTED_COUNT" \
    --resolution 256 || STATUS=1
fi
"$PYTHON_BIN" "$TIMING_SCRIPT" record \
  --output-root "$OUTPUT_ROOT" --manifest "$ARCHIVED_MANIFEST" \
  --invocation-id "$INVOCATION" --start-ns "$START_NS" --end-ns "$END_NS" \
  --launcher-status "$STATUS" --baseline-count "$BASELINE_COUNT" --world-size 4
[[ $STATUS -eq 0 ]] || { echo "Launcher or validation failed; inspect $OUTPUT_ROOT/logs" >&2; exit "$STATUS"; }
ELAPSED="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["cumulative_elapsed_seconds"])' "$OUTPUT_ROOT/launcher_timing.json")"
echo "Completed $MODEL: $EXPECTED_COUNT samples in cumulative $ELAPSED seconds"
