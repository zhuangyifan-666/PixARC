#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --config FILE --manifest FILE --output-root DIR [--resume]" >&2
}

CONFIG=""
MANIFEST=""
OUTPUT_ROOT=""
RESUME=0
while (($#)); do
  case "$1" in
    --config) [[ $# -ge 2 ]] || { usage; exit 2; }; CONFIG="$2"; shift 2 ;;
    --manifest) [[ $# -ge 2 ]] || { usage; exit 2; }; MANIFEST="$2"; shift 2 ;;
    --output-root) [[ $# -ge 2 ]] || { usage; exit 2; }; OUTPUT_ROOT="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    *) usage; exit 2 ;;
  esac
done
[[ -f "$CONFIG" && -f "$MANIFEST" && -n "$OUTPUT_ROOT" ]] || { usage; exit 2; }
CONFIG_ORIGIN_DIR="$(cd -- "$(dirname -- "$CONFIG")" && pwd -P)"
if [[ "${SEACACHE_GPU_TESTS_ALLOWED:-0}" != "1" ]]; then
  echo "Refusing GPU generation. Set SEACACHE_GPU_TESTS_ALLOWED=1 only after all target GPUs are idle." >&2
  exit 2
fi

[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || {
  echo "Refusing launch: CUDA_VISIBLE_DEVICES must name exactly four allocated GPUs." >&2
  exit 2
}
IFS=',' read -r -a GPU_IDS <<<"$CUDA_VISIBLE_DEVICES"
if [[ ${#GPU_IDS[@]} -ne 4 ]]; then
  echo "Refusing launch: CUDA_VISIBLE_DEVICES must contain exactly four entries." >&2
  exit 2
fi
for ((I=0; I<4; I++)); do
  GPU_IDS[$I]="${GPU_IDS[$I]//[[:space:]]/}"
  [[ -n "${GPU_IDS[$I]}" ]] || {
    echo "Refusing launch: CUDA_VISIBLE_DEVICES contains an empty entry." >&2
    exit 2
  }
  for ((J=0; J<I; J++)); do
    if [[ "${GPU_IDS[$I]}" == "${GPU_IDS[$J]}" ]]; then
      echo "Refusing launch: CUDA_VISIBLE_DEVICES entries must be unique." >&2
      exit 2
    fi
  done
done

# Refuse on either visible compute PIDs or material utilization/memory.  This
# second check is conservative when container PID namespaces hide process rows.
BUSY_PIDS=""
BUSY_GPU=""
PHYSICAL_GPU_UUIDS=()
for DEVICE in "${GPU_IDS[@]}"; do
  if ! DEVICE_PIDS_RAW="$(nvidia-smi -i "$DEVICE" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
    echo "Refusing launch: nvidia-smi process query failed for GPU $DEVICE." >&2
    exit 3
  fi
  if ! DEVICE_STATE_RAW="$(nvidia-smi -i "$DEVICE" --query-gpu=uuid,index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null)"; then
    echo "Refusing launch: nvidia-smi state query failed for GPU $DEVICE." >&2
    exit 3
  fi
  if ! awk -F',' 'NR == 1 {for (i=1; i<=4; i++) gsub(/[[:space:]]/, "", $i); valid=(NF == 4 && $1 != "" && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+([.][0-9]+)?$/ && $4 ~ /^[0-9]+([.][0-9]+)?$/)} END {exit !(NR == 1 && valid)}' <<<"$DEVICE_STATE_RAW"; then
    echo "Refusing launch: nvidia-smi returned missing or non-numeric GPU state for $DEVICE." >&2
    exit 3
  fi
  DEVICE_PIDS="$(sed '/^[[:space:]]*$/d' <<<"$DEVICE_PIDS_RAW")"
  DEVICE_UUID="$(awk -F',' 'NR == 1 {gsub(/[[:space:]]/, "", $1); print $1}' <<<"$DEVICE_STATE_RAW")"
  [[ -n "$DEVICE_UUID" ]] || {
    echo "Refusing launch: nvidia-smi returned no UUID for GPU $DEVICE." >&2
    exit 3
  }
  for EXISTING_UUID in "${PHYSICAL_GPU_UUIDS[@]}"; do
    if [[ "$DEVICE_UUID" == "$EXISTING_UUID" ]]; then
      echo "Refusing launch: CUDA_VISIBLE_DEVICES aliases the same physical GPU twice." >&2
      exit 3
    fi
  done
  PHYSICAL_GPU_UUIDS+=("$DEVICE_UUID")
  DEVICE_STATE="$(awk -F',' '$3+0 > 5 || $4+0 > 1024 {print $0}' <<<"$DEVICE_STATE_RAW")"
  [[ -z "$DEVICE_PIDS" ]] || BUSY_PIDS+="$DEVICE:$DEVICE_PIDS "
  [[ -z "$DEVICE_STATE" ]] || BUSY_GPU+="$DEVICE_STATE"$'\n'
done
if [[ -n "$BUSY_PIDS" || -n "$BUSY_GPU" ]]; then
  echo "Refusing to start: at least one target GPU appears busy." >&2
  [[ -n "$BUSY_GPU" ]] && echo "$BUSY_GPU" >&2
  exit 3
fi

if [[ -e "$OUTPUT_ROOT" && $RESUME -eq 0 ]]; then
  if find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "Refusing non-empty output root: $OUTPUT_ROOT" >&2
    exit 4
  fi
fi
mkdir -p "$OUTPUT_ROOT"
LOCK_DIR="$OUTPUT_ROOT/.seacache-launch.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Refusing launch: another seacache-style launcher owns $LOCK_DIR." >&2
  exit 4
fi
GPU_LOCK_DIRS=()
PIDS=()
INTERRUPTED=0
cleanup_lock() {
  for GPU_LOCK_DIR in "${GPU_LOCK_DIRS[@]}"; do
    rmdir "$GPU_LOCK_DIR" 2>/dev/null || true
  done
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
wait_on_signal() {
  trap '' INT TERM HUP
  if [[ ${#PIDS[@]} -eq 0 ]]; then
    echo "Launcher interrupted before any rank started; no process was signalled." >&2
    exit 130
  fi
  INTERRUPTED=1
  echo "Launcher interrupted; not signalling owned ranks. It will retain locks, wait, and write an interrupted wall-clock record." >&2
}
trap wait_on_signal INT TERM HUP
trap cleanup_lock EXIT
GPU_LOCK_ROOT="/tmp/pixarc-seacache-gpu-locks"
mkdir -p "$GPU_LOCK_ROOT"
for DEVICE_UUID in "${PHYSICAL_GPU_UUIDS[@]}"; do
  [[ "$DEVICE_UUID" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "Refusing launch: unsafe GPU UUID returned by nvidia-smi." >&2
    exit 4
  }
  GPU_LOCK_DIR="$GPU_LOCK_ROOT/$DEVICE_UUID.lock"
  if ! mkdir "$GPU_LOCK_DIR" 2>/dev/null; then
    echo "Refusing launch: GPU coordination lock exists: $GPU_LOCK_DIR" >&2
    echo "Verify no seacache-style rank is active before removing a stale lock." >&2
    exit 4
  fi
  GPU_LOCK_DIRS+=("$GPU_LOCK_DIR")
done
mkdir -p "$OUTPUT_ROOT"/{samples,metadata,summaries,logs}
ARCHIVED_CONFIG="$OUTPUT_ROOT/config_resolved.yaml"
ARCHIVED_MANIFEST="$OUTPUT_ROOT/input_manifest.jsonl"
SNAPSHOT_SCRIPT="$(dirname "$0")/snapshot_input.py"
HAS_DURABLE_OUTPUT=0
if [[ -f "$OUTPUT_ROOT/run_manifest.json" ]] \
  || find "$OUTPUT_ROOT/samples" -maxdepth 1 -type f -name '*.png' -print -quit | grep -q . \
  || find "$OUTPUT_ROOT/metadata" -maxdepth 1 -type f -name 'rank_*.jsonl' -print -quit | grep -q .; then
  HAS_DURABLE_OUTPUT=1
fi
if [[ $RESUME -eq 1 ]]; then
  if [[ $HAS_DURABLE_OUTPUT -eq 1 ]] \
    && [[ ! -f "$ARCHIVED_CONFIG" || ! -f "$ARCHIVED_MANIFEST" ]]; then
    echo "Refusing resume: durable output exists but an archived input is missing." >&2
    exit 5
  fi
  [[ -f "$ARCHIVED_CONFIG" ]] || python "$SNAPSHOT_SCRIPT" "$CONFIG" "$ARCHIVED_CONFIG"
  [[ -f "$ARCHIVED_MANIFEST" ]] || python "$SNAPSHOT_SCRIPT" "$MANIFEST" "$ARCHIVED_MANIFEST"
else
  python "$SNAPSHOT_SCRIPT" "$CONFIG" "$ARCHIVED_CONFIG"
  python "$SNAPSHOT_SCRIPT" "$MANIFEST" "$ARCHIVED_MANIFEST"
fi
cmp -s "$CONFIG" "$ARCHIVED_CONFIG" || {
  echo "Refusing launch: config differs from archived config_resolved.yaml." >&2
  exit 5
}
cmp -s "$MANIFEST" "$ARCHIVED_MANIFEST" || {
  echo "Refusing launch: manifest differs from archived input_manifest.jsonl." >&2
  exit 5
}

export OUTPUT_ROOT
BASELINE_GENERATED="$(python -c 'import json, os; from pathlib import Path; root=Path(os.environ["OUTPUT_ROOT"])/"metadata"; paths=[root/f"rank_{rank}.jsonl" for rank in range(4)]; ids=[int(json.loads(line)["sample_id"]) for path in paths if path.is_file() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]; assert len(ids) == len(set(ids)), "duplicate sample IDs in baseline metadata"; print(len(ids))')"
START_NS="$(date +%s%N)"
for RANK in 0 1 2 3; do
  [[ $INTERRUPTED -eq 0 ]] || break
  EXTRA=()
  [[ $RESUME -eq 1 ]] && EXTRA+=(--resume)
  SEACACHE_INVOCATION_ID="$START_NS" CUDA_VISIBLE_DEVICES="${GPU_IDS[$RANK]}" \
    python "$(dirname "$0")/generate_shard.py" \
    --config "$ARCHIVED_CONFIG" --config-origin-dir "$CONFIG_ORIGIN_DIR" \
    --manifest "$ARCHIVED_MANIFEST" --shard-id "$RANK" \
    --world-size 4 --output-root "$OUTPUT_ROOT" --acknowledge-gpu-job \
    "${EXTRA[@]}" >>"$OUTPUT_ROOT/logs/rank_${RANK}.log" 2>&1 & PIDS+=("$!")
done

STATUS=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then STATUS=1; fi
done
if [[ $INTERRUPTED -eq 1 ]]; then
  for PID in "${PIDS[@]}"; do wait "$PID" 2>/dev/null || true; done
  STATUS=130
fi
END_NS="$(date +%s%N)"
export START_NS END_NS STATUS OUTPUT_ROOT BASELINE_GENERATED ARCHIVED_MANIFEST
PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH:-}" python -c 'import json, os; from pathlib import Path; from seacache_style.manifest import load_manifest; from seacache_style.metadata import atomic_write_json; root=Path(os.environ["OUTPUT_ROOT"]); start_ns=os.environ["START_NS"]; elapsed=(int(os.environ["END_NS"])-int(start_ns))/1e9; expected=len(load_manifest(os.environ["ARCHIVED_MANIFEST"])); summary_paths=[root/"summaries"/f"rank_{rank}_summary.json" for rank in range(4)]; loaded=[(rank, json.loads(path.read_text(encoding="utf-8"))) for rank, path in enumerate(summary_paths) if path.is_file()]; current=[(rank, item) for rank, item in loaded if str(item.get("invocation_id")) == start_ns]; missing=sorted(set(range(4))-{rank for rank, item in current}); summaries=[item for rank, item in current]; metadata_paths=[root/"metadata"/f"rank_{rank}.jsonl" for rank in range(4)]; ids=[int(json.loads(line)["sample_id"]) for path in metadata_paths if path.is_file() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]; duplicate_count=len(ids)-len(set(ids)); cumulative=len(set(ids)); baseline=int(os.environ["BASELINE_GENERATED"]); invocation=max(0, cumulative-baseline); reported=sum(int(item["generated_this_invocation"]) for item in summaries); status=int(os.environ["STATUS"]); valid=status == 0 and not missing and duplicate_count == 0 and cumulative == expected and reported == invocation; payload={"invocation_id": start_ns, "launcher_status": status, "interrupted": status == 130, "elapsed_seconds": elapsed, "manifest_sample_count": expected, "baseline_sample_count": baseline, "invocation_sample_count": invocation, "reported_invocation_sample_count": reported, "cumulative_sample_count": cumulative, "duplicate_metadata_sample_ids": duplicate_count, "missing_summary_ranks": missing, "completed": valid, "images_per_second": invocation / elapsed if valid and elapsed > 0 else None, "throughput_scope": "current launcher invocation only", "world_size": 4}; atomic_write_json(root/f"four_gpu_wall_clock_{start_ns}.json", payload); atomic_write_json(root/"four_gpu_wall_clock.json", payload); raise SystemExit(0 if status != 0 or valid else 1)'
exit "$STATUS"
