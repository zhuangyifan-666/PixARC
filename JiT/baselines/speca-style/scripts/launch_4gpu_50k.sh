#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --config FILE --manifest FILE --output-root DIR --gpu-ids ID0,ID1,ID2,ID3 [--resume] [--nonfinal-proxy --expected-records N --expected-per-class N --expected-num-classes N]" >&2
}

CONFIG=""
MANIFEST=""
OUTPUT_ROOT=""
GPU_IDS_CSV=""
RESUME=0
NONFINAL_PROXY=0
EXPECTED_RECORDS=""
EXPECTED_PER_CLASS=""
EXPECTED_NUM_CLASSES=""
while (($#)); do
  case "$1" in
    --config) [[ $# -ge 2 ]] || { usage; exit 2; }; CONFIG="$2"; shift 2 ;;
    --manifest) [[ $# -ge 2 ]] || { usage; exit 2; }; MANIFEST="$2"; shift 2 ;;
    --output-root) [[ $# -ge 2 ]] || { usage; exit 2; }; OUTPUT_ROOT="$2"; shift 2 ;;
    --gpu-ids) [[ $# -ge 2 ]] || { usage; exit 2; }; GPU_IDS_CSV="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --nonfinal-proxy) NONFINAL_PROXY=1; shift ;;
    --expected-records) [[ $# -ge 2 ]] || { usage; exit 2; }; EXPECTED_RECORDS="$2"; shift 2 ;;
    --expected-per-class) [[ $# -ge 2 ]] || { usage; exit 2; }; EXPECTED_PER_CLASS="$2"; shift 2 ;;
    --expected-num-classes) [[ $# -ge 2 ]] || { usage; exit 2; }; EXPECTED_NUM_CLASSES="$2"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -f "$CONFIG" && -f "$MANIFEST" && -n "$OUTPUT_ROOT" && -n "$GPU_IDS_CSV" ]] || { usage; exit 2; }

if [[ $NONFINAL_PROXY -eq 0 ]]; then
  if [[ -n "$EXPECTED_RECORDS" || -n "$EXPECTED_PER_CLASS" || -n "$EXPECTED_NUM_CLASSES" ]]; then
    echo "Refusing custom counts without --nonfinal-proxy; the default path is exactly 50K." >&2
    exit 2
  fi
  EXPECTED_RECORDS=50000
  EXPECTED_PER_CLASS=50
  EXPECTED_NUM_CLASSES=1000
else
  if [[ -z "$EXPECTED_RECORDS" || -z "$EXPECTED_PER_CLASS" || -z "$EXPECTED_NUM_CLASSES" ]]; then
    echo "Refusing proxy launch: all three --expected-* counts are required." >&2
    exit 2
  fi
  for VALUE in "$EXPECTED_RECORDS" "$EXPECTED_PER_CLASS" "$EXPECTED_NUM_CLASSES"; do
    if [[ ! "$VALUE" =~ ^[1-9][0-9]*$ ]] || ((${#VALUE} > 5)) || ((VALUE > 50000)); then
      echo "Refusing proxy launch: expected counts must be positive decimal integers no greater than 50000." >&2
      exit 2
    fi
  done
  if ((EXPECTED_RECORDS >= 50000)); then
    echo "Refusing proxy launch: non-final proxy runs must contain fewer than 50000 records." >&2
    exit 2
  fi
  if ((EXPECTED_RECORDS != EXPECTED_PER_CLASS * EXPECTED_NUM_CLASSES)); then
    echo "Refusing proxy launch: expected records must equal per-class count times class count." >&2
    exit 2
  fi
  if ((EXPECTED_RECORDS % 4 != 0)); then
    echo "Refusing proxy launch: expected records must divide evenly across four ranks." >&2
    exit 2
  fi
fi

SCRIPT_ROOT="$(cd -- "$(dirname -- "$0")" && pwd -P)"
BASELINE_ROOT="$(cd -- "$SCRIPT_ROOT/.." && pwd -P)"
PIXARC_ROOT="$(cd -- "$BASELINE_ROOT/../../.." && pwd -P)"
MODEL_FAMILY="$(basename -- "$(cd -- "$BASELINE_ROOT/../.." && pwd -P)")"
UPSTREAM_ROOT="$PIXARC_ROOT/third-party/$MODEL_FAMILY"
GENERATE_SHARD="$SCRIPT_ROOT/generate_shard.py"
SNAPSHOT_SCRIPT="$SCRIPT_ROOT/snapshot_input.py"
WALL_CLOCK_SCRIPT="$SCRIPT_ROOT/launcher_wall_clock.py"
GUARD_SCRIPT="$SCRIPT_ROOT/deferred_run_guard.py"
for REQUIRED in "$GENERATE_SHARD" "$SNAPSHOT_SCRIPT" "$WALL_CLOCK_SCRIPT" "$GUARD_SCRIPT"; do
  [[ -f "$REQUIRED" ]] || { echo "Missing required launcher helper: $REQUIRED" >&2; exit 2; }
done
CONFIG_ORIGIN_DIR="$(cd -- "$(dirname -- "$CONFIG")" && pwd -P)"
CONFIG="$(cd -- "$(dirname -- "$CONFIG")" && pwd -P)/$(basename -- "$CONFIG")"
MANIFEST="$(cd -- "$(dirname -- "$MANIFEST")" && pwd -P)/$(basename -- "$MANIFEST")"
MANIFEST_SIDECAR="${MANIFEST}.meta.json"
[[ -f "$MANIFEST_SIDECAR" ]] || {
  echo "Missing required manifest sidecar: $MANIFEST_SIDECAR" >&2
  exit 2
}

python "$GUARD_SCRIPT" --config "$CONFIG" --manifest "$MANIFEST" \
  --max-records "$EXPECTED_RECORDS" --expected-records "$EXPECTED_RECORDS" \
  --expected-per-class "$EXPECTED_PER_CLASS" --expected-num-classes "$EXPECTED_NUM_CLASSES" \
  --expected-world-size 4 \
  --allowed-mode upstream_full --allowed-mode instrumented_full --allowed-mode speca

if [[ "${SPECA_GPU_TESTS_ALLOWED:-0}" != "1" ]]; then
  echo "Refusing GPU generation. Set SPECA_GPU_TESTS_ALLOWED=1 only after all target GPUs are idle." >&2
  exit 2
fi
IFS=',' read -r -a GPU_IDS <<<"$GPU_IDS_CSV"
if [[ ${#GPU_IDS[@]} -ne 4 ]]; then
  echo "Refusing launch: --gpu-ids must contain exactly four entries." >&2
  exit 2
fi
for ((I=0; I<4; I++)); do
  GPU_IDS[$I]="${GPU_IDS[$I]//[[:space:]]/}"
  [[ -n "${GPU_IDS[$I]}" ]] || {
    echo "Refusing launch: --gpu-ids contains an empty entry." >&2
    exit 2
  }
  for ((J=0; J<I; J++)); do
    if [[ "${GPU_IDS[$I]}" == "${GPU_IDS[$J]}" ]]; then
      echo "Refusing launch: --gpu-ids entries must be unique." >&2
      exit 2
    fi
  done
done

# Fail closed on visible compute PIDs or material utilization/memory. This is
# conservative when a PID namespace hides process rows. It never signals jobs.
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
    echo "Refusing launch: nvidia-smi returned invalid GPU state for $DEVICE." >&2
    exit 3
  fi
  DEVICE_PIDS="$(sed '/^[[:space:]]*$/d' <<<"$DEVICE_PIDS_RAW")"
  DEVICE_UUID="$(awk -F',' 'NR == 1 {gsub(/[[:space:]]/, "", $1); print $1}' <<<"$DEVICE_STATE_RAW")"
  for EXISTING_UUID in "${PHYSICAL_GPU_UUIDS[@]}"; do
    if [[ "$DEVICE_UUID" == "$EXISTING_UUID" ]]; then
      echo "Refusing launch: --gpu-ids aliases one physical GPU twice." >&2
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
  [[ -z "$BUSY_GPU" ]] || echo "$BUSY_GPU" >&2
  exit 3
fi

OUTPUT_WAS_NONEMPTY=0
if [[ -e "$OUTPUT_ROOT" ]] \
  && [[ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  OUTPUT_WAS_NONEMPTY=1
fi
if [[ $OUTPUT_WAS_NONEMPTY -eq 1 && $RESUME -eq 0 ]]; then
  echo "Refusing non-empty output root: $OUTPUT_ROOT" >&2
  exit 4
fi
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd -- "$OUTPUT_ROOT" && pwd -P)"
LOCK_DIR="$OUTPUT_ROOT/.speca-launch.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Refusing launch: another SpeCa launcher owns $LOCK_DIR." >&2
  exit 4
fi

GPU_LOCK_DIRS=()
PIDS=()
INTERRUPTED=0
cleanup_locks() {
  for GPU_LOCK_DIR in "${GPU_LOCK_DIRS[@]}"; do
    rmdir "$GPU_LOCK_DIR" 2>/dev/null || true
  done
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
wait_on_signal() {
  trap '' INT TERM HUP
  if [[ ${#PIDS[@]} -eq 0 ]]; then
    echo "Launcher interrupted before ranks started; no process was signalled." >&2
    exit 130
  fi
  INTERRUPTED=1
  echo "Launcher interrupted; it will not signal owned ranks. Waiting for them before recording status." >&2
}
trap wait_on_signal INT TERM HUP
trap cleanup_locks EXIT

GPU_LOCK_ROOT="/tmp/pixarc-speca-gpu-locks"
mkdir -p "$GPU_LOCK_ROOT"
for DEVICE_UUID in "${PHYSICAL_GPU_UUIDS[@]}"; do
  [[ "$DEVICE_UUID" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "Refusing launch: unsafe GPU UUID returned by nvidia-smi." >&2
    exit 4
  }
  GPU_LOCK_DIR="$GPU_LOCK_ROOT/$DEVICE_UUID.lock"
  if ! mkdir "$GPU_LOCK_DIR" 2>/dev/null; then
    echo "Refusing launch: GPU coordination lock exists: $GPU_LOCK_DIR" >&2
    echo "Verify no SpeCa rank is active before removing a stale lock." >&2
    exit 4
  fi
  GPU_LOCK_DIRS+=("$GPU_LOCK_DIR")
done

mkdir -p "$OUTPUT_ROOT"/{samples,metadata,summaries,logs}
ARCHIVED_CONFIG="$OUTPUT_ROOT/config_resolved.yaml"
ARCHIVED_MANIFEST="$OUTPUT_ROOT/input_manifest.jsonl"
ARCHIVED_MANIFEST_SIDECAR="$OUTPUT_ROOT/input_manifest.jsonl.meta.json"
HAS_DURABLE_OUTPUT=0
if [[ -f "$OUTPUT_ROOT/run_manifest.json" ]] \
  || [[ -n "$(find "$OUTPUT_ROOT/samples" -maxdepth 1 -type f -name '*.png' -print -quit)" ]] \
  || [[ -n "$(find "$OUTPUT_ROOT/metadata" -maxdepth 1 -type f -name 'rank_*.jsonl' -print -quit)" ]]; then
  HAS_DURABLE_OUTPUT=1
fi
if [[ $RESUME -eq 1 ]]; then
  if [[ $OUTPUT_WAS_NONEMPTY -eq 1 && $HAS_DURABLE_OUTPUT -eq 0 ]] \
    && [[ ! -f "$ARCHIVED_CONFIG" || ! -f "$ARCHIVED_MANIFEST" || ! -f "$ARCHIVED_MANIFEST_SIDECAR" ]]; then
    echo "Refusing resume: non-empty output has neither durable samples nor a complete archived input set." >&2
    exit 5
  fi
  if [[ $HAS_DURABLE_OUTPUT -eq 1 ]] \
    && [[ ! -f "$ARCHIVED_CONFIG" || ! -f "$ARCHIVED_MANIFEST" || ! -f "$ARCHIVED_MANIFEST_SIDECAR" ]]; then
    echo "Refusing resume: durable output exists but an archived input is missing." >&2
    exit 5
  fi
  [[ -f "$ARCHIVED_CONFIG" ]] || python "$SNAPSHOT_SCRIPT" "$CONFIG" "$ARCHIVED_CONFIG"
  [[ -f "$ARCHIVED_MANIFEST" ]] || python "$SNAPSHOT_SCRIPT" "$MANIFEST" "$ARCHIVED_MANIFEST"
  [[ -f "$ARCHIVED_MANIFEST_SIDECAR" ]] || python "$SNAPSHOT_SCRIPT" "$MANIFEST_SIDECAR" "$ARCHIVED_MANIFEST_SIDECAR"
else
  python "$SNAPSHOT_SCRIPT" "$CONFIG" "$ARCHIVED_CONFIG"
  python "$SNAPSHOT_SCRIPT" "$MANIFEST" "$ARCHIVED_MANIFEST"
  python "$SNAPSHOT_SCRIPT" "$MANIFEST_SIDECAR" "$ARCHIVED_MANIFEST_SIDECAR"
fi
cmp -s "$CONFIG" "$ARCHIVED_CONFIG" || {
  echo "Refusing launch: config differs from archived config_resolved.yaml." >&2
  exit 5
}
cmp -s "$MANIFEST" "$ARCHIVED_MANIFEST" || {
  echo "Refusing launch: manifest differs from archived input_manifest.jsonl." >&2
  exit 5
}
cmp -s "$MANIFEST_SIDECAR" "$ARCHIVED_MANIFEST_SIDECAR" || {
  echo "Refusing launch: manifest sidecar differs from archived input_manifest.jsonl.meta.json." >&2
  exit 5
}

export OUTPUT_ROOT
BASELINE_GENERATED="$(python -c 'import json,os; from pathlib import Path; root=Path(os.environ["OUTPUT_ROOT"])/"metadata"; paths=[root/f"rank_{rank}.jsonl" for rank in range(4)]; ids=[int(json.loads(line)["sample_id"]) for path in paths if path.is_file() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]; assert len(ids)==len(set(ids)), "duplicate sample IDs in existing metadata"; print(len(ids))')"
START_NS="$(date +%s%N)"
export PYTHONPATH="$UPSTREAM_ROOT:$BASELINE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
for RANK in 0 1 2 3; do
  [[ $INTERRUPTED -eq 0 ]] || break
  EXTRA=()
  [[ $RESUME -eq 1 ]] && EXTRA+=(--resume)
  SPECA_INVOCATION_ID="$START_NS" CUDA_VISIBLE_DEVICES="${GPU_IDS[$RANK]}" \
    python "$GENERATE_SHARD" \
    --config "$ARCHIVED_CONFIG" --config-origin-dir "$CONFIG_ORIGIN_DIR" \
    --manifest "$ARCHIVED_MANIFEST" --shard-id "$RANK" \
    --world-size 4 --output-root "$OUTPUT_ROOT" --acknowledge-gpu-job \
    "${EXTRA[@]}" >>"$OUTPUT_ROOT/logs/rank_${RANK}.log" 2>&1 &
  PIDS+=("$!")
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
python "$WALL_CLOCK_SCRIPT" \
  --output-root "$OUTPUT_ROOT" --manifest "$ARCHIVED_MANIFEST" \
  --invocation-id "$START_NS" --start-ns "$START_NS" --end-ns "$END_NS" \
  --launcher-status "$STATUS" --baseline-count "$BASELINE_GENERATED" --world-size 4
exit "$STATUS"
