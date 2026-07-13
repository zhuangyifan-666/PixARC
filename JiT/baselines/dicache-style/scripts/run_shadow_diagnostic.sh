#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --config FILE --manifest FILE --output-root DIR" >&2
}

CONFIG=""
MANIFEST=""
OUTPUT_ROOT=""
while (($#)); do
  case "$1" in
    --config) [[ $# -ge 2 ]] || { usage; exit 2; }; CONFIG="$2"; shift 2 ;;
    --manifest) [[ $# -ge 2 ]] || { usage; exit 2; }; MANIFEST="$2"; shift 2 ;;
    --output-root) [[ $# -ge 2 ]] || { usage; exit 2; }; OUTPUT_ROOT="$2"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -f "$CONFIG" && -f "$MANIFEST" && -n "$OUTPUT_ROOT" ]] || { usage; exit 2; }
if [[ "${DICACHE_GPU_TESTS_ALLOWED:-0}" != "1" ]]; then
  echo "Shadow diagnostics are locked. Set DICACHE_GPU_TESTS_ALLOWED=1 only after the target GPU is idle." >&2
  exit 2
fi
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != *,* ]] || {
  echo "Shadow wrapper requires exactly one explicitly visible allocated GPU." >&2
  exit 2
}
DEVICE="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
[[ -n "$DEVICE" ]] || { echo "Refusing shadow diagnostic: empty GPU identifier." >&2; exit 2; }
if ! DEVICE_PIDS="$(nvidia-smi -i "$DEVICE" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)" \
  || [[ -n "$(sed '/^[[:space:]]*$/d' <<<"$DEVICE_PIDS")" ]]; then
  echo "Refusing shadow diagnostic: GPU $DEVICE has a compute process or could not be inspected." >&2
  exit 3
fi
if ! DEVICE_STATE="$(nvidia-smi -i "$DEVICE" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null)" \
  || ! awk -F',' 'NR == 1 {for(i=1;i<=2;i++) gsub(/[[:space:]]/, "", $i); valid=(NF==2 && $1~/^[0-9]+([.][0-9]+)?$/ && $2~/^[0-9]+([.][0-9]+)?$/); busy=($1+0>5 || $2+0>1024)} END {exit !(NR==1 && valid && !busy)}' <<<"$DEVICE_STATE"; then
  echo "Refusing shadow diagnostic: GPU $DEVICE is busy or returned invalid telemetry." >&2
  exit 3
fi

SCRIPT_ROOT="$(cd -- "$(dirname -- "$0")" && pwd -P)"
BASELINE_ROOT="$(cd -- "$SCRIPT_ROOT/.." && pwd -P)"
PIXARC_ROOT="$(cd -- "$BASELINE_ROOT/../../.." && pwd -P)"
MODEL_FAMILY="$(basename -- "$(cd -- "$BASELINE_ROOT/../.." && pwd -P)")"
UPSTREAM_ROOT="$PIXARC_ROOT/third-party/$MODEL_FAMILY"
[[ -f "$SCRIPT_ROOT/generate_shard.py" ]] || { echo "Missing $SCRIPT_ROOT/generate_shard.py" >&2; exit 2; }
if [[ -e "$OUTPUT_ROOT" && -n "$(find "$OUTPUT_ROOT" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing non-empty shadow output root: $OUTPUT_ROOT" >&2
  exit 4
fi
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd -- "$OUTPUT_ROOT" && pwd -P)"
export PYTHONPATH="$UPSTREAM_ROOT:$BASELINE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python "$SCRIPT_ROOT/deferred_run_guard.py" \
  --config "$CONFIG" --manifest "$MANIFEST" --max-records 8 \
  --require-mode probe_shadow_full --require-trace-mode shadow
DICACHE_INVOCATION_ID="shadow-$(date +%s%N)" \
python "$SCRIPT_ROOT/generate_shard.py" \
  --config "$CONFIG" --config-origin-dir "$(cd -- "$(dirname -- "$CONFIG")" && pwd -P)" \
  --manifest "$MANIFEST" --shard-id 0 --world-size 1 \
  --output-root "$OUTPUT_ROOT" --acknowledge-gpu-job --nonfinal-proxy
EXPECTED_COUNT="$(wc -l < "$MANIFEST")"
python "$SCRIPT_ROOT/validate_outputs.py" \
  --sample-dir "$OUTPUT_ROOT/samples" --metadata-dir "$OUTPUT_ROOT/metadata" \
  --run-metadata "$OUTPUT_ROOT/run_manifest.json" --manifest "$MANIFEST" \
  --expected-count "$EXPECTED_COUNT"
python "$SCRIPT_ROOT/aggregate_dicache_trace.py" \
  --metadata-dir "$OUTPUT_ROOT/metadata" --world-size 1 \
  --output-json "$OUTPUT_ROOT/shadow_diagnostics.json"
python -c 'import json,math,sys; r=json.load(open(sys.argv[1], encoding="utf-8")); s=r.get("shadow_diagnostics", {}); stages=s.get("by_solver_stage", {}); required=("predictor","corrector"); assert r.get("all_call_counts_valid") is True, r; assert s.get("probe_full_pair_count",0)>=2 and s.get("probe_full_spearman") is not None, s; assert s.get("reuse_error_pair_count",0)>0 and s.get("mean_zero_order_relative_error") is not None and s.get("mean_dcta_relative_error") is not None, s; assert s.get("gamma_count",0)>0 and s.get("dcta_used_count",0)>0 and math.isfinite(float(s["gamma_mean"])), s; assert all(name in stages and stages[name].get("event_count",0)>0 for name in required), stages' \
  "$OUTPUT_ROOT/shadow_diagnostics.json"
