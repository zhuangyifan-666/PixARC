#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --upstream-config FILE --full-config FILE --candidate-config FILE --manifest FILE --output-root DIR" >&2
}

UPSTREAM_CONFIG=""
FULL_CONFIG=""
CANDIDATE_CONFIG=""
MANIFEST=""
OUTPUT_ROOT=""
while (($#)); do
  case "$1" in
    --upstream-config) [[ $# -ge 2 ]] || { usage; exit 2; }; UPSTREAM_CONFIG="$2"; shift 2 ;;
    --full-config) [[ $# -ge 2 ]] || { usage; exit 2; }; FULL_CONFIG="$2"; shift 2 ;;
    --candidate-config) [[ $# -ge 2 ]] || { usage; exit 2; }; CANDIDATE_CONFIG="$2"; shift 2 ;;
    --manifest) [[ $# -ge 2 ]] || { usage; exit 2; }; MANIFEST="$2"; shift 2 ;;
    --output-root) [[ $# -ge 2 ]] || { usage; exit 2; }; OUTPUT_ROOT="$2"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -f "$UPSTREAM_CONFIG" && -f "$FULL_CONFIG" && -f "$CANDIDATE_CONFIG" \
   && -f "$MANIFEST" && -n "$OUTPUT_ROOT" ]] || { usage; exit 2; }

if [[ "${DICACHE_GPU_TESTS_ALLOWED:-0}" != "1" ]]; then
  echo "Deferred GPU smoke tests are locked. Set DICACHE_GPU_TESTS_ALLOWED=1 only after the target GPU is idle." >&2
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
for required in generate_shard.py deferred_run_guard.py run_gpu_model_parity.py \
  validate_outputs.py compare_image_trees.py record_smoke_gate.py; do
  [[ -f "$SCRIPT_ROOT/$required" ]] || { echo "Missing $SCRIPT_ROOT/$required" >&2; exit 2; }
done
if [[ -e "$OUTPUT_ROOT" && -n "$(find "$OUTPUT_ROOT" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing non-empty smoke output root: $OUTPUT_ROOT" >&2
  exit 4
fi
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd -- "$OUTPUT_ROOT" && pwd -P)"
export PYTHONPATH="$UPSTREAM_ROOT:$BASELINE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python "$SCRIPT_ROOT/deferred_run_guard.py" \
  --config "$UPSTREAM_CONFIG" --manifest "$MANIFEST" --max-records 8 \
  --require-mode upstream_full
python "$SCRIPT_ROOT/deferred_run_guard.py" \
  --config "$FULL_CONFIG" --manifest "$MANIFEST" --max-records 8 \
  --require-mode instrumented_full
python "$SCRIPT_ROOT/deferred_run_guard.py" \
  --config "$CANDIDATE_CONFIG" --manifest "$MANIFEST" --max-records 8 \
  --require-mode dicache

EXPECTED_COUNT="$(python -c 'from dicache_style.manifest import load_manifest; import sys; print(len(load_manifest(sys.argv[1])))' "$MANIFEST")"
[[ "$EXPECTED_COUNT" =~ ^[1-8]$ ]] || {
  echo "Smoke manifest must contain between 1 and 8 records, got $EXPECTED_COUNT." >&2
  exit 5
}

# Hard gate before any output generation: every exact body/head call, final
# sample, decoded float image, the nine resume invariants, and candidate raw
# finiteness/reset/cache lifecycle must pass on the first manifest record.
python "$SCRIPT_ROOT/run_gpu_model_parity.py" \
  --model-config "$CANDIDATE_CONFIG" --manifest "$MANIFEST" \
  --output-json "$OUTPUT_ROOT/model_parity.json" --atol 0 --rtol 0

DICACHE_INVOCATION_ID="smoke-upstream-$(date +%s%N)" \
python "$SCRIPT_ROOT/generate_shard.py" \
  --config "$UPSTREAM_CONFIG" --config-origin-dir "$(cd -- "$(dirname -- "$UPSTREAM_CONFIG")" && pwd -P)" \
  --manifest "$MANIFEST" --shard-id 0 --world-size 1 \
  --output-root "$OUTPUT_ROOT/upstream" --acknowledge-gpu-job --nonfinal-proxy
DICACHE_INVOCATION_ID="smoke-full-$(date +%s%N)" \
python "$SCRIPT_ROOT/generate_shard.py" \
  --config "$FULL_CONFIG" --config-origin-dir "$(cd -- "$(dirname -- "$FULL_CONFIG")" && pwd -P)" \
  --manifest "$MANIFEST" --shard-id 0 --world-size 1 \
  --output-root "$OUTPUT_ROOT/full" --acknowledge-gpu-job --nonfinal-proxy
DICACHE_INVOCATION_ID="smoke-candidate-$(date +%s%N)" \
python "$SCRIPT_ROOT/generate_shard.py" \
  --config "$CANDIDATE_CONFIG" --config-origin-dir "$(cd -- "$(dirname -- "$CANDIDATE_CONFIG")" && pwd -P)" \
  --manifest "$MANIFEST" --shard-id 0 --world-size 1 \
  --output-root "$OUTPUT_ROOT/candidate" --acknowledge-gpu-job --nonfinal-proxy

for run in upstream full candidate; do
  python "$SCRIPT_ROOT/validate_outputs.py" \
    --sample-dir "$OUTPUT_ROOT/$run/samples" --manifest "$MANIFEST" \
    --metadata-dir "$OUTPUT_ROOT/$run/metadata" \
    --run-metadata "$OUTPUT_ROOT/$run/run_manifest.json" \
    --expected-count "$EXPECTED_COUNT" --resolution 256 \
    >"$OUTPUT_ROOT/${run}_validation.json"
done

python "$SCRIPT_ROOT/compare_image_trees.py" \
  --reference-dir "$OUTPUT_ROOT/upstream/samples" \
  --candidate-dir "$OUTPUT_ROOT/full/samples" \
  --output-json "$OUTPUT_ROOT/upstream_vs_instrumented_png.json" \
  --require-exact

python - "$OUTPUT_ROOT/candidate/summaries/rank_0_summary.json" \
  "$OUTPUT_ROOT/candidate/metadata/rank_0.jsonl" \
  "$OUTPUT_ROOT/model_parity.json" "$EXPECTED_COUNT" <<'PY'
import json
import math
import sys


def assert_finite(value, path="root"):
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")
    elif isinstance(value, float):
        assert math.isfinite(value), f"non-finite numeric summary at {path}"


summary_path, metadata_path, parity_path, count_text = sys.argv[1:]
expected_count = int(count_text)
with open(parity_path, encoding="utf-8") as handle:
    parity = json.load(handle)
assert_finite(parity, "model_parity")
assert parity.get("passed") is True
assert all(parity.get("nine_resume_invariants", {}).values())
expected_nfe = int(parity.get("expected_nfe", -1))
expected_forwards = int(parity.get("expected_combined_forwards", -1))
assert expected_nfe > 0 and expected_forwards > 0

with open(summary_path, encoding="utf-8") as handle:
    summary = json.load(handle)
assert_finite(summary, "rank_summary")
trajectory_count = int(summary.get("trajectory_count", -1))
assert trajectory_count == expected_count, "candidate trajectory count mismatch"
assert int(summary.get("total_nfe", -1)) == expected_nfe * trajectory_count
assert int(summary.get("total_stream_calls", -1)) == expected_forwards * trajectory_count
assert int(summary.get("network_forward_count", -1)) == expected_forwards * trajectory_count
assert int(summary.get("direct_full_count", 0)) + int(summary.get("resumed_full_count", 0)) > 0, "candidate executed no Full"
assert int(summary.get("reuse_count", 0)) > 0, "candidate executed no Reuse"
assert int(summary.get("probe_count", 0)) > 0, "candidate executed no probe"
assert int(summary.get("dcta_count", 0)) > 0, "candidate executed no DCTA"

rows = []
with open(metadata_path, encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            row = json.loads(line)
            assert_finite(row, f"metadata[{len(rows)}]")
            rows.append(row)
assert len(rows) == expected_count, "candidate metadata row count mismatch"
for row in rows:
    assert row.get("trajectory_call_count_valid") is True
    assert int(row.get("trajectory_total_nfe", -1)) == expected_nfe
    assert int(row.get("trajectory_total_stream_calls", -1)) == expected_forwards
    assert int(row.get("trajectory_network_forward_count", -1)) == expected_forwards

operational = parity.get("operational_invariants", {})
for key in (
    "candidate_sample_is_finite_before_uint8",
    "candidate_decoded_image_is_finite_before_uint8",
    "candidate_runtime_reset",
    "candidate_cache_is_zero",
    "probe_only_runtime_reset",
    "probe_only_cache_is_zero",
):
    assert operational.get(key) is True, f"failed lifecycle/finiteness gate: {key}"
PY

python "$SCRIPT_ROOT/record_smoke_gate.py" \
  --model-family "$MODEL_FAMILY" --expected-count "$EXPECTED_COUNT" \
  --resume-parity "$OUTPUT_ROOT/model_parity.json" \
  --png-parity "$OUTPUT_ROOT/upstream_vs_instrumented_png.json" \
  --candidate-validation "$OUTPUT_ROOT/candidate_validation.json" \
  --candidate-summary "$OUTPUT_ROOT/candidate/summaries/rank_0_summary.json" \
  --candidate-metadata "$OUTPUT_ROOT/candidate/metadata/rank_0.jsonl" \
  --upstream-config "$UPSTREAM_CONFIG" --full-config "$FULL_CONFIG" \
  --candidate-config "$CANDIDATE_CONFIG" \
  --output "$OUTPUT_ROOT/smoke_gate.json"

echo "PixelGen deferred GPU smoke passed; bound gate: $OUTPUT_ROOT/smoke_gate.json"
