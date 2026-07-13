# JiT DiCache deferred runbook

This is a fail-closed, deferred protocol for `released_code_faithful_image_profile` (`profile=flux_image_released`). None of the CUDA/GPU commands below were run while implementing or editing this baseline. Run them only after the named devices are allocated, idle, and explicitly authorized. The GPU wrappers inspect their own target devices, refuse busy devices, and never attach to or signal another process.

Use one Bash session and a new output root. All unknown paths, seeds, experiment values, and GPU assignments are variables; this document deliberately supplies no final threshold, final gamma-nonfinite policy, or device index. Keep generated manifests, reports, and resolved configs immutable.

## 1. Environment

```bash
: "${PIXARC_ROOT:?absolute path to the PixARC checkout}"
: "${JIT_PYTHON:?absolute path to the JiT environment Python}"
: "${CHECKPOINT:?absolute path to the JiT-B/16 checkpoint}"
: "${OUTPUT_ROOT:?absolute path to a new output root outside PixARC}"
: "${IMAGENET_REFERENCE_NPZ:?absolute path to the frozen ImageNet ADM reference NPZ}"
: "${ADM_EVALUATOR:?absolute path to the frozen local ADM evaluator.py}"

export BASELINE_ROOT="$PIXARC_ROOT/JiT/baselines/dicache-style"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
export PIXARC_ROOT JIT_PYTHON CHECKPOINT OUTPUT_ROOT IMAGENET_REFERENCE_NPZ ADM_EVALUATOR
export PATH="$(dirname "$JIT_PYTHON"):$PATH"
export PYTHONPATH="$PIXARC_ROOT/third-party/JiT:$BASELINE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

test -x "$JIT_PYTHON"
test -s "$CHECKPOINT"
test -s "$IMAGENET_REFERENCE_NPZ"
test -f "$ADM_EVALUATOR"
test -d "$PIXARC_ROOT/.git"
test ! -e "$OUTPUT_ROOT"
[[ "$OUTPUT_ROOT/" != "$PIXARC_ROOT/"* ]]
mkdir -p "$OUTPUT_ROOT"/{manifests,configs,reports,selection,benchmark,metrics}
cd "$BASELINE_ROOT"
```

Fail if the audited sources changed; a mismatch requires a new compatibility review rather than continuing this campaign:

```bash
test "$(git -C "$PIXARC_ROOT" rev-parse HEAD)" = \
  "3377371d9b72fcdfa9407df3eca66759c91d2901"
test "$(git -C "$PIXARC_ROOT/baselines/DiCache" rev-parse HEAD)" = \
  "fdbe20b669c9174bbed5ec994de073fd881c8010"
test "$(git -C "$PIXARC_ROOT" rev-parse HEAD:third-party/JiT)" = \
  "d697163e4899e279a3c969d429832efecc9da115"
```

Run all implementation gates with CUDA hidden:

```bash
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" -m compileall -q dicache_style scripts tests
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" -m pytest -q -p no:cacheprovider tests
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/compare_official_core_parity.py \
  --output "$OUTPUT_ROOT/reports/official_core_parity.json"
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/compare_common_tool_interfaces.py \
  --pixarc-root "$PIXARC_ROOT"
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/generate_shard.py \
  --config configs/jit_b16_256_instrumented_full.yaml --preflight
! CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/generate_shard.py \
  --config configs/jit_b16_256_dicache.yaml --preflight
```

The last command must fail because the template's threshold/policy are unresolved.

## 2. Manifest

Build four seed-disjoint, immutable manifests. `build_manifest.py` also creates each required `MANIFEST.meta.json` sidecar.

```bash
: "${SMOKE_BASE_SEED:?unique smoke base seed}"
: "${PILOT_BASE_SEED:?unique 1K base seed}"
: "${VALIDATION_BASE_SEED:?unique 8K base seed}"
: "${FINAL_BASE_SEED:?unique 50K base seed}"
: "${SELECTION_RULE:?complete deterministic rule preregistered before 1K}"

export SMOKE_MANIFEST="$OUTPUT_ROOT/manifests/smoke8.jsonl"
export PILOT_MANIFEST="$OUTPUT_ROOT/manifests/search1k.jsonl"
export VALIDATION_MANIFEST="$OUTPUT_ROOT/manifests/validation8k.jsonl"
export MANIFEST="$OUTPUT_ROOT/manifests/final50k.jsonl"

CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/build_manifest.py \
  --output "$SMOKE_MANIFEST" --samples-per-class 1 --num-classes 8 \
  --base-seed "$SMOKE_BASE_SEED" --split-name smoke8 --world-size 1 --batch-size 1
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/build_manifest.py \
  --output "$PILOT_MANIFEST" --samples-per-class 1 --num-classes 1000 \
  --base-seed "$PILOT_BASE_SEED" --split-name search1k --world-size 4 --batch-size 1
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/build_manifest.py \
  --output "$VALIDATION_MANIFEST" --samples-per-class 8 --num-classes 1000 \
  --base-seed "$VALIDATION_BASE_SEED" --split-name validation8k --world-size 4 --batch-size 1
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/build_manifest.py \
  --output "$MANIFEST" --samples-per-class 50 --num-classes 1000 \
  --base-seed "$FINAL_BASE_SEED" --split-name final50k --world-size 4 --batch-size 1

CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/validate_manifest.py \
  --manifest "$SMOKE_MANIFEST" --expected-count 8 --expected-per-class 1 \
  --expected-num-classes 8 --world-size 1 --batch-size 1 \
  --base-seed "$SMOKE_BASE_SEED" --require-sidecar
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/validate_manifest.py \
  --manifest "$PILOT_MANIFEST" --expected-count 1000 --expected-per-class 1 \
  --expected-num-classes 1000 --world-size 4 --batch-size 1 \
  --base-seed "$PILOT_BASE_SEED" --require-sidecar \
  --disjoint-with "$SMOKE_MANIFEST"
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/validate_manifest.py \
  --manifest "$VALIDATION_MANIFEST" --expected-count 8000 --expected-per-class 8 \
  --expected-num-classes 1000 --world-size 4 --batch-size 1 \
  --base-seed "$VALIDATION_BASE_SEED" --require-sidecar \
  --disjoint-with "$SMOKE_MANIFEST" --disjoint-with "$PILOT_MANIFEST"
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/validate_manifest.py \
  --manifest "$MANIFEST" --expected-count 50000 --expected-per-class 50 \
  --expected-num-classes 1000 --world-size 4 --batch-size 1 \
  --base-seed "$FINAL_BASE_SEED" --require-sidecar \
  --disjoint-with "$SMOKE_MANIFEST" --disjoint-with "$PILOT_MANIFEST" \
  --disjoint-with "$VALIDATION_MANIFEST"
```

The preregistered rule must name quality constraints, invalid-run handling, the primary batch-1 CUDA-event latency statistic, tie-breaking, and that 8K performs the one-time selection while 50K is evaluation-only. Freeze its exact text before observing 1K:

```bash
test ! -e "$OUTPUT_ROOT/selection/selection_rule.txt"
printf '%s\n' "$SELECTION_RULE" > "$OUTPUT_ROOT/selection/selection_rule.txt"
```

Changing this file starts a new experiment.

## 3. Memory

These CPU estimates are planning evidence; measured allocated/reserved peaks come from stages 8–10.

```bash
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/estimate_cache_memory.py \
  --preset jit-b16-256 --batch-size 1 --probe-depth 1 --cache-dtype bfloat16 \
  --output-json "$OUTPUT_ROOT/reports/cache_memory_bf16.json"
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/estimate_cache_memory.py \
  --preset jit-b16-256 --batch-size 1 --probe-depth 1 --cache-dtype float32 \
  --output-json "$OUTPUT_ROOT/reports/cache_memory_fp32.json"
```

## 4. Smoke

Choose a provisional smoke-only threshold/policy and one allocated idle GPU. One matching provisional report is used by all three resolved smoke configs; inactive Full/upstream threshold fields are resolved to the same values so every materializer invocation is provenance-consistent.

```bash
: "${SMOKE_REL_L1_THRESHOLD:?provisional smoke-only threshold}"
: "${SMOKE_GAMMA_NONFINITE_POLICY:?provisional smoke-only gamma policy}"
: "${SINGLE_GPU_ID:?one allocated idle GPU ID or UUID}"
[[ "$SINGLE_GPU_ID" != *,* ]]

export SMOKE_SELECTION_REPORT="$OUTPUT_ROOT/selection/smoke_provisional.json"
export SMOKE_UPSTREAM_CONFIG="$OUTPUT_ROOT/configs/smoke_upstream.yaml"
export SMOKE_FULL_CONFIG="$OUTPUT_ROOT/configs/smoke_instrumented_full.yaml"
export SMOKE_CANDIDATE_CONFIG="$OUTPUT_ROOT/configs/smoke_dicache.yaml"

CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/record_selection.py \
  --model-family JiT --status provisional --probe-depth 1 \
  --rel-l1-thresh "$SMOKE_REL_L1_THRESHOLD" \
  --gamma-nonfinite-policy "$SMOKE_GAMMA_NONFINITE_POLICY" \
  --output "$SMOKE_SELECTION_REPORT"
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/materialize_dicache_config.py \
  --input configs/jit_b16_256_upstream_full.yaml --output "$SMOKE_UPSTREAM_CONFIG" \
  --checkpoint "$CHECKPOINT" --rel-l1-thresh "$SMOKE_REL_L1_THRESHOLD" \
  --gamma-nonfinite-policy "$SMOKE_GAMMA_NONFINITE_POLICY" \
  --selection-report "$SMOKE_SELECTION_REPORT"
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/materialize_dicache_config.py \
  --input configs/jit_b16_256_instrumented_full.yaml --output "$SMOKE_FULL_CONFIG" \
  --checkpoint "$CHECKPOINT" --rel-l1-thresh "$SMOKE_REL_L1_THRESHOLD" \
  --gamma-nonfinite-policy "$SMOKE_GAMMA_NONFINITE_POLICY" \
  --selection-report "$SMOKE_SELECTION_REPORT"
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/materialize_dicache_config.py \
  --input configs/jit_b16_256_dicache.yaml --output "$SMOKE_CANDIDATE_CONFIG" \
  --checkpoint "$CHECKPOINT" --rel-l1-thresh "$SMOKE_REL_L1_THRESHOLD" \
  --gamma-nonfinite-policy "$SMOKE_GAMMA_NONFINITE_POLICY" \
  --selection-report "$SMOKE_SELECTION_REPORT"

DICACHE_GPU_TESTS_ALLOWED=1 CUDA_VISIBLE_DEVICES="$SINGLE_GPU_ID" \
  bash scripts/run_deferred_smoke_tests.sh \
  --upstream-config "$SMOKE_UPSTREAM_CONFIG" --full-config "$SMOKE_FULL_CONFIG" \
  --candidate-config "$SMOKE_CANDIDATE_CONFIG" --manifest "$SMOKE_MANIFEST" \
  --output-root "$OUTPUT_ROOT/smoke_main"

test -f "$OUTPUT_ROOT/smoke_main/tensor_upstream_scratch_vs_probe_resume_parity.json"
test -f "$OUTPUT_ROOT/smoke_main/upstream_vs_instrumented_png_parity.json"
test -f "$OUTPUT_ROOT/smoke_main/smoke_gate.json"
```

That single wrapper invocation runs the real three-config generation paths and produces the tensor parity report, exact upstream/instrumented PNG report, candidate validation/metadata, and config-bound smoke gate. Do not create substitute reports.

After reviewing smoke-only nonfinite/fallback behavior, freeze one policy and the complete coarse threshold grid before 1K:

```bash
: "${GAMMA_NONFINITE_POLICY:?policy selected using smoke evidence only}"
: "${COARSE_CANDIDATES:?whitespace-separated safe tag=value coarse grid}"
case "$GAMMA_NONFINITE_POLICY" in
  official_propagate|latest_residual_fallback|force_full) ;;
  *) echo "invalid GAMMA_NONFINITE_POLICY" >&2; exit 2 ;;
esac
test ! -e "$OUTPUT_ROOT/selection/gamma_policy.txt"
test ! -e "$OUTPUT_ROOT/selection/coarse_candidates.txt"
printf '%s\n' "$GAMMA_NONFINITE_POLICY" > "$OUTPUT_ROOT/selection/gamma_policy.txt"
printf '%s\n' "$COARSE_CANDIDATES" > "$OUTPUT_ROOT/selection/coarse_candidates.txt"
```

The policy is now fixed for 1K, 8K, benchmark, and 50K.

## 5. Full parity

Consume the stage-4 upstream-versus-instrumented artifact; do not launch another smoke run.

```bash
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); assert r.get("schema_version")=="pixarc-image-tree-parity-v1"; assert r.get("sample_count")==8; assert r.get("exact") is True and r.get("differing_image_count")==0; assert r.get("max_absolute_uint8_error")==0 and r.get("max_relative_uint8_error")==0' \
  "$OUTPUT_ROOT/smoke_main/upstream_vs_instrumented_png_parity.json"
```

Any PNG difference blocks every later stage.

## 6. Resume parity

Consume the stage-4 real-model tensor report. Expected NFE and network-forward counts are derived from the resolved exact-Heun sampler; no count is hardcoded.

```bash
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); n=r.get("nine_resume_invariants",{}); o=r.get("operational_invariants",{}); assert r.get("schema_version")=="pixarc-jit-dicache-resume-parity-v2" and r.get("passed") is True; assert len(n)==9 and all(v is True for v in n.values()); assert o and all(v is True for v in o.values()); assert r.get("body_allclose") is True and r.get("probe_feature_allclose") is True and r.get("final_sample_allclose") is True; assert int(r.get("expected_nfe",0))>0 and int(r.get("expected_network_forwards",0))>0; assert all(all(float(v)==0.0 for v in r[k].values()) for k in ("body_final_layer_inputs","probe_depth_features","final_sample"))' \
  "$OUTPUT_ROOT/smoke_main/tensor_upstream_scratch_vs_probe_resume_parity.json"
```

All nine block/context/RoPE/probe/head/anchor invariants, lifecycle checks, raw finiteness checks, and zero-error tensor comparisons must pass.

## 7. Shadow

Run depth 1/2/3 as explicit diagnostic ablations. Every depth receives a matching provisional report. Depth 2/3 never become the main image profile.

```bash
: "${SHADOW_REL_L1_THRESHOLD:?diagnostic-only shadow threshold}"
test "$(cat "$OUTPUT_ROOT/selection/gamma_policy.txt")" = "$GAMMA_NONFINITE_POLICY"

for DEPTH in 1 2 3; do
  CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/record_selection.py \
    --model-family JiT --status provisional --probe-depth "$DEPTH" \
    --rel-l1-thresh "$SHADOW_REL_L1_THRESHOLD" \
    --gamma-nonfinite-policy "$GAMMA_NONFINITE_POLICY" \
    --output "$OUTPUT_ROOT/selection/shadow_depth_${DEPTH}_provisional.json"

  CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/materialize_dicache_config.py \
    --input configs/jit_b16_256_probe_shadow_full.yaml \
    --output "$OUTPUT_ROOT/configs/shadow_depth_${DEPTH}.yaml" \
    --checkpoint "$CHECKPOINT" --rel-l1-thresh "$SHADOW_REL_L1_THRESHOLD" \
    --gamma-nonfinite-policy "$GAMMA_NONFINITE_POLICY" --probe-depth "$DEPTH" \
    --selection-report "$OUTPUT_ROOT/selection/shadow_depth_${DEPTH}_provisional.json"

  DICACHE_GPU_TESTS_ALLOWED=1 CUDA_VISIBLE_DEVICES="$SINGLE_GPU_ID" \
    bash scripts/run_shadow_diagnostic.sh \
    --config "$OUTPUT_ROOT/configs/shadow_depth_${DEPTH}.yaml" \
    --manifest "$SMOKE_MANIFEST" \
    --output-root "$OUTPUT_ROOT/shadow/depth_${DEPTH}"
done
```

Retain each `shadow_diagnostics.json`. Only depth 1 may inform the main profile; depth 2/3 report diagnostic cost/correlation only.

## 8. 1K

Generate matched instrumented Full once. Then every preregistered coarse candidate must run four-GPU generation, strict paired metrics, trace aggregation, and a real batch-1 single-GPU CUDA-event Full/DiCache benchmark.

```bash
: "${FOUR_GPU_IDS:?four unique allocated idle GPU IDs or UUIDs, comma-separated}"
: "${PAIRED_CUDA_VISIBLE_DEVICES:?paired-metric CUDA visibility, or -1 for CPU}"
: "${LPIPS_DEVICE:?LPIPS torch device, such as cpu or cuda}"
: "${LPIPS_BATCH_SIZE:?positive LPIPS batch size}"
test "$(awk -F, '{print NF}' <<<"$FOUR_GPU_IDS")" -eq 4
test "$(cat "$OUTPUT_ROOT/selection/gamma_policy.txt")" = "$GAMMA_NONFINITE_POLICY"
test "$(cat "$OUTPUT_ROOT/selection/coarse_candidates.txt")" = "$COARSE_CANDIDATES"

export PILOT_FULL_CONFIG="$OUTPUT_ROOT/configs/search_full.yaml"
export PILOT_FULL_ROOT="$OUTPUT_ROOT/search1k/full"
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/materialize_dicache_config.py \
  --input configs/jit_b16_256_instrumented_full.yaml --output "$PILOT_FULL_CONFIG" \
  --checkpoint "$CHECKPOINT" --rel-l1-thresh "$SMOKE_REL_L1_THRESHOLD" \
  --gamma-nonfinite-policy "$SMOKE_GAMMA_NONFINITE_POLICY" \
  --selection-report "$SMOKE_SELECTION_REPORT"
DICACHE_GPU_TESTS_ALLOWED=1 bash scripts/launch_4gpu_50k.sh \
  --nonfinal-proxy --expected-records 1000 --expected-per-class 1 \
  --expected-num-classes 1000 --config "$PILOT_FULL_CONFIG" \
  --manifest "$PILOT_MANIFEST" --output-root "$PILOT_FULL_ROOT" \
  --gpu-ids "$FOUR_GPU_IDS"
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/aggregate_dicache_trace.py \
  --metadata-dir "$PILOT_FULL_ROOT/metadata" --world-size 4 \
  --output-json "$OUTPUT_ROOT/metrics/search_full_trace.json"

for SPEC in $COARSE_CANDIDATES; do
  TAG="${SPEC%%=*}"
  REL_L1_THRESHOLD="${SPEC#*=}"
  [[ "$SPEC" == *=* && "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2

  CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/record_selection.py \
    --model-family JiT --status provisional --probe-depth 1 \
    --rel-l1-thresh "$REL_L1_THRESHOLD" \
    --gamma-nonfinite-policy "$GAMMA_NONFINITE_POLICY" \
    --output "$OUTPUT_ROOT/selection/search_${TAG}_provisional.json"
  CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/materialize_dicache_config.py \
    --input configs/jit_b16_256_dicache.yaml \
    --output "$OUTPUT_ROOT/configs/search_dicache_${TAG}.yaml" \
    --checkpoint "$CHECKPOINT" --rel-l1-thresh "$REL_L1_THRESHOLD" \
    --gamma-nonfinite-policy "$GAMMA_NONFINITE_POLICY" \
    --selection-report "$OUTPUT_ROOT/selection/search_${TAG}_provisional.json"

  DICACHE_GPU_TESTS_ALLOWED=1 bash scripts/launch_4gpu_50k.sh \
    --nonfinal-proxy --expected-records 1000 --expected-per-class 1 \
    --expected-num-classes 1000 \
    --config "$OUTPUT_ROOT/configs/search_dicache_${TAG}.yaml" \
    --manifest "$PILOT_MANIFEST" --output-root "$OUTPUT_ROOT/search1k/dicache_${TAG}" \
    --gpu-ids "$FOUR_GPU_IDS"

  CUDA_VISIBLE_DEVICES="$PAIRED_CUDA_VISIBLE_DEVICES" bash scripts/evaluate_paired.sh \
    --reference-dir "$PILOT_FULL_ROOT/samples" \
    --candidate-dir "$OUTPUT_ROOT/search1k/dicache_${TAG}/samples" \
    --reference-manifest "$PILOT_MANIFEST" --candidate-manifest "$PILOT_MANIFEST" \
    --reference-run-metadata "$PILOT_FULL_ROOT/run_manifest.json" \
    --candidate-run-metadata "$OUTPUT_ROOT/search1k/dicache_${TAG}/run_manifest.json" \
    --output-json "$OUTPUT_ROOT/metrics/search_${TAG}_paired.json" \
    --output-csv "$OUTPUT_ROOT/metrics/search_${TAG}_paired.csv" \
    --expected-count 1000 --expected-per-class 1 --expected-num-classes 1000 \
    --lpips-device "$LPIPS_DEVICE" --lpips-batch-size "$LPIPS_BATCH_SIZE"

  CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/aggregate_dicache_trace.py \
    --metadata-dir "$OUTPUT_ROOT/search1k/dicache_${TAG}/metadata" --world-size 4 \
    --output-json "$OUTPUT_ROOT/metrics/search_${TAG}_trace.json"
  CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/build_benchmark_runner.py \
    --model-config "$OUTPUT_ROOT/configs/search_dicache_${TAG}.yaml" \
    --manifest "$PILOT_MANIFEST" --batch-size 1 \
    --output "$OUTPUT_ROOT/benchmark/search_${TAG}.runner.json"
  DICACHE_GPU_TESTS_ALLOWED=1 CUDA_VISIBLE_DEVICES="$SINGLE_GPU_ID" \
    bash scripts/benchmark_single_gpu.sh \
    --runner-factory dicache_style.jit_benchmark:build_benchmark_spec \
    --runner-config "$OUTPUT_ROOT/benchmark/search_${TAG}.runner.json" \
    --output-json "$OUTPUT_ROOT/metrics/search_${TAG}_benchmark.json" \
    --warmup-batches 10 --measured-batches 30
done
```

Use 1K only to define the fine grid. Freeze that complete grid before any 8K run; do not choose the final operating point on 1K.

```bash
: "${FINE_CANDIDATES:?whitespace-separated safe tag=value fine grid selected from 1K only}"
test ! -e "$OUTPUT_ROOT/selection/fine_candidates.txt"
printf '%s\n' "$FINE_CANDIDATES" > "$OUTPUT_ROOT/selection/fine_candidates.txt"
```

## 9. 8K

Generate matched Full once and give every frozen fine-grid point the same four evidence products as in 1K. The grid, policy, constraints, and tie-breaker are not changed after 8K starts.

```bash
test "$(cat "$OUTPUT_ROOT/selection/fine_candidates.txt")" = "$FINE_CANDIDATES"
test "$(cat "$OUTPUT_ROOT/selection/gamma_policy.txt")" = "$GAMMA_NONFINITE_POLICY"

export VALIDATION_FULL_CONFIG="$OUTPUT_ROOT/configs/validation_full.yaml"
export VALIDATION_FULL_ROOT="$OUTPUT_ROOT/validation8k/full"
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/materialize_dicache_config.py \
  --input configs/jit_b16_256_instrumented_full.yaml --output "$VALIDATION_FULL_CONFIG" \
  --checkpoint "$CHECKPOINT" --rel-l1-thresh "$SMOKE_REL_L1_THRESHOLD" \
  --gamma-nonfinite-policy "$SMOKE_GAMMA_NONFINITE_POLICY" \
  --selection-report "$SMOKE_SELECTION_REPORT"
DICACHE_GPU_TESTS_ALLOWED=1 bash scripts/launch_4gpu_50k.sh \
  --nonfinal-proxy --expected-records 8000 --expected-per-class 8 \
  --expected-num-classes 1000 --config "$VALIDATION_FULL_CONFIG" \
  --manifest "$VALIDATION_MANIFEST" --output-root "$VALIDATION_FULL_ROOT" \
  --gpu-ids "$FOUR_GPU_IDS"
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/aggregate_dicache_trace.py \
  --metadata-dir "$VALIDATION_FULL_ROOT/metadata" --world-size 4 \
  --output-json "$OUTPUT_ROOT/metrics/validation_full_trace.json"

for SPEC in $FINE_CANDIDATES; do
  TAG="${SPEC%%=*}"
  REL_L1_THRESHOLD="${SPEC#*=}"
  [[ "$SPEC" == *=* && "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2

  CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/record_selection.py \
    --model-family JiT --status provisional --probe-depth 1 \
    --rel-l1-thresh "$REL_L1_THRESHOLD" \
    --gamma-nonfinite-policy "$GAMMA_NONFINITE_POLICY" \
    --output "$OUTPUT_ROOT/selection/validation_${TAG}_provisional.json"
  CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/materialize_dicache_config.py \
    --input configs/jit_b16_256_dicache.yaml \
    --output "$OUTPUT_ROOT/configs/validation_dicache_${TAG}.yaml" \
    --checkpoint "$CHECKPOINT" --rel-l1-thresh "$REL_L1_THRESHOLD" \
    --gamma-nonfinite-policy "$GAMMA_NONFINITE_POLICY" \
    --selection-report "$OUTPUT_ROOT/selection/validation_${TAG}_provisional.json"

  DICACHE_GPU_TESTS_ALLOWED=1 bash scripts/launch_4gpu_50k.sh \
    --nonfinal-proxy --expected-records 8000 --expected-per-class 8 \
    --expected-num-classes 1000 \
    --config "$OUTPUT_ROOT/configs/validation_dicache_${TAG}.yaml" \
    --manifest "$VALIDATION_MANIFEST" \
    --output-root "$OUTPUT_ROOT/validation8k/dicache_${TAG}" \
    --gpu-ids "$FOUR_GPU_IDS"

  CUDA_VISIBLE_DEVICES="$PAIRED_CUDA_VISIBLE_DEVICES" bash scripts/evaluate_paired.sh \
    --reference-dir "$VALIDATION_FULL_ROOT/samples" \
    --candidate-dir "$OUTPUT_ROOT/validation8k/dicache_${TAG}/samples" \
    --reference-manifest "$VALIDATION_MANIFEST" \
    --candidate-manifest "$VALIDATION_MANIFEST" \
    --reference-run-metadata "$VALIDATION_FULL_ROOT/run_manifest.json" \
    --candidate-run-metadata "$OUTPUT_ROOT/validation8k/dicache_${TAG}/run_manifest.json" \
    --output-json "$OUTPUT_ROOT/metrics/validation_${TAG}_paired.json" \
    --output-csv "$OUTPUT_ROOT/metrics/validation_${TAG}_paired.csv" \
    --expected-count 8000 --expected-per-class 8 --expected-num-classes 1000 \
    --lpips-device "$LPIPS_DEVICE" --lpips-batch-size "$LPIPS_BATCH_SIZE"

  CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/aggregate_dicache_trace.py \
    --metadata-dir "$OUTPUT_ROOT/validation8k/dicache_${TAG}/metadata" --world-size 4 \
    --output-json "$OUTPUT_ROOT/metrics/validation_${TAG}_trace.json"
  CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/build_benchmark_runner.py \
    --model-config "$OUTPUT_ROOT/configs/validation_dicache_${TAG}.yaml" \
    --manifest "$VALIDATION_MANIFEST" --batch-size 1 \
    --output "$OUTPUT_ROOT/benchmark/validation_${TAG}.runner.json"
  DICACHE_GPU_TESTS_ALLOWED=1 CUDA_VISIBLE_DEVICES="$SINGLE_GPU_ID" \
    bash scripts/benchmark_single_gpu.sh \
    --runner-factory dicache_style.jit_benchmark:build_benchmark_spec \
    --runner-config "$OUTPUT_ROOT/benchmark/validation_${TAG}.runner.json" \
    --output-json "$OUTPUT_ROOT/metrics/validation_${TAG}_benchmark.json" \
    --warmup-batches 10 --measured-batches 30
done
```

Apply the stage-2 rule exactly once, without looking at 50K. The chosen tag and value must identify an existing fine-grid artifact. The decision binds the exact 8K paired/trace/benchmark files; the selected report binds that decision; both final configs then bind matching provenance.

```bash
: "${SELECTED_TAG:?winner chosen by the frozen 8K rule}"
: "${SELECTED_REL_L1_THRESHOLD:?winner threshold from the matching 8K artifact}"
case " $FINE_CANDIDATES " in
  *" $SELECTED_TAG=$SELECTED_REL_L1_THRESHOLD "*) ;;
  *) echo "selected tag/value is not in the frozen fine grid" >&2; exit 2 ;;
esac

export SELECTION_DECISION="$OUTPUT_ROOT/selection/selection_decision.json"
export FINAL_SELECTION_REPORT="$OUTPUT_ROOT/selection/selected.json"
export FINAL_FULL_SELECTION_REPORT="$OUTPUT_ROOT/selection/final_full_provisional.json"
export FINAL_FULL_CONFIG="$OUTPUT_ROOT/configs/final_full.yaml"
export FINAL_DICACHE_CONFIG="$OUTPUT_ROOT/configs/final_dicache.yaml"

CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/record_selection_decision.py \
  --model-family JiT --rel-l1-thresh "$SELECTED_REL_L1_THRESHOLD" \
  --gamma-nonfinite-policy "$GAMMA_NONFINITE_POLICY" \
  --selection-rule "$(cat "$OUTPUT_ROOT/selection/selection_rule.txt")" \
  --paired-report "$OUTPUT_ROOT/metrics/validation_${SELECTED_TAG}_paired.json" \
  --trace-report "$OUTPUT_ROOT/metrics/validation_${SELECTED_TAG}_trace.json" \
  --benchmark-report "$OUTPUT_ROOT/metrics/validation_${SELECTED_TAG}_benchmark.json" \
  --output "$SELECTION_DECISION"
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/record_selection.py \
  --model-family JiT --status selected --probe-depth 1 \
  --rel-l1-thresh "$SELECTED_REL_L1_THRESHOLD" \
  --gamma-nonfinite-policy "$GAMMA_NONFINITE_POLICY" \
  --decision-report "$SELECTION_DECISION" --output "$FINAL_SELECTION_REPORT"
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/record_selection.py \
  --model-family JiT --status provisional --probe-depth 1 \
  --rel-l1-thresh "$SELECTED_REL_L1_THRESHOLD" \
  --gamma-nonfinite-policy "$GAMMA_NONFINITE_POLICY" \
  --output "$FINAL_FULL_SELECTION_REPORT"
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/materialize_dicache_config.py \
  --input configs/jit_b16_256_instrumented_full.yaml --output "$FINAL_FULL_CONFIG" \
  --checkpoint "$CHECKPOINT" --rel-l1-thresh "$SELECTED_REL_L1_THRESHOLD" \
  --gamma-nonfinite-policy "$GAMMA_NONFINITE_POLICY" \
  --selection-report "$FINAL_FULL_SELECTION_REPORT"
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/materialize_dicache_config.py \
  --input configs/jit_b16_256_dicache.yaml --output "$FINAL_DICACHE_CONFIG" \
  --checkpoint "$CHECKPOINT" --rel-l1-thresh "$SELECTED_REL_L1_THRESHOLD" \
  --gamma-nonfinite-policy "$GAMMA_NONFINITE_POLICY" \
  --selection-report "$FINAL_SELECTION_REPORT"
```

The threshold, policy, rule, depth, and batch size are now immutable. Neither stage 10 nor 50K may tune them.

## 10. Benchmark

Run at least two repeated `matched_eager` pair benchmarks on the same single allocated GPU. Each uses 10 warmup and 30 measured batches and records CUDA-event latency, first-execution time, cache counters, peak allocated/reserved memory, graph breaks, and guard failures.

```bash
: "${BENCHMARK_REPEATS:?integer repeat count of at least two}"
case "$BENCHMARK_REPEATS" in ''|*[!0-9]*) exit 2 ;; esac
test "$BENCHMARK_REPEATS" -ge 2

export FINAL_BENCH_RUNNER="$OUTPUT_ROOT/benchmark/final_matched.runner.json"
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/build_benchmark_runner.py \
  --model-config "$FINAL_DICACHE_CONFIG" --manifest "$MANIFEST" --batch-size 1 \
  --output "$FINAL_BENCH_RUNNER"
for REP in $(seq 1 "$BENCHMARK_REPEATS"); do
  DICACHE_GPU_TESTS_ALLOWED=1 CUDA_VISIBLE_DEVICES="$SINGLE_GPU_ID" \
    bash scripts/benchmark_single_gpu.sh \
    --runner-factory dicache_style.jit_benchmark:build_benchmark_spec \
    --runner-config "$FINAL_BENCH_RUNNER" \
    --output-json "$OUTPUT_ROOT/metrics/final_matched_repeat_${REP}.json" \
    --warmup-batches 10 --measured-batches 30
done
```

Then run the isolated three-mode/five-role compile matrix. `upstream` measures whole-model `upstream_full`; `matched_eager` and `blockwise` each measure instrumented Full and DiCache. The orchestrator derives sampler counts, compares raw floating outputs, retains `TORCH_LOGS=graph_breaks,recompiles`, and records first/steady latency and memory. Its default zero tolerances are intentionally left unchanged.

```bash
export COMPILE_MATRIX="$OUTPUT_ROOT/metrics/compile_matrix.json"
DICACHE_GPU_TESTS_ALLOWED=1 CUDA_VISIBLE_DEVICES="$SINGLE_GPU_ID" \
  bash scripts/run_compile_matrix.sh \
  --runner-config "$FINAL_BENCH_RUNNER" \
  --output-dir "$OUTPUT_ROOT/benchmark/compile_matrix_rows" \
  --output-json "$COMPILE_MATRIX" --warmup-batches 10 --measured-batches 30
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); assert r.get("schema_version")=="pixarc-jit-compile-matrix-v1" and r.get("passed") is True' \
  "$COMPILE_MATRIX"
```

Primary speedup remains matched-eager instrumented Full versus matched-eager DiCache. Whole-model upstream and blockwise rows are supplemental.

## 11. 4GPU50K

Create the release gate before either final launch. It binds the two final configs, final manifest and sidecar, selected report and its 8K decision, stage-4 tensor parity and smoke gate, stage-10 compile matrix, and the exact port/upstream executable/config source bytes. Smoke and compile reports must carry the same source binding, so an edit between evidence stages fails closed.

```bash
test -f "$MANIFEST.meta.json"
export RELEASE_GATE="$OUTPUT_ROOT/selection/release_gate.json"
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/release_gate.py create \
  --model-family JiT --full-config "$FINAL_FULL_CONFIG" \
  --candidate-config "$FINAL_DICACHE_CONFIG" --manifest "$MANIFEST" \
  --selection-report "$FINAL_SELECTION_REPORT" \
  --parity-report "$OUTPUT_ROOT/smoke_main/tensor_upstream_scratch_vs_probe_resume_parity.json" \
  --smoke-report "$OUTPUT_ROOT/smoke_main/smoke_gate.json" \
  --compile-report "$COMPILE_MATRIX" --output "$RELEASE_GATE"

export FULL_ROOT="$OUTPUT_ROOT/final50k/full"
export DICACHE_ROOT="$OUTPUT_ROOT/final50k/dicache"
DICACHE_GPU_TESTS_ALLOWED=1 bash scripts/launch_4gpu_50k.sh \
  --config "$FINAL_FULL_CONFIG" --manifest "$MANIFEST" \
  --output-root "$FULL_ROOT" --release-gate "$RELEASE_GATE" \
  --gpu-ids "$FOUR_GPU_IDS"
DICACHE_GPU_TESTS_ALLOWED=1 bash scripts/launch_4gpu_50k.sh \
  --config "$FINAL_DICACHE_CONFIG" --manifest "$MANIFEST" \
  --output-root "$DICACHE_ROOT" --release-gate "$RELEASE_GATE" \
  --gpu-ids "$FOUR_GPU_IDS"
```

The launcher verifies the release gate before any CUDA access and archives that exact gate beside the config/manifest prefix. For an interrupted validated run, repeat the identical corresponding command with `--resume`; a durable prefix without the same archived gate is rejected. Never change grouping or overwrite partial outputs. The 50K results are evaluation-only and must not alter selection.

## 12. Validate

```bash
for RUN_ROOT in "$FULL_ROOT" "$DICACHE_ROOT"; do
  CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/validate_outputs.py \
    --sample-dir "$RUN_ROOT/samples" --metadata-dir "$RUN_ROOT/metadata" \
    --run-metadata "$RUN_ROOT/run_manifest.json" --manifest "$MANIFEST" \
    --expected-count 50000 --expected-per-class 50 --expected-num-classes 1000 \
    --resolution 256
done
```

Both trees must contain exactly the manifest IDs, RGB `uint8` 256x256 PNGs, finite metadata, valid call counts, and matching checkpoint/config/manifest identities.

## 13. Distribution

Use the same frozen local evaluator, reference NPZ, preprocessing, and user-assigned metric device for both methods.

```bash
: "${DISTRIBUTION_CUDA_VISIBLE_DEVICES:?assigned metric GPU visibility, or -1 for CPU}"
for METHOD in full dicache; do
  RUN_ROOT="$OUTPUT_ROOT/final50k/$METHOD"
  CUDA_VISIBLE_DEVICES="$DISTRIBUTION_CUDA_VISIBLE_DEVICES" \
    bash scripts/evaluate_distribution.sh \
    --sample-dir "$RUN_ROOT/samples" --manifest "$MANIFEST" \
    --run-metadata "$RUN_ROOT/run_manifest.json" \
    --reference-npz "$IMAGENET_REFERENCE_NPZ" --evaluator "$ADM_EVALUATOR" \
    --sample-npz "$OUTPUT_ROOT/metrics/${METHOD}_samples.npz" \
    --output-json "$OUTPUT_ROOT/metrics/${METHOD}_distribution.json" \
    --expected-count 50000 --expected-per-class 50 --expected-num-classes 1000
done
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" -c 'import json,sys; from dicache_style.distribution_metrics import distribution_deltas; from dicache_style.metadata import atomic_write_json; f=json.load(open(sys.argv[1],encoding="utf-8")); d=json.load(open(sys.argv[2],encoding="utf-8")); atomic_write_json(sys.argv[3],distribution_deltas(f,d))' \
  "$OUTPUT_ROOT/metrics/full_distribution.json" \
  "$OUTPUT_ROOT/metrics/dicache_distribution.json" \
  "$OUTPUT_ROOT/metrics/distribution_delta.json"
```

Report FID, sFID, IS, precision, recall, evaluator/reference identities, and deltas without post-hoc tuning.

## 14. Paired

Pair only the new release-gated Full and DiCache trees produced from the identical final manifest. Never pair against a legacy Full output.

```bash
CUDA_VISIBLE_DEVICES="$PAIRED_CUDA_VISIBLE_DEVICES" bash scripts/evaluate_paired.sh \
  --reference-dir "$FULL_ROOT/samples" --candidate-dir "$DICACHE_ROOT/samples" \
  --reference-manifest "$MANIFEST" --candidate-manifest "$MANIFEST" \
  --reference-run-metadata "$FULL_ROOT/run_manifest.json" \
  --candidate-run-metadata "$DICACHE_ROOT/run_manifest.json" \
  --output-json "$OUTPUT_ROOT/metrics/final_paired.json" \
  --output-csv "$OUTPUT_ROOT/metrics/final_paired.csv" \
  --expected-count 50000 --expected-per-class 50 --expected-num-classes 1000 \
  --lpips-device "$LPIPS_DEVICE" --lpips-batch-size "$LPIPS_BATCH_SIZE"
```

Strict identity/pairing validation runs before PSNR, SSIM, or LPIPS. If LPIPS dependencies or local AlexNet weights are unavailable, use `--skip-lpips` only as an explicitly incomplete secondary artifact; never silently substitute a model.

## 15. Trace

Persist Full and DiCache trajectory summaries; aggregate every repeated matched benchmark into one latency/peak-memory artifact; retain and validate the compile matrix and both launcher wall-clock reports.

```bash
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/aggregate_dicache_trace.py \
  --metadata-dir "$FULL_ROOT/metadata" --world-size 4 \
  --output-json "$OUTPUT_ROOT/metrics/final_full_trace.json"
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/aggregate_dicache_trace.py \
  --metadata-dir "$DICACHE_ROOT/metadata" --world-size 4 \
  --output-json "$OUTPUT_ROOT/metrics/final_dicache_trace.json"

BENCHMARK_INPUT_ARGS=()
for REP in $(seq 1 "$BENCHMARK_REPEATS"); do
  REPORT="$OUTPUT_ROOT/metrics/final_matched_repeat_${REP}.json"
  test -f "$REPORT"
  BENCHMARK_INPUT_ARGS+=(--input "$REPORT")
done
CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" scripts/aggregate_benchmark_reports.py \
  "${BENCHMARK_INPUT_ARGS[@]}" \
  --output "$OUTPUT_ROOT/metrics/final_matched_benchmark_aggregate.json"

CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); assert r.get("schema_version")=="pixarc-jit-compile-matrix-v1" and r.get("passed") is True; m=r.get("matrix",{}); assert set(m)=={"upstream","matched_eager","blockwise"}; assert sum(len(v) for v in m.values())==5' \
  "$COMPILE_MATRIX"
test -d "$OUTPUT_ROOT/benchmark/compile_matrix_rows"

CUDA_VISIBLE_DEVICES='' "$JIT_PYTHON" -c 'import json,sys; rows=[json.load(open(p,encoding="utf-8")) for p in sys.argv[1:]]; assert all(r.get("schema_version")=="pixarc-dicache-cumulative-wall-clock-v1" and r.get("completed") is True and r.get("invocation_chain_valid") is True and r.get("world_size")==4 and r.get("cumulative_sample_count")==50000 for r in rows)' \
  "$FULL_ROOT/four_gpu_wall_clock.json" "$DICACHE_ROOT/four_gpu_wall_clock.json"
test ! -e "$OUTPUT_ROOT/metrics/full_four_gpu_wall_clock.json"
test ! -e "$OUTPUT_ROOT/metrics/dicache_four_gpu_wall_clock.json"
cp "$FULL_ROOT/four_gpu_wall_clock.json" \
  "$OUTPUT_ROOT/metrics/full_four_gpu_wall_clock.json"
cp "$DICACHE_ROOT/four_gpu_wall_clock.json" \
  "$OUTPUT_ROOT/metrics/dicache_four_gpu_wall_clock.json"
```

The persistent evidence set is the two final trace reports, the repeated matched CUDA-event latency/memory aggregate, `compile_matrix.json` plus its row reports and Dynamo logs, and the two four-GPU wall-clock JSONs. Component host timers are diagnostic only; primary latency is batch-1 CUDA-event timing. Each launcher report retains immutable per-invocation wall clocks, reports their active-time sum separately from first-start-to-final-end time (which includes resume gaps), and must never label only the final resumed suffix as 50K throughput.
