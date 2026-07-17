# PixelGen DiCache deferred runbook

This is a deferred, fail-closed protocol for `released_code_faithful_image_profile` (`profile=flux_image_released`). None of the CUDA commands below were run while implementing or editing this baseline. Run a GPU command only after the devices have been allocated, confirmed idle, and new CUDA work has been explicitly authorized. The wrappers inspect telemetry, refuse busy devices, never attach to or signal another process, and refuse non-empty output roots unless their documented resume contract is satisfied.

All unknown paths and device assignments are required variables; the runbook deliberately contains no default GPU index, final threshold, or final gamma-nonfinite policy. Start from a new `OUTPUT_ROOT` and keep every generated JSON/YAML immutable.

```bash
: "${PIXARC_ROOT:?absolute path to the PixARC checkout}"
: "${CHECKPOINT:?absolute path to PixelGen_XL_160ep.ckpt}"
: "${OUTPUT_ROOT:?absolute path to a new output root}"
: "${IMAGENET_REFERENCE_NPZ:?absolute path to the local ADM ImageNet reference NPZ}"
: "${ADM_EVALUATOR:?absolute path to the one frozen local ADM evaluator.py}"
: "${SINGLE_GPU_VISIBLE_DEVICE:?one allocated idle CUDA_VISIBLE_DEVICES entry}"
: "${FOUR_GPU_VISIBLE_DEVICES:?four comma-separated allocated idle entries}"

export BASELINE_ROOT="$PIXARC_ROOT/PixelGen/baselines/dicache-style"
export MANIFEST="$OUTPUT_ROOT/manifests/final_50k.jsonl"
export BATCH_SIZE=4

test -d "$PIXARC_ROOT/.git"
test -f "$CHECKPOINT"
test "$BATCH_SIZE" -eq 1
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"/{manifests,configs,metrics,selection,benchmark}
cd "$PIXARC_ROOT"
export PYTHONPATH="$PIXARC_ROOT/third-party/PixelGen:$BASELINE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

## 1. Environment audit

Perform this read-only audit immediately before the deferred campaign. If the allocation or source tree changes, stop and audit again; never attach to, pause, stop, or signal a listed process.

```bash
git -C "$PIXARC_ROOT" rev-parse HEAD
git -C "$PIXARC_ROOT/baselines/DiCache" rev-parse HEAD
git -C "$PIXARC_ROOT" status --short
bash "$BASELINE_ROOT/scripts/inspect_active_runs.sh"
```

The implementation was prepared against PixARC `3377371d9b72fcdfa9407df3eca66759c91d2901`, DiCache `fdbe20b669c9174bbed5ec994de073fd881c8010`, and PixelGen subtree `3043acf90f255a264f1445bda9ea8d468ba91a58`. A mismatch requires a new compatibility review.

Run the CPU-only gates with CUDA hidden:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q "$BASELINE_ROOT/tests"
CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/compare_official_core_parity.py"
CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/compare_common_tool_interfaces.py" \
  --pixarc-root "$PIXARC_ROOT"
```

Verify the device-variable shapes without selecting any particular GPU in this document:

```bash
test -n "$SINGLE_GPU_VISIBLE_DEVICE"
test "${SINGLE_GPU_VISIBLE_DEVICE#*,}" = "$SINGLE_GPU_VISIBLE_DEVICE"
test "$(awk -F, '{print NF}' <<<"$FOUR_GPU_VISIBLE_DEVICES")" -eq 4
```

## 2. Immutable manifests and preregistered selection rule

Build four disjoint CPU-noise manifests. `build_manifest.py` also writes the required `MANIFEST.meta.json` sidecar. Full and DiCache always consume the same manifest within a stage.

```bash
CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/build_manifest.py" \
  --output "$OUTPUT_ROOT/manifests/smoke_4.jsonl" \
  --samples-per-class 1 --num-classes 4 --base-seed 100000 \
  --split-name smoke --world-size 1 --batch-size "$BATCH_SIZE" \
  --generator-device cpu --noise-dtype float32 --noise-shape 3 256 256

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/build_manifest.py" \
  --output "$OUTPUT_ROOT/manifests/search_1k.jsonl" \
  --samples-per-class 1 --num-classes 1000 --base-seed 1100000 \
  --split-name threshold-search-1k --world-size 4 --batch-size "$BATCH_SIZE" \
  --generator-device cpu --noise-dtype float32 --noise-shape 3 256 256

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/build_manifest.py" \
  --output "$OUTPUT_ROOT/manifests/validation_8k.jsonl" \
  --samples-per-class 8 --num-classes 1000 --base-seed 2100000 \
  --split-name threshold-validation-8k --world-size 4 --batch-size "$BATCH_SIZE" \
  --generator-device cpu --noise-dtype float32 --noise-shape 3 256 256

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/build_manifest.py" \
  --output "$MANIFEST" \
  --samples-per-class 50 --num-classes 1000 --base-seed 3100000 \
  --split-name final-50k --world-size 4 --batch-size "$BATCH_SIZE" \
  --generator-device cpu --noise-dtype float32 --noise-shape 3 256 256

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/validate_manifest.py" \
  --manifest "$OUTPUT_ROOT/manifests/smoke_4.jsonl" \
  --expected-count 4 --expected-per-class 1 --expected-num-classes 4 \
  --world-size 1 --batch-size 4 --base-seed 100000 --require-sidecar

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/validate_manifest.py" \
  --manifest "$OUTPUT_ROOT/manifests/search_1k.jsonl" \
  --expected-count 1000 --expected-per-class 1 --expected-num-classes 1000 \
  --world-size 4 --batch-size 4 --base-seed 1100000 --require-sidecar \
  --disjoint-with "$OUTPUT_ROOT/manifests/smoke_4.jsonl"

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/validate_manifest.py" \
  --manifest "$OUTPUT_ROOT/manifests/validation_8k.jsonl" \
  --expected-count 8000 --expected-per-class 8 --expected-num-classes 1000 \
  --world-size 4 --batch-size 4 --base-seed 2100000 --require-sidecar \
  --disjoint-with "$OUTPUT_ROOT/manifests/smoke_4.jsonl" \
  --disjoint-with "$OUTPUT_ROOT/manifests/search_1k.jsonl"

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/validate_manifest.py" \
  --manifest "$MANIFEST" \
  --expected-count 50000 --expected-per-class 50 --expected-num-classes 1000 \
  --world-size 4 --batch-size 4 --base-seed 3100000 --require-sidecar \
  --disjoint-with "$OUTPUT_ROOT/manifests/smoke_4.jsonl" \
  --disjoint-with "$OUTPUT_ROOT/manifests/search_1k.jsonl" \
  --disjoint-with "$OUTPUT_ROOT/manifests/validation_8k.jsonl"
```

Before seeing 1K results, preregister one deterministic rule that names the quality constraints, invalid-run rules, latency statistic, tie-breaker, and the fact that the 8K split chooses the operating point while 50K is evaluation-only. This text is later embedded in the signed selection-decision artifact.

```bash
: "${SELECTION_RULE:?write the complete deterministic rule before running 1K}"
test ! -e "$OUTPUT_ROOT/selection/selection_rule.txt"
printf '%s\n' "$SELECTION_RULE" > "$OUTPUT_ROOT/selection/selection_rule.txt"
```

Do not amend this file after observing 1K, 8K, or 50K evidence. A changed rule is a new experiment.

## 3. CPU cache-memory estimate

```bash
CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/estimate_cache_memory.py" \
  --preset pixelgen-xl-256 --batch-size 4 --probe-depth 1 \
  --cache-dtype bfloat16 \
  --output-json "$OUTPUT_ROOT/metrics/cache_estimate.json"
```

This estimate is planning evidence only. The single-GPU benchmark and compile matrix record measured allocated/reserved peaks.

## 4. One three-config smoke run

Choose a provisional smoke-only threshold and policy; neither assignment is a final operating point. Every materialization is bound to a matching `pixarc-dicache-selection-v1` report. The inactive Full/upstream policy is `official_propagate` because that is the immutable base-config value, not a candidate-policy selection.

```bash
: "${SMOKE_REL_L1_THRESHOLD:?provisional smoke-only threshold}"
: "${SMOKE_GAMMA_NONFINITE_POLICY:?provisional smoke-only gamma policy}"

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/record_selection.py" \
  --model-family PixelGen --status provisional --probe-depth 1 \
  --gamma-nonfinite-policy official_propagate \
  --output "$OUTPUT_ROOT/selection/full_provisional.json"

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/record_selection.py" \
  --model-family PixelGen --status provisional --probe-depth 1 \
  --rel-l1-thresh "$SMOKE_REL_L1_THRESHOLD" \
  --gamma-nonfinite-policy "$SMOKE_GAMMA_NONFINITE_POLICY" \
  --output "$OUTPUT_ROOT/selection/smoke_candidate_provisional.json"

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/materialize_dicache_config.py" \
  --base "$BASELINE_ROOT/configs/pixelgen_xl_256_upstream_full.yaml" \
  --checkpoint "$CHECKPOINT" \
  --selection-report "$OUTPUT_ROOT/selection/full_provisional.json" \
  --output "$OUTPUT_ROOT/configs/smoke_upstream_full.yaml"

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/materialize_dicache_config.py" \
  --base "$BASELINE_ROOT/configs/pixelgen_xl_256_instrumented_full.yaml" \
  --checkpoint "$CHECKPOINT" \
  --selection-report "$OUTPUT_ROOT/selection/full_provisional.json" \
  --output "$OUTPUT_ROOT/configs/smoke_instrumented_full.yaml"

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/materialize_dicache_config.py" \
  --base "$BASELINE_ROOT/configs/pixelgen_xl_256_dicache.yaml" \
  --checkpoint "$CHECKPOINT" --threshold "$SMOKE_REL_L1_THRESHOLD" \
  --gamma-nonfinite-policy "$SMOKE_GAMMA_NONFINITE_POLICY" \
  --selection-report "$OUTPUT_ROOT/selection/smoke_candidate_provisional.json" \
  --output "$OUTPUT_ROOT/configs/smoke_dicache_provisional.yaml"

DICACHE_GPU_TESTS_ALLOWED=1 CUDA_VISIBLE_DEVICES="$SINGLE_GPU_VISIBLE_DEVICE" \
  bash "$BASELINE_ROOT/scripts/run_deferred_smoke_tests.sh" \
  --upstream-config "$OUTPUT_ROOT/configs/smoke_upstream_full.yaml" \
  --full-config "$OUTPUT_ROOT/configs/smoke_instrumented_full.yaml" \
  --candidate-config "$OUTPUT_ROOT/configs/smoke_dicache_provisional.yaml" \
  --manifest "$OUTPUT_ROOT/manifests/smoke_4.jsonl" \
  --output-root "$OUTPUT_ROOT/smoke_main"
```

The one wrapper invocation must create and consume all three real hard-gate artifacts: tensor/resume parity, upstream-versus-instrumented PNG parity, and the bound smoke gate. Revalidate them with CUDA hidden rather than constructing substitute reports:

```bash
CUDA_VISIBLE_DEVICES='' python -c 'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); i=json.load(open(sys.argv[2],encoding="utf-8")); s=json.load(open(sys.argv[3],encoding="utf-8")); assert p.get("passed") is True and len(p.get("nine_resume_invariants",{}))==9 and all(p["nine_resume_invariants"].values()); assert i.get("exact") is True and i.get("differing_image_count")==0; assert s.get("passed") is True' \
  "$OUTPUT_ROOT/smoke_main/model_parity.json" \
  "$OUTPUT_ROOT/smoke_main/upstream_vs_instrumented_png.json" \
  "$OUTPUT_ROOT/smoke_main/smoke_gate.json"
```

After reviewing smoke-only nonfinite/fallback behavior, freeze exactly one candidate gamma policy before 1K. Also preregister the complete coarse grid as whitespace-separated safe `tag=value` pairs, for example a tag may be `t_coarse_a`; the values themselves are experiment inputs and are not supplied here.

```bash
: "${GAMMA_NONFINITE_POLICY:?select using smoke evidence only}"
: "${COARSE_CANDIDATES:?whitespace-separated safe tag=value pairs}"
case "$GAMMA_NONFINITE_POLICY" in
  official_propagate|latest_residual_fallback|force_full) ;;
  *) echo "invalid GAMMA_NONFINITE_POLICY" >&2; exit 2 ;;
esac
test ! -e "$OUTPUT_ROOT/selection/gamma_policy.txt"
test ! -e "$OUTPUT_ROOT/selection/coarse_candidates.txt"
printf '%s\n' "$GAMMA_NONFINITE_POLICY" > "$OUTPUT_ROOT/selection/gamma_policy.txt"
printf '%s\n' "$COARSE_CANDIDATES" > "$OUTPUT_ROOT/selection/coarse_candidates.txt"
```

Do not change the frozen policy during 1K, 8K, benchmark, or 50K.

## 5. Full parity gate

Do not launch a second smoke workflow. Consume the exact upstream-versus-instrumented result produced in stage 4:

```bash
CUDA_VISIBLE_DEVICES='' python -c 'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); assert r.get("schema_version")=="pixarc-image-tree-parity-v1"; assert r.get("sample_count")==4; assert r.get("exact") is True; assert r.get("differing_image_count")==0; assert r.get("max_absolute_uint8_error")==0' \
  "$OUTPUT_ROOT/smoke_main/upstream_vs_instrumented_png.json"
```

The wrapper already compared the real upstream generator route with `instrumented_full`. Any difference blocks every later stage.

## 6. Probe-then-resume parity gate

Again consume the stage-4 model report; do not replace it with a PNG-only or toy-module comparison. `run_gpu_model_parity.py` derives the expected NFE and combined-forward counts from the configured sampler and checks the nine block/context/RoPE/probe/head/anchor invariants.

```bash
CUDA_VISIBLE_DEVICES='' python -c 'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); n=r.get("nine_resume_invariants",{}); o=r.get("operational_invariants",{}); assert r.get("schema_version")=="pixarc-pixelgen-dicache-resume-parity-v1" and r.get("passed") is True; assert len(n)==9 and all(v is True for v in n.values()); assert o and all(v is True for v in o.values()); assert r.get("body_allclose") is True and r.get("probe_feature_allclose") is True and r.get("final_sample_allclose") is True and r.get("decoded_image_allclose") is True; assert int(r.get("expected_nfe",0))>0 and int(r.get("expected_combined_forwards",0))>0' \
  "$OUTPUT_ROOT/smoke_main/model_parity.json"
```

Failure of any single invariant, raw-finiteness check, reset check, or cache-release check blocks search.

## 7. Shadow diagnostic and explicit depth ablations

Choose one diagnostic-only threshold. Materialize a distinct provisional selection report for every depth; depth 2/3 are cost/correlation ablations and never become the main `released_code_faithful_image_profile` candidate.

```bash
: "${SHADOW_REL_L1_THRESHOLD:?diagnostic-only shadow threshold}"

for DEPTH in 1 2 3; do
  CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/record_selection.py" \
    --model-family PixelGen --status provisional --probe-depth "$DEPTH" \
    --rel-l1-thresh "$SHADOW_REL_L1_THRESHOLD" \
    --gamma-nonfinite-policy "$GAMMA_NONFINITE_POLICY" \
    --output "$OUTPUT_ROOT/selection/shadow_depth_${DEPTH}_provisional.json"

  CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/materialize_dicache_config.py" \
    --base "$BASELINE_ROOT/configs/pixelgen_xl_256_probe_shadow_full.yaml" \
    --checkpoint "$CHECKPOINT" --threshold "$SHADOW_REL_L1_THRESHOLD" \
    --gamma-nonfinite-policy "$GAMMA_NONFINITE_POLICY" --probe-depth "$DEPTH" \
    --selection-report "$OUTPUT_ROOT/selection/shadow_depth_${DEPTH}_provisional.json" \
    --output "$OUTPUT_ROOT/configs/shadow_depth_${DEPTH}.yaml"

  DICACHE_GPU_TESTS_ALLOWED=1 CUDA_VISIBLE_DEVICES="$SINGLE_GPU_VISIBLE_DEVICE" \
    bash "$BASELINE_ROOT/scripts/run_shadow_diagnostic.sh" \
    --config "$OUTPUT_ROOT/configs/shadow_depth_${DEPTH}.yaml" \
    --manifest "$OUTPUT_ROOT/manifests/smoke_4.jsonl" \
    --output-root "$OUTPUT_ROOT/shadow/depth_${DEPTH}"
done
```

Keep `shadow_diagnostics.json` for each depth. Only depth 1 informs the main profile; the 8K/50K configurations remain depth 1 regardless of depth-ablation results.

## 8. Disjoint 1K coarse search

Generate the matched instrumented Full once. Then, for every preregistered `tag=value` candidate, create a matching provisional report/config, run four-GPU generation, strict paired metrics, trace aggregation, and a real batch-4 single-GPU CUDA-event Full/DiCache pair benchmark. No candidate is allowed to skip a column of evidence.

```bash
test "$(cat "$OUTPUT_ROOT/selection/gamma_policy.txt")" = "$GAMMA_NONFINITE_POLICY"
test "$(cat "$OUTPUT_ROOT/selection/coarse_candidates.txt")" = "$COARSE_CANDIDATES"

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/materialize_dicache_config.py" \
  --base "$BASELINE_ROOT/configs/pixelgen_xl_256_instrumented_full.yaml" \
  --checkpoint "$CHECKPOINT" \
  --selection-report "$OUTPUT_ROOT/selection/full_provisional.json" \
  --output "$OUTPUT_ROOT/configs/search_full.yaml"

DICACHE_GPU_TESTS_ALLOWED=1 CUDA_VISIBLE_DEVICES="$FOUR_GPU_VISIBLE_DEVICES" \
  bash "$BASELINE_ROOT/scripts/launch_4gpu_50k.sh" \
  --config "$OUTPUT_ROOT/configs/search_full.yaml" \
  --manifest "$OUTPUT_ROOT/manifests/search_1k.jsonl" \
  --output-root "$OUTPUT_ROOT/search_1k/full" --nonfinal-proxy \
  --expected-records 1000 --expected-per-class 1 --expected-num-classes 1000

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/aggregate_dicache_trace.py" \
  --metadata-dir "$OUTPUT_ROOT/search_1k/full/metadata" --world-size 4 \
  --output-json "$OUTPUT_ROOT/metrics/search_full_trace.json"

for SPEC in $COARSE_CANDIDATES; do
  TAG="${SPEC%%=*}"
  REL_L1_THRESHOLD="${SPEC#*=}"
  [[ "$SPEC" == *=* && "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2

  CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/record_selection.py" \
    --model-family PixelGen --status provisional --probe-depth 1 \
    --rel-l1-thresh "$REL_L1_THRESHOLD" \
    --gamma-nonfinite-policy "$GAMMA_NONFINITE_POLICY" \
    --output "$OUTPUT_ROOT/selection/search_${TAG}_provisional.json"

  CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/materialize_dicache_config.py" \
    --base "$BASELINE_ROOT/configs/pixelgen_xl_256_dicache.yaml" \
    --checkpoint "$CHECKPOINT" --threshold "$REL_L1_THRESHOLD" \
    --gamma-nonfinite-policy "$GAMMA_NONFINITE_POLICY" \
    --selection-report "$OUTPUT_ROOT/selection/search_${TAG}_provisional.json" \
    --output "$OUTPUT_ROOT/configs/search_dicache_${TAG}.yaml"

  DICACHE_GPU_TESTS_ALLOWED=1 CUDA_VISIBLE_DEVICES="$FOUR_GPU_VISIBLE_DEVICES" \
    bash "$BASELINE_ROOT/scripts/launch_4gpu_50k.sh" \
    --config "$OUTPUT_ROOT/configs/search_dicache_${TAG}.yaml" \
    --manifest "$OUTPUT_ROOT/manifests/search_1k.jsonl" \
    --output-root "$OUTPUT_ROOT/search_1k/dicache_${TAG}" --nonfinal-proxy \
    --expected-records 1000 --expected-per-class 1 --expected-num-classes 1000

  CUDA_VISIBLE_DEVICES='' bash "$BASELINE_ROOT/scripts/evaluate_paired.sh" \
    --reference-dir "$OUTPUT_ROOT/search_1k/full/samples" \
    --candidate-dir "$OUTPUT_ROOT/search_1k/dicache_${TAG}/samples" \
    --reference-manifest "$OUTPUT_ROOT/manifests/search_1k.jsonl" \
    --candidate-manifest "$OUTPUT_ROOT/manifests/search_1k.jsonl" \
    --reference-run-metadata "$OUTPUT_ROOT/search_1k/full/run_manifest.json" \
    --candidate-run-metadata "$OUTPUT_ROOT/search_1k/dicache_${TAG}/run_manifest.json" \
    --expected-count 1000 --expected-per-class 1 --expected-num-classes 1000 \
    --output-json "$OUTPUT_ROOT/metrics/search_${TAG}_paired.json" \
    --output-csv "$OUTPUT_ROOT/metrics/search_${TAG}_paired.csv" \
    --lpips-device cpu --lpips-batch-size 16

  CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/aggregate_dicache_trace.py" \
    --metadata-dir "$OUTPUT_ROOT/search_1k/dicache_${TAG}/metadata" --world-size 4 \
    --output-json "$OUTPUT_ROOT/metrics/search_${TAG}_trace.json"

  CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/build_benchmark_runner.py" \
    --model-config "$OUTPUT_ROOT/configs/search_dicache_${TAG}.yaml" \
    --manifest "$OUTPUT_ROOT/manifests/search_1k.jsonl" \
    --batch-size 4 --output "$OUTPUT_ROOT/benchmark/search_${TAG}.runner.json"

  DICACHE_GPU_TESTS_ALLOWED=1 CUDA_VISIBLE_DEVICES="$SINGLE_GPU_VISIBLE_DEVICE" \
    bash "$BASELINE_ROOT/scripts/benchmark_single_gpu.sh" \
    --runner-factory dicache_style.pixelgen_benchmark:build_benchmark_spec \
    --runner-config "$OUTPUT_ROOT/benchmark/search_${TAG}.runner.json" \
    --output-json "$OUTPUT_ROOT/metrics/search_${TAG}_benchmark.json" \
    --warmup-batches 10 --measured-batches 30
done
```

Use 1K only to define a fine grid. Before any 8K run, freeze that grid as safe `tag=value` pairs; never choose the final operating point on 1K.

```bash
: "${FINE_CANDIDATES:?whitespace-separated fine-grid tag=value pairs chosen from 1K only}"
test ! -e "$OUTPUT_ROOT/selection/fine_candidates.txt"
printf '%s\n' "$FINE_CANDIDATES" > "$OUTPUT_ROOT/selection/fine_candidates.txt"
```

## 9. Disjoint 8K validation and immutable selection

Generate matched Full once. Every fine candidate must receive the same four evidence products as in 1K. Apply the stage-2 rule only after all candidate artifacts exist. The selected decision binds the exact 8K paired, trace, and matched benchmark JSONs by SHA-256; the selected report then binds that decision; final configs bind the selected report.

```bash
test "$(cat "$OUTPUT_ROOT/selection/fine_candidates.txt")" = "$FINE_CANDIDATES"

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/materialize_dicache_config.py" \
  --base "$BASELINE_ROOT/configs/pixelgen_xl_256_instrumented_full.yaml" \
  --checkpoint "$CHECKPOINT" \
  --selection-report "$OUTPUT_ROOT/selection/full_provisional.json" \
  --output "$OUTPUT_ROOT/configs/validation_full.yaml"

DICACHE_GPU_TESTS_ALLOWED=1 CUDA_VISIBLE_DEVICES="$FOUR_GPU_VISIBLE_DEVICES" \
  bash "$BASELINE_ROOT/scripts/launch_4gpu_50k.sh" \
  --config "$OUTPUT_ROOT/configs/validation_full.yaml" \
  --manifest "$OUTPUT_ROOT/manifests/validation_8k.jsonl" \
  --output-root "$OUTPUT_ROOT/validation_8k/full" --nonfinal-proxy \
  --expected-records 8000 --expected-per-class 8 --expected-num-classes 1000

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/aggregate_dicache_trace.py" \
  --metadata-dir "$OUTPUT_ROOT/validation_8k/full/metadata" --world-size 4 \
  --output-json "$OUTPUT_ROOT/metrics/validation_full_trace.json"

for SPEC in $FINE_CANDIDATES; do
  TAG="${SPEC%%=*}"
  REL_L1_THRESHOLD="${SPEC#*=}"
  [[ "$SPEC" == *=* && "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2

  CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/record_selection.py" \
    --model-family PixelGen --status provisional --probe-depth 1 \
    --rel-l1-thresh "$REL_L1_THRESHOLD" \
    --gamma-nonfinite-policy "$GAMMA_NONFINITE_POLICY" \
    --output "$OUTPUT_ROOT/selection/validation_${TAG}_provisional.json"

  CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/materialize_dicache_config.py" \
    --base "$BASELINE_ROOT/configs/pixelgen_xl_256_dicache.yaml" \
    --checkpoint "$CHECKPOINT" --threshold "$REL_L1_THRESHOLD" \
    --gamma-nonfinite-policy "$GAMMA_NONFINITE_POLICY" \
    --selection-report "$OUTPUT_ROOT/selection/validation_${TAG}_provisional.json" \
    --output "$OUTPUT_ROOT/configs/validation_dicache_${TAG}.yaml"

  DICACHE_GPU_TESTS_ALLOWED=1 CUDA_VISIBLE_DEVICES="$FOUR_GPU_VISIBLE_DEVICES" \
    bash "$BASELINE_ROOT/scripts/launch_4gpu_50k.sh" \
    --config "$OUTPUT_ROOT/configs/validation_dicache_${TAG}.yaml" \
    --manifest "$OUTPUT_ROOT/manifests/validation_8k.jsonl" \
    --output-root "$OUTPUT_ROOT/validation_8k/dicache_${TAG}" --nonfinal-proxy \
    --expected-records 8000 --expected-per-class 8 --expected-num-classes 1000

  CUDA_VISIBLE_DEVICES='' bash "$BASELINE_ROOT/scripts/evaluate_paired.sh" \
    --reference-dir "$OUTPUT_ROOT/validation_8k/full/samples" \
    --candidate-dir "$OUTPUT_ROOT/validation_8k/dicache_${TAG}/samples" \
    --reference-manifest "$OUTPUT_ROOT/manifests/validation_8k.jsonl" \
    --candidate-manifest "$OUTPUT_ROOT/manifests/validation_8k.jsonl" \
    --reference-run-metadata "$OUTPUT_ROOT/validation_8k/full/run_manifest.json" \
    --candidate-run-metadata "$OUTPUT_ROOT/validation_8k/dicache_${TAG}/run_manifest.json" \
    --expected-count 8000 --expected-per-class 8 --expected-num-classes 1000 \
    --output-json "$OUTPUT_ROOT/metrics/validation_${TAG}_paired.json" \
    --output-csv "$OUTPUT_ROOT/metrics/validation_${TAG}_paired.csv" \
    --lpips-device cpu --lpips-batch-size 16

  CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/aggregate_dicache_trace.py" \
    --metadata-dir "$OUTPUT_ROOT/validation_8k/dicache_${TAG}/metadata" --world-size 4 \
    --output-json "$OUTPUT_ROOT/metrics/validation_${TAG}_trace.json"

  CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/build_benchmark_runner.py" \
    --model-config "$OUTPUT_ROOT/configs/validation_dicache_${TAG}.yaml" \
    --manifest "$OUTPUT_ROOT/manifests/validation_8k.jsonl" \
    --batch-size 4 --output "$OUTPUT_ROOT/benchmark/validation_${TAG}.runner.json"

  DICACHE_GPU_TESTS_ALLOWED=1 CUDA_VISIBLE_DEVICES="$SINGLE_GPU_VISIBLE_DEVICE" \
    bash "$BASELINE_ROOT/scripts/benchmark_single_gpu.sh" \
    --runner-factory dicache_style.pixelgen_benchmark:build_benchmark_spec \
    --runner-config "$OUTPUT_ROOT/benchmark/validation_${TAG}.runner.json" \
    --output-json "$OUTPUT_ROOT/metrics/validation_${TAG}_benchmark.json" \
    --warmup-batches 10 --measured-batches 30
done
```

Apply the frozen rule without looking at 50K. Set both selected variables from the winning 8K artifact; `SELECTED_TAG` must name the matching entry in `FINE_CANDIDATES`.

```bash
: "${SELECTED_TAG:?tag selected by the preregistered 8K rule}"
: "${SELECTED_REL_L1_THRESHOLD:?threshold selected by the preregistered 8K rule}"
case " $FINE_CANDIDATES " in
  *" $SELECTED_TAG=$SELECTED_REL_L1_THRESHOLD "*) ;;
  *) echo "selected tag/value is not in the frozen fine grid" >&2; exit 2 ;;
esac

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/record_selection_decision.py" \
  --model-family PixelGen \
  --rel-l1-thresh "$SELECTED_REL_L1_THRESHOLD" \
  --gamma-nonfinite-policy "$GAMMA_NONFINITE_POLICY" \
  --selection-rule "$(cat "$OUTPUT_ROOT/selection/selection_rule.txt")" \
  --paired-report "$OUTPUT_ROOT/metrics/validation_${SELECTED_TAG}_paired.json" \
  --trace-report "$OUTPUT_ROOT/metrics/validation_${SELECTED_TAG}_trace.json" \
  --benchmark-report "$OUTPUT_ROOT/metrics/validation_${SELECTED_TAG}_benchmark.json" \
  --output "$OUTPUT_ROOT/selection/selection_decision.json"

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/record_selection.py" \
  --model-family PixelGen --status selected --probe-depth 1 \
  --rel-l1-thresh "$SELECTED_REL_L1_THRESHOLD" \
  --gamma-nonfinite-policy "$GAMMA_NONFINITE_POLICY" \
  --decision-report "$OUTPUT_ROOT/selection/selection_decision.json" \
  --output "$OUTPUT_ROOT/selection/selected.json"

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/materialize_dicache_config.py" \
  --base "$BASELINE_ROOT/configs/pixelgen_xl_256_dicache.yaml" \
  --checkpoint "$CHECKPOINT" --threshold "$SELECTED_REL_L1_THRESHOLD" \
  --gamma-nonfinite-policy "$GAMMA_NONFINITE_POLICY" \
  --selection-report "$OUTPUT_ROOT/selection/selected.json" \
  --output "$OUTPUT_ROOT/configs/final_dicache.yaml"

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/materialize_dicache_config.py" \
  --base "$BASELINE_ROOT/configs/pixelgen_xl_256_instrumented_full.yaml" \
  --checkpoint "$CHECKPOINT" \
  --selection-report "$OUTPUT_ROOT/selection/full_provisional.json" \
  --output "$OUTPUT_ROOT/configs/final_full.yaml"
```

The selected threshold and policy are now immutable. Do not tune either from stage 10 or from the final 50K.

## 10. Repeated matched benchmark and five-row compile matrix

First run the final `matched_eager` Full/DiCache benchmark repeatedly on the same single allocated device. Each report records CUDA-event latency, first-execution upper bound, cache bytes, peak allocated/reserved memory, graph breaks, and guard failures.

```bash
: "${BENCHMARK_REPEATS:?integer repeat count of at least two}"
case "$BENCHMARK_REPEATS" in ''|*[!0-9]*) exit 2 ;; esac
test "$BENCHMARK_REPEATS" -ge 2

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/build_benchmark_runner.py" \
  --model-config "$OUTPUT_ROOT/configs/final_dicache.yaml" \
  --manifest "$MANIFEST" --batch-size 4 \
  --output "$OUTPUT_ROOT/benchmark/final_matched.runner.json"

for REP in $(seq 1 "$BENCHMARK_REPEATS"); do
  DICACHE_GPU_TESTS_ALLOWED=1 CUDA_VISIBLE_DEVICES="$SINGLE_GPU_VISIBLE_DEVICE" \
    bash "$BASELINE_ROOT/scripts/benchmark_single_gpu.sh" \
    --runner-factory dicache_style.pixelgen_benchmark:build_benchmark_spec \
    --runner-config "$OUTPUT_ROOT/benchmark/final_matched.runner.json" \
    --output-json "$OUTPUT_ROOT/metrics/final_matched_repeat_${REP}.json" \
    --warmup-batches 10 --measured-batches 30
done
```

The compile comparison has exactly five isolated rows:

| Row | DiCache mode | Compile mode |
| --- | --- | --- |
| `upstream_whole_model` | `upstream_full` | `upstream` |
| `matched_eager_full` | `instrumented_full` | `matched_eager` |
| `matched_eager_dicache` | `dicache` | `matched_eager` |
| `blockwise_full` | `instrumented_full` | `blockwise` |
| `blockwise_dicache` | `dicache` | `blockwise` |

Materialize all five with matching provisional/selected provenance, then run the orchestrator. It launches five fresh subprocesses, keeps `TORCH_LOGS=graph_breaks,recompiles`, checks raw finiteness and cache lifecycle, and fails on any non-byte-exact required output comparison.

```bash
CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/materialize_dicache_config.py" \
  --base "$BASELINE_ROOT/configs/pixelgen_xl_256_upstream_full.yaml" \
  --checkpoint "$CHECKPOINT" --compile-mode upstream \
  --selection-report "$OUTPUT_ROOT/selection/full_provisional.json" \
  --output "$OUTPUT_ROOT/configs/compile_upstream_whole_model.yaml"

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/materialize_dicache_config.py" \
  --base "$BASELINE_ROOT/configs/pixelgen_xl_256_instrumented_full.yaml" \
  --checkpoint "$CHECKPOINT" --compile-mode matched_eager \
  --selection-report "$OUTPUT_ROOT/selection/full_provisional.json" \
  --output "$OUTPUT_ROOT/configs/compile_matched_eager_full.yaml"

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/materialize_dicache_config.py" \
  --base "$BASELINE_ROOT/configs/pixelgen_xl_256_dicache.yaml" \
  --checkpoint "$CHECKPOINT" --compile-mode matched_eager \
  --threshold "$SELECTED_REL_L1_THRESHOLD" \
  --gamma-nonfinite-policy "$GAMMA_NONFINITE_POLICY" \
  --selection-report "$OUTPUT_ROOT/selection/selected.json" \
  --output "$OUTPUT_ROOT/configs/compile_matched_eager_dicache.yaml"

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/materialize_dicache_config.py" \
  --base "$BASELINE_ROOT/configs/pixelgen_xl_256_instrumented_full.yaml" \
  --checkpoint "$CHECKPOINT" --compile-mode blockwise \
  --selection-report "$OUTPUT_ROOT/selection/full_provisional.json" \
  --output "$OUTPUT_ROOT/configs/compile_blockwise_full.yaml"

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/materialize_dicache_config.py" \
  --base "$BASELINE_ROOT/configs/pixelgen_xl_256_dicache.yaml" \
  --checkpoint "$CHECKPOINT" --compile-mode blockwise \
  --threshold "$SELECTED_REL_L1_THRESHOLD" \
  --gamma-nonfinite-policy "$GAMMA_NONFINITE_POLICY" \
  --selection-report "$OUTPUT_ROOT/selection/selected.json" \
  --output "$OUTPUT_ROOT/configs/compile_blockwise_dicache.yaml"

DICACHE_GPU_TESTS_ALLOWED=1 CUDA_VISIBLE_DEVICES="$SINGLE_GPU_VISIBLE_DEVICE" \
  python "$BASELINE_ROOT/scripts/run_compile_matrix.py" \
  --upstream-whole-model-config "$OUTPUT_ROOT/configs/compile_upstream_whole_model.yaml" \
  --matched-eager-full-config "$OUTPUT_ROOT/configs/compile_matched_eager_full.yaml" \
  --matched-eager-dicache-config "$OUTPUT_ROOT/configs/compile_matched_eager_dicache.yaml" \
  --blockwise-full-config "$OUTPUT_ROOT/configs/compile_blockwise_full.yaml" \
  --blockwise-dicache-config "$OUTPUT_ROOT/configs/compile_blockwise_dicache.yaml" \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_ROOT/benchmark/compile_matrix_rows" \
  --output-json "$OUTPUT_ROOT/metrics/compile_matrix.json" \
  --warmup-batches 10 --measured-batches 30
```

The primary speedup remains matched eager Full versus matched eager DiCache. Upstream whole-model and blockwise rows are supplemental and cannot replace that comparison.

## 11. Release-gated four-GPU 50K

Create the release gate before either 50K launch. `release_gate.py create` hashes both final configs, the final manifest and its `.meta.json` sidecar, the selected report and bound 8K decision, stage-4 resume parity, the stage-4 smoke gate, the five-row compile matrix, and the exact port/upstream executable/config source bytes. Smoke and compile reports must carry the same source binding, so an edit between evidence stages fails closed.

```bash
test -f "$MANIFEST.meta.json"
CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/release_gate.py" create \
  --model-family PixelGen \
  --full-config "$OUTPUT_ROOT/configs/final_full.yaml" \
  --candidate-config "$OUTPUT_ROOT/configs/final_dicache.yaml" \
  --manifest "$MANIFEST" \
  --selection-report "$OUTPUT_ROOT/selection/selected.json" \
  --parity-report "$OUTPUT_ROOT/smoke_main/model_parity.json" \
  --smoke-report "$OUTPUT_ROOT/smoke_main/smoke_gate.json" \
  --compile-report "$OUTPUT_ROOT/metrics/compile_matrix.json" \
  --output "$OUTPUT_ROOT/selection/release_gate.json"

DICACHE_GPU_TESTS_ALLOWED=1 CUDA_VISIBLE_DEVICES="$FOUR_GPU_VISIBLE_DEVICES" \
  bash "$BASELINE_ROOT/scripts/launch_4gpu_50k.sh" \
  --config "$OUTPUT_ROOT/configs/final_full.yaml" \
  --manifest "$MANIFEST" \
  --output-root "$OUTPUT_ROOT/final_50k/full" \
  --release-gate "$OUTPUT_ROOT/selection/release_gate.json"

DICACHE_GPU_TESTS_ALLOWED=1 CUDA_VISIBLE_DEVICES="$FOUR_GPU_VISIBLE_DEVICES" \
  bash "$BASELINE_ROOT/scripts/launch_4gpu_50k.sh" \
  --config "$OUTPUT_ROOT/configs/final_dicache.yaml" \
  --manifest "$MANIFEST" \
  --output-root "$OUTPUT_ROOT/final_50k/dicache" \
  --release-gate "$OUTPUT_ROOT/selection/release_gate.json"
```

The launcher revalidates the gate before CUDA access and archives that exact gate beside the output prefix. Use `--resume` only with the identical archived config, manifest, sidecar, checkpoint, release gate, and fixed batch groups; a durable prefix without the same archived gate is rejected. Fifty-thousand-sample outputs are evaluation-only: never use them to revise the threshold, policy, rule, or selected report.

## 12. Output validation

```bash
for METHOD in full dicache; do
  CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/validate_outputs.py" \
    --sample-dir "$OUTPUT_ROOT/final_50k/$METHOD/samples" \
    --manifest "$MANIFEST" \
    --metadata-dir "$OUTPUT_ROOT/final_50k/$METHOD/metadata" \
    --run-metadata "$OUTPUT_ROOT/final_50k/$METHOD/run_manifest.json" \
    --expected-count 50000 --expected-per-class 50 \
    --expected-num-classes 1000 --resolution 256
done
```

Both trees must contain exactly the manifest IDs, RGB `uint8` 256×256 PNGs, finite trajectory metadata, valid call counts, the expected checkpoint/config/manifest identities, and no duplicate or missing samples.

## 13. Distribution metrics

Use only the frozen local ADM evaluator and reference NPZ named at the top; the wrappers never download a replacement.

```bash
for METHOD in full dicache; do
  CUDA_VISIBLE_DEVICES='' bash "$BASELINE_ROOT/scripts/evaluate_distribution.sh" \
    --sample-dir "$OUTPUT_ROOT/final_50k/$METHOD/samples" \
    --manifest "$MANIFEST" \
    --reference-npz "$IMAGENET_REFERENCE_NPZ" \
    --evaluator "$ADM_EVALUATOR" \
    --run-metadata "$OUTPUT_ROOT/final_50k/$METHOD/run_manifest.json" \
    --output-json "$OUTPUT_ROOT/metrics/${METHOD}_distribution.json" \
    --sample-npz "$OUTPUT_ROOT/metrics/${METHOD}_samples.npz" \
    --expected-count 50000 --expected-per-class 50 --expected-num-classes 1000
done

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/compare_distribution.py" \
  --full-json "$OUTPUT_ROOT/metrics/full_distribution.json" \
  --dicache-json "$OUTPUT_ROOT/metrics/dicache_distribution.json" \
  --output-json "$OUTPUT_ROOT/metrics/distribution_delta.json"
```

Report FID, sFID, IS, precision, recall, evaluator identity, and the Full-minus/DiCache deltas without using distribution metrics for post-hoc tuning.

## 14. Strict paired metrics

Pair only the new release-gated Full/DiCache trees generated from the identical 50K manifest. Never pair against a legacy Full tree.

```bash
CUDA_VISIBLE_DEVICES='' bash "$BASELINE_ROOT/scripts/evaluate_paired.sh" \
  --reference-dir "$OUTPUT_ROOT/final_50k/full/samples" \
  --candidate-dir "$OUTPUT_ROOT/final_50k/dicache/samples" \
  --reference-manifest "$MANIFEST" --candidate-manifest "$MANIFEST" \
  --reference-run-metadata "$OUTPUT_ROOT/final_50k/full/run_manifest.json" \
  --candidate-run-metadata "$OUTPUT_ROOT/final_50k/dicache/run_manifest.json" \
  --output-json "$OUTPUT_ROOT/metrics/paired.json" \
  --output-csv "$OUTPUT_ROOT/metrics/paired.csv" \
  --expected-count 50000 --expected-per-class 50 --expected-num-classes 1000 \
  --lpips-device cpu --lpips-batch-size 16
```

If LPIPS is unavailable, rerun with `--skip-lpips` and label the paired artifact incomplete; do not silently substitute another perceptual model.

## 15. Trace, latency/memory, compile, and wall-clock aggregation

Persist independent Full and DiCache trajectory summaries, then combine all repeated matched benchmark reports into one latency/peak-memory artifact.

```bash
CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/aggregate_dicache_trace.py" \
  --metadata-dir "$OUTPUT_ROOT/final_50k/full/metadata" --world-size 4 \
  --output-json "$OUTPUT_ROOT/metrics/full_trace.json"

CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/aggregate_dicache_trace.py" \
  --metadata-dir "$OUTPUT_ROOT/final_50k/dicache/metadata" --world-size 4 \
  --output-json "$OUTPUT_ROOT/metrics/dicache_trace.json"

BENCHMARK_INPUT_ARGS=()
for REP in $(seq 1 "$BENCHMARK_REPEATS"); do
  REPORT="$OUTPUT_ROOT/metrics/final_matched_repeat_${REP}.json"
  test -f "$REPORT"
  BENCHMARK_INPUT_ARGS+=(--input "$REPORT")
done
CUDA_VISIBLE_DEVICES='' python "$BASELINE_ROOT/scripts/aggregate_benchmark_reports.py" \
  "${BENCHMARK_INPUT_ARGS[@]}" \
  --output "$OUTPUT_ROOT/metrics/final_matched_benchmark_aggregate.json"
```

Retain the compile report and every row/log in place, and recheck its release-gate status:

```bash
CUDA_VISIBLE_DEVICES='' python -c 'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); assert r.get("schema_version")=="pixarc-dicache-compile-matrix-v1" and r.get("passed") is True; assert len(r.get("matrix",{}))==5' \
  "$OUTPUT_ROOT/metrics/compile_matrix.json"
test -d "$OUTPUT_ROOT/benchmark/compile_matrix_rows"
```

Finally preserve both launcher wall-clock summaries beside the metric artifacts without overwriting anything:

```bash
CUDA_VISIBLE_DEVICES='' python -c 'import json,sys; rows=[json.load(open(p,encoding="utf-8")) for p in sys.argv[1:]]; assert all(r.get("schema_version")=="pixarc-dicache-cumulative-wall-clock-v1" and r.get("completed") is True and r.get("invocation_chain_valid") is True and r.get("world_size")==4 and r.get("cumulative_sample_count")==50000 for r in rows)' \
  "$OUTPUT_ROOT/final_50k/full/four_gpu_wall_clock.json" \
  "$OUTPUT_ROOT/final_50k/dicache/four_gpu_wall_clock.json"
test ! -e "$OUTPUT_ROOT/metrics/full_four_gpu_wall_clock.json"
test ! -e "$OUTPUT_ROOT/metrics/dicache_four_gpu_wall_clock.json"
cp "$OUTPUT_ROOT/final_50k/full/four_gpu_wall_clock.json" \
  "$OUTPUT_ROOT/metrics/full_four_gpu_wall_clock.json"
cp "$OUTPUT_ROOT/final_50k/dicache/four_gpu_wall_clock.json" \
  "$OUTPUT_ROOT/metrics/dicache_four_gpu_wall_clock.json"
```

The final persistent evidence set is `full_trace.json`, `dicache_trace.json`, `final_matched_benchmark_aggregate.json`, `compile_matrix.json` plus its five row reports/logs, and the two copied wall-clock JSONs. Host component timers are diagnostic only; primary latency is the CUDA-event batch-4 per-image measurement. Each launcher JSON retains immutable per-invocation clocks, distinguishes summed active launcher time from first-start-to-final-end time including resume gaps, and must never present only the last resumed suffix as 50K throughput.
