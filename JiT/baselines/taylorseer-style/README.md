# Unofficial TaylorSeer-style port for JiT

This directory provides a research-only, branch-level TaylorSeer port for JiT
class-to-image inference. It is not an official TaylorSeer or JiT integration.
The primary implementation follows original TaylorSeer-DiT semantics; it is
not TaylorSeer-Lite and does not forecast a whole model/body residual.

> Safety snapshot, 2026-07-13 UTC: PixelGen Full 50K was active and the JiT
> Full 50K launcher was queued. This implementation started no CUDA workload.
> Every generation, compile, benchmark, FID, and LPIPS command below is
> deferred until the current jobs finish and the allocated GPUs are idle.

## Method contract

Every Transformer block owns separate `(layer,attn)` and `(layer,mlp)` finite
difference histories. Full calls compute and store the complete attention and
MLP branch outputs **before** their gates. Taylor calls recompute current
timestep/class embeddings, AdaLN modulation, and gates, then apply forecasts;
they skip norm1/attention/norm2/MLP. Only Full updates history. Final norm,
final AdaLN/projection, and unpatchify are always exact.

At exact anchors, signed recursive differences are:

```text
new D0=f(q); new D(k+1)=(new Dk-old Dk)/(q-newest_previous_q)
```

Forecast uses `sum(Dk*(q-anchor)^k/k!)`. `max_order=K` is the highest order,
so at most K+1 tensors mature over K+1 exact anchors. No forecast is written
back, and there is no clipping/damping/normalization.

The faithful fixed scheduler uses `first_enhance=2`; both first NFE are Full
and reset the counter. Afterwards there are `interval-1` Taylor calls before
the next Full. Interval 1 is all Full. The official source computes but does
not use `last_steps`, so `force_last_full=false` is the main default; enabling
it is an ablation.

## JiT and Heun mapping

JiT-B/16 inserts 32 class-context tokens before block 4, keeps them through
all remaining blocks, uses image/context-aware RoPE on the corresponding Full
blocks, and removes context once before the head. Per-layer state therefore
has 256 tokens before insertion and 288 after it.

One NFE consists of conditional then unconditional forwards plus CFG. Both
streams share the action and q but have independent factor tensors. For Heun
50:

```text
99 NFE decisions
99 conditional + 99 unconditional = 198 JiT forwards
q = 98,97,...,0
```

NFE index is used because a corrector and the next predictor can share
continuous t while receiving different states. See `HEUN_ADAPTATION.md`.

## Modes

- `upstream_full`: direct upstream forward; no scheduler/history.
- `instrumented_full`: local branch-split path, every NFE exact; parity and
  matched latency reference.
- `taylorseer`: fixed interval branch forecasts.
- `shadow_forecast`: still exact, additionally measures forecast error; never
  use for latency or 50K.

The provided Taylor YAML intentionally has null interval/order. Copy it to an
external immutable config after the independent 8K sweep and set both values;
the runner must reject unresolved nulls. No best setting is claimed.

## Environment

```bash
PIXARC_ROOT="$(git rev-parse --show-toplevel)"
BASE="$PIXARC_ROOT/JiT/baselines/taylorseer-style"
UPSTREAM_JIT="$PIXARC_ROOT/third-party/JiT"
CHECKPOINT="$PIXARC_ROOT/JiT/checkpoints/JiT-B-16-256/checkpoint-last.pth"
: "${JIT_PYTHON:?set the JiT environment Python executable}"
: "${OUTPUT_ROOT:?set an output directory outside PixARC}"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
[[ "$OUTPUT_ROOT/" != "$PIXARC_ROOT/"* ]] || exit 2
export PIXARC_ROOT BASE UPSTREAM_JIT CHECKPOINT OUTPUT_ROOT JIT_PYTHON
export PATH="$(dirname "$JIT_PYTHON"):$PATH"
export PYTHONPATH="$UPSTREAM_JIT:$BASE:${PYTHONPATH:-}"
cd "$BASE"
test -s "$CHECKPOINT"
mkdir -p "$OUTPUT_ROOT"
```

No dependency is installed automatically. `requirements-extra.txt` lists
evaluation/test extras; the checkpoint, ADM evaluator, ImageNet reference NPZ,
and LPIPS Alex weights must already be local.

## CPU-only checks

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/pixarc-jit-taylorseer-pycache \
  "$JIT_PYTHON" -m compileall "$BASE"
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  "$JIT_PYTHON" -m pytest -q -p no:cacheprovider "$BASE/tests"
python scripts/compare_common_tool_interfaces.py --pixarc-root "$PIXARC_ROOT"
git -C "$PIXARC_ROOT" diff --check
```

Real JiT construction is CUDA-coupled through upstream rotary embedding, so
CPU tests use small tensors/mocks and dynamic official-helper imports. They do
not replace deferred model parity.

## Deterministic manifests

Choose disjoint 63-bit seed ranges. Full and TaylorSeer must use the same final
manifest; validation and final ranges must not overlap.

```bash
: "${VALIDATION_BASE_SEED:?set an 8K-only base seed}"
: "${FINAL_BASE_SEED:?set a disjoint 50K base seed}"
VALIDATION_MANIFEST="$OUTPUT_ROOT/manifests/jit_imagenet8k.jsonl"
MANIFEST="$OUTPUT_ROOT/manifests/jit_imagenet50k.jsonl"
python scripts/build_manifest.py --output "$VALIDATION_MANIFEST" \
  --samples-per-class 8 --base-seed "$VALIDATION_BASE_SEED" \
  --split-name imagenet8k_validation --world-size 4 --batch-size 32 \
  --generator-device cuda --noise-dtype float32
python scripts/build_manifest.py --output "$MANIFEST" \
  --samples-per-class 50 --base-seed "$FINAL_BASE_SEED" \
  --split-name imagenet50k_final --world-size 4 --batch-size 32 \
  --generator-device cuda --noise-dtype float32
python scripts/validate_manifest.py --manifest "$VALIDATION_MANIFEST" \
  --expected-count 8000 --expected-per-class 8 \
  --expected-num-classes 1000 --world-size 4 --batch-size 32 \
  --base-seed "$VALIDATION_BASE_SEED" --disjoint-with "$MANIFEST"
python scripts/validate_manifest.py --manifest "$MANIFEST" \
  --expected-count 50000 --expected-per-class 50 \
  --expected-num-classes 1000 --world-size 4 --batch-size 32 \
  --base-seed "$FINAL_BASE_SEED" --disjoint-with "$VALIDATION_MANIFEST"
```

Every sample uses its own seed, so rank launch order, resume, prior samples,
and Taylor actions cannot change its noise. Sidecars record SHA-256, class and
shard rules, grouping, generator device/algorithm, dtype, and shape.

## Deferred workflow overview

First inspect active work. Proceed only with allocated idle devices:

```bash
bash scripts/inspect_active_runs.sh
export TAYLORSEER_GPU_TESTS_ALLOWED=1
```

The commands below are a workflow synopsis. Materialize `MAX_ORDER`,
`SMOKE_MANIFEST`, `SHADOW_CONFIG`, `RUNNER_CONFIG`, `SELECTED_CONFIG`, and
`MANIFEST` exactly as specified in `RUNBOOK.md` before executing them.

Estimate factor memory without CUDA:

```bash
python scripts/estimate_cache_memory.py --preset jit-b16-256 \
  --batch-size "${BATCH_SIZE:-32}" --max-order "$MAX_ORDER" \
  --cache-dtype bfloat16 --output-json "$OUTPUT_ROOT/jit_cache_estimate.json"
```

Use a manifest of at most eight records for Full-versus-instrumented smoke;
then use a separately materialized numeric Taylor config for reuse smoke and
shadow diagnostics:

```bash
bash scripts/run_deferred_smoke_tests.sh \
  --full-config configs/jit_b16_256_upstream_full.yaml \
  --candidate-config configs/jit_b16_256_instrumented_full.yaml \
  --manifest "$SMOKE_MANIFEST" --output-root "$OUTPUT_ROOT/smoke/jit_parity"
bash scripts/run_shadow_diagnostic.sh --config "$SHADOW_CONFIG" \
  --manifest "$SMOKE_MANIFEST" --output-root "$OUTPUT_ROOT/shadow/jit"
```

The full 1K/8K sweep/materialization and parity gates are in `RUNBOOK.md`.
Select JiT interval/order only on 8K, freeze `SELECTED_CONFIG` and its hash,
then benchmark matched modes:

```bash
bash scripts/benchmark_single_gpu.sh \
  --runner-factory taylorseer_style.jit_benchmark:build_benchmark_spec \
  --runner-config "$RUNNER_CONFIG" \
  --output-json "$OUTPUT_ROOT/benchmark/jit.json" \
  --warmup-batches 10 --measured-batches 30
```

Launch a manifest-backed 50K on four independently visible idle GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/launch_4gpu_50k.sh --config "$SELECTED_CONFIG" \
  --manifest "$MANIFEST" --output-root "$OUTPUT_ROOT/final/jit_taylorseer"
```

Resume uses the same exact arguments plus `--resume`; it refuses changed
archived inputs or mismatched/damaged samples.

Validate and evaluate:

```bash
python scripts/validate_outputs.py \
  --sample-dir "$OUTPUT_ROOT/final/jit_taylorseer/samples" \
  --metadata-dir "$OUTPUT_ROOT/final/jit_taylorseer/metadata" \
  --run-metadata "$OUTPUT_ROOT/final/jit_taylorseer/run_manifest.json" \
  --manifest "$MANIFEST" --expected-count 50000 --expected-per-class 50 \
  --expected-num-classes 1000 --resolution 256
bash scripts/evaluate_distribution.sh \
  --sample-dir "$OUTPUT_ROOT/final/jit_taylorseer/samples" \
  --manifest "$MANIFEST" --reference-npz "$IMAGENET_REFERENCE_NPZ" \
  --evaluator "$ADM_EVALUATOR" \
  --run-metadata "$OUTPUT_ROOT/final/jit_taylorseer/run_manifest.json" \
  --output-json "$OUTPUT_ROOT/metrics/jit_taylorseer_distribution.json"
bash scripts/evaluate_paired.sh \
  --reference-dir "$PAIRED_FULL_ROOT/samples" \
  --candidate-dir "$OUTPUT_ROOT/final/jit_taylorseer/samples" \
  --reference-manifest "$MANIFEST" --candidate-manifest "$MANIFEST" \
  --reference-run-metadata "$PAIRED_FULL_ROOT/run_manifest.json" \
  --candidate-run-metadata "$OUTPUT_ROOT/final/jit_taylorseer/run_manifest.json" \
  --output-json "$OUTPUT_ROOT/metrics/jit_paired.json" \
  --output-csv "$OUTPUT_ROOT/metrics/jit_paired.csv"
```

The queued upstream JiT Full is only conditionally pairable pending exact RNG
replay. If proof fails, `PAIRED_FULL_ROOT` must be a new manifest-backed Full
run. Distribution metrics may still use a completed validated upstream Full.

## Reporting and limitations

Report FID/sFID/IS/precision/recall and candidate-minus-Full deltas; paired
PSNR/SSIM/LPIPS only after strict metadata validation. Primary `speedup` is
matched Full median latency divided by TaylorSeer median latency at identical
compile mode/batch/noise. Report upstream compiled Full separately. Cache
upper bounds, current compatibility, licenses, and unverified risks are in
`MEMORY_REPORT.md`, `BASELINE_COMPATIBILITY_REPORT.md`, `NOTICE.md`, and
`SAFETY_AND_LIMITATIONS.md`.
