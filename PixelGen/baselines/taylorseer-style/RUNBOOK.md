# PixelGen TaylorSeer execution runbook

Run these phases in order. Phases 4 onward require CUDA and were **not run**
during implementation. At authoring time PixelGen Full was active and JiT Full
was queued.

## 0. Environment and safety audit

```bash
PIXARC_ROOT="${PIXARC_ROOT:-$(git rev-parse --show-toplevel)}"
BASE="$PIXARC_ROOT/PixelGen/baselines/taylorseer-style"
UPSTREAM_PIXELGEN="$PIXARC_ROOT/third-party/PixelGen"
CHECKPOINT="${CHECKPOINT:-$PIXARC_ROOT/PixelGen/checkpoints/PixelGen_XL_160ep.ckpt}"
CHECKPOINT="$(realpath "$CHECKPOINT")"
: "${PIXELGEN_PYTHON:?set the PixelGen environment Python executable}"
: "${OUTPUT_ROOT:?set an external output root}"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
[[ "$OUTPUT_ROOT/" != "$PIXARC_ROOT/"* ]] || exit 2
export PIXARC_ROOT BASE UPSTREAM_PIXELGEN CHECKPOINT OUTPUT_ROOT PIXELGEN_PYTHON
export PATH="$(dirname "$PIXELGEN_PYTHON"):$PATH"
export PYTHONPATH="$UPSTREAM_PIXELGEN:$BASE:${PYTHONPATH:-}"
cd "$BASE"
test -s "$CHECKPOINT"
mkdir -p "$OUTPUT_ROOT"
bash scripts/inspect_active_runs.sh
```

Do not proceed if a target GPU is occupied or cannot be queried. Never kill or
reset another job. Only after current work finishes and allocated GPUs are
idle, set `TAYLORSEER_GPU_TESTS_ALLOWED=1` and an explicit
`CUDA_VISIBLE_DEVICES`.

## 1. CPU verification

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/pixarc-pixelgen-taylorseer-pycache \
  "$PIXELGEN_PYTHON" -m compileall "$BASE"
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  "$PIXELGEN_PYTHON" -m pytest -q -p no:cacheprovider "$BASE/tests"
CUDA_VISIBLE_DEVICES="" python scripts/compare_common_tool_interfaces.py \
  --pixarc-root "$PIXARC_ROOT" \
  --output-json "$OUTPUT_ROOT/common_interface_check.json"
git -C "$PIXARC_ROOT" diff --check
git -C "$PIXARC_ROOT" status --short
```

Require official formula/schedule parity, 99-NFE sequence, exact-only and
combined-CFG state, deepcopy/reset, context/memory, manifest/sharding,
metric-toy, and metadata tests. CPU success is not real-model parity.

## 2. Immutable manifests

Choose four disjoint signed 63-bit seed ranges:

```bash
: "${SMOKE_BASE_SEED:?set smoke seed base}"
: "${PROXY_BASE_SEED:?set disjoint 1K seed base}"
: "${VALIDATION_BASE_SEED:?set disjoint 8K seed base}"
: "${FINAL_BASE_SEED:?set disjoint 50K seed base}"
SMOKE_MANIFEST="$OUTPUT_ROOT/manifests/pixelgen_smoke8.jsonl"
PROXY_MANIFEST="$OUTPUT_ROOT/manifests/pixelgen_imagenet1k.jsonl"
VALIDATION_MANIFEST="$OUTPUT_ROOT/manifests/pixelgen_imagenet8k.jsonl"
MANIFEST="$OUTPUT_ROOT/manifests/pixelgen_imagenet50k.jsonl"
export SMOKE_MANIFEST PROXY_MANIFEST VALIDATION_MANIFEST MANIFEST

python scripts/build_manifest.py --output "$SMOKE_MANIFEST" \
  --samples-per-class 1 --num-classes 8 --base-seed "$SMOKE_BASE_SEED" \
  --split-name smoke8 --world-size 1 --batch-size 2 \
  --generator-device cpu --noise-dtype float32
python scripts/build_manifest.py --output "$PROXY_MANIFEST" \
  --samples-per-class 1 --base-seed "$PROXY_BASE_SEED" \
  --split-name imagenet1k_proxy --world-size 4 --batch-size 4 \
  --generator-device cpu --noise-dtype float32
python scripts/build_manifest.py --output "$VALIDATION_MANIFEST" \
  --samples-per-class 8 --base-seed "$VALIDATION_BASE_SEED" \
  --split-name imagenet8k_validation --world-size 4 --batch-size 4 \
  --generator-device cpu --noise-dtype float32
python scripts/build_manifest.py --output "$MANIFEST" \
  --samples-per-class 50 --base-seed "$FINAL_BASE_SEED" \
  --split-name imagenet50k_final --world-size 4 --batch-size 4 \
  --generator-device cpu --noise-dtype float32
```

Validate counts, class balance, grouping, seed rule, and disjointness:

```bash
python scripts/validate_manifest.py --manifest "$SMOKE_MANIFEST" \
  --expected-count 8 --expected-per-class 1 --expected-num-classes 8 \
  --world-size 1 --batch-size 2 --base-seed "$SMOKE_BASE_SEED" \
  --disjoint-with "$PROXY_MANIFEST" --disjoint-with "$VALIDATION_MANIFEST" \
  --disjoint-with "$MANIFEST"
python scripts/validate_manifest.py --manifest "$PROXY_MANIFEST" \
  --expected-count 1000 --expected-per-class 1 --expected-num-classes 1000 \
  --world-size 4 --batch-size 4 --base-seed "$PROXY_BASE_SEED" \
  --disjoint-with "$SMOKE_MANIFEST" --disjoint-with "$VALIDATION_MANIFEST" \
  --disjoint-with "$MANIFEST"
python scripts/validate_manifest.py --manifest "$VALIDATION_MANIFEST" \
  --expected-count 8000 --expected-per-class 8 --expected-num-classes 1000 \
  --world-size 4 --batch-size 4 --base-seed "$VALIDATION_BASE_SEED" \
  --disjoint-with "$SMOKE_MANIFEST" --disjoint-with "$PROXY_MANIFEST" \
  --disjoint-with "$MANIFEST"
python scripts/validate_manifest.py --manifest "$MANIFEST" \
  --expected-count 50000 --expected-per-class 50 --expected-num-classes 1000 \
  --world-size 4 --batch-size 4 --base-seed "$FINAL_BASE_SEED" \
  --disjoint-with "$SMOKE_MANIFEST" --disjoint-with "$PROXY_MANIFEST" \
  --disjoint-with "$VALIDATION_MANIFEST"
sha256sum "$OUTPUT_ROOT"/manifests/*.jsonl "$OUTPUT_ROOT"/manifests/*.meta.json
```

Never edit a manifest or sidecar after any run begins.

## 3. Memory estimate

```bash
: "${MAX_ORDER:=4}"
: "${BATCH_SIZE:=4}"
python scripts/estimate_cache_memory.py --preset pixelgen-xl-256 \
  --batch-size "$BATCH_SIZE" --max-order "$MAX_ORDER" \
  --cache-dtype bfloat16 \
  --output-json "$OUTPUT_ROOT/pixelgen_cache_estimate.json"
```

This excludes parameters/VAE/activations/compiler workspace and cannot certify
a 3090 batch size.

## 4. Materialize smoke configs

```bash
SMOKE_CONFIG_DIR="$OUTPUT_ROOT/configs/pixelgen_smoke"
mkdir -p "$SMOKE_CONFIG_DIR"
export SMOKE_CONFIG_DIR
python - <<'PY'
import copy, os
from pathlib import Path
import yaml

root = Path(os.environ["BASE"])
out = Path(os.environ["SMOKE_CONFIG_DIR"])
def set_ts(cfg, mode, interval, order, trace):
    cfg["checkpoint"] = os.environ["CHECKPOINT"]
    cfg["runtime"]["batch_size"] = 2
    cfg["data"]["train_batch_size"] = 2
    cfg["data"]["pred_batch_size"] = 2
    cfg["taylorseer"].update(mode=mode, interval=interval, max_order=order, trace_mode=trace)
    args = cfg["model"]["denoiser"]["init_args"]
    args.update(taylorseer_mode=mode, taylorseer_interval=interval,
                taylorseer_max_order=order, taylorseer_trace_mode=trace)
for name in ("upstream_full", "instrumented_full"):
    src = root / "configs" / f"pixelgen_xl_256_{name}.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    top = cfg["taylorseer"]
    set_ts(cfg, top["mode"], top["interval"], top["max_order"], top["trace_mode"])
    (out/f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
base = yaml.safe_load((root/"configs/pixelgen_xl_256_taylorseer.yaml").read_text(encoding="utf-8"))
for name, mode, interval, order, trace in (
    ("interval1", "taylorseer", 1, 4, "full"),
    ("reuse_diagnostic", "taylorseer", 2, 1, "full"),
    ("shadow", "shadow_forecast", 2, 4, "shadow"),
):
    cfg = copy.deepcopy(base)
    set_ts(cfg, mode, interval, order, trace)
    (out/f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
sha256sum "$SMOKE_CONFIG_DIR"/*.yaml
```

Interval 2/order 1 is a reuse diagnostic, not a selected operating point.

## 5. Deferred smoke and three-way Full parity

```bash
: "${CUDA_VISIBLE_DEVICES:?set exactly one allocated idle GPU}"
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || exit 2
export CUDA_VISIBLE_DEVICES TAYLORSEER_GPU_TESTS_ALLOWED=1
bash scripts/inspect_active_runs.sh
bash scripts/run_deferred_smoke_tests.sh \
  --full-config "$SMOKE_CONFIG_DIR/upstream_full.yaml" \
  --candidate-config "$SMOKE_CONFIG_DIR/instrumented_full.yaml" \
  --manifest "$SMOKE_MANIFEST" \
  --output-root "$OUTPUT_ROOT/smoke/pixelgen_upstream_vs_instrumented"
bash scripts/run_deferred_smoke_tests.sh \
  --full-config "$SMOKE_CONFIG_DIR/upstream_full.yaml" \
  --candidate-config "$SMOKE_CONFIG_DIR/interval1.yaml" \
  --manifest "$SMOKE_MANIFEST" \
  --output-root "$OUTPUT_ROOT/smoke/pixelgen_upstream_vs_interval1"
python scripts/generate_shard.py --config "$SMOKE_CONFIG_DIR/reuse_diagnostic.yaml" \
  --manifest "$SMOKE_MANIFEST" --shard-id 0 --world-size 1 \
  --output-root "$OUTPUT_ROOT/smoke/pixelgen_reuse" --acknowledge-gpu-job
```

Validate each output at count 8. Require strict checkpoint/EMA keys, finite
images, 99 combined forwards in unconditional-first order, expected schedule,
return tuple parity, correct context shapes, independent empty deepcopy state,
cleanup, and recorded equality tolerance. Stop on failure.

## 6. Deferred shadow diagnosis

```bash
bash scripts/run_shadow_diagnostic.sh --config "$SMOKE_CONFIG_DIR/shadow.yaml" \
  --manifest "$SMOKE_MANIFEST" --output-root "$OUTPUT_ROOT/shadow/pixelgen"
```

Inspect errors by layer/module/order/horizon and solver stage. Shadow is exact
diagnostic work and is excluded from latency.

## 7. Deferred 1K proxy

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/inspect_active_runs.sh
bash scripts/launch_4gpu_50k.sh --config "$SMOKE_CONFIG_DIR/upstream_full.yaml" \
  --manifest "$PROXY_MANIFEST" --output-root "$OUTPUT_ROOT/proxy1k/pixelgen_full"
bash scripts/launch_4gpu_50k.sh --config "$SMOKE_CONFIG_DIR/reuse_diagnostic.yaml" \
  --manifest "$PROXY_MANIFEST" \
  --output-root "$OUTPUT_ROOT/proxy1k/pixelgen_interval2_order1"
```

Validate 1,000 images/one per class, then inspect paired scores, schedule,
latency, memory, and failure tails. This stage does not choose final settings.

## 8. Deferred 8K interval/order sweep

Materialize every pair from `configs/interval_order_sweep_template.yaml` into
an external directory. Set the absolute checkpoint and update both the
top-level TaylorSeer block and prefixed denoiser constructor values:

```bash
CANDIDATE_CONFIG_DIR="$OUTPUT_ROOT/configs/pixelgen_8k_candidates"
mkdir -p "$CANDIDATE_CONFIG_DIR"
export CANDIDATE_CONFIG_DIR
python - <<'PY'
import copy, os
from pathlib import Path
import yaml

root = Path(os.environ["BASE"])
base = yaml.safe_load((root/"configs/pixelgen_xl_256_taylorseer.yaml").read_text(encoding="utf-8"))
grid = yaml.safe_load((root/"configs/interval_order_sweep_template.yaml").read_text(encoding="utf-8"))
base["checkpoint"] = os.environ["CHECKPOINT"]
out = Path(os.environ["CANDIDATE_CONFIG_DIR"])
for interval in grid["candidate_intervals"]:
    for order in grid["candidate_max_orders"]:
        cfg = copy.deepcopy(base)
        cfg["taylorseer"].update(mode="taylorseer", interval=int(interval),
                                  max_order=int(order), trace_mode="summary")
        args = cfg["model"]["denoiser"]["init_args"]
        args.update(taylorseer_mode="taylorseer", taylorseer_interval=int(interval),
                    taylorseer_max_order=int(order), taylorseer_trace_mode="summary")
        (out/f"pixelgen_i{interval}_k{order}.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
sha256sum "$CANDIDATE_CONFIG_DIR"/*.yaml
```

Generate one upstream Full 8K reference, then each candidate sequentially on
`$VALIDATION_MANIFEST`. Validate before evaluating. Select
conservative/medium/aggressive points from measured matched latency, memory,
distribution and paired quality, and trace stability. Do not use final 50K.

```bash
VALIDATION_FULL_ROOT="$OUTPUT_ROOT/validation8k/pixelgen_full"
bash scripts/launch_4gpu_50k.sh --config "$SMOKE_CONFIG_DIR/upstream_full.yaml" \
  --manifest "$VALIDATION_MANIFEST" --output-root "$VALIDATION_FULL_ROOT"
python scripts/validate_outputs.py --sample-dir "$VALIDATION_FULL_ROOT/samples" \
  --metadata-dir "$VALIDATION_FULL_ROOT/metadata" \
  --run-metadata "$VALIDATION_FULL_ROOT/run_manifest.json" \
  --manifest "$VALIDATION_MANIFEST" --expected-count 8000 \
  --expected-per-class 8 --expected-num-classes 1000 --resolution 256
for CANDIDATE_CONFIG in "$CANDIDATE_CONFIG_DIR"/*.yaml; do
  NAME="$(basename "$CANDIDATE_CONFIG" .yaml)"
  RUN_ROOT="$OUTPUT_ROOT/validation8k/$NAME"
  bash scripts/launch_4gpu_50k.sh --config "$CANDIDATE_CONFIG" \
    --manifest "$VALIDATION_MANIFEST" --output-root "$RUN_ROOT"
  python scripts/validate_outputs.py --sample-dir "$RUN_ROOT/samples" \
    --metadata-dir "$RUN_ROOT/metadata" \
    --run-metadata "$RUN_ROOT/run_manifest.json" \
    --manifest "$VALIDATION_MANIFEST" --expected-count 8000 \
    --expected-per-class 8 --expected-num-classes 1000 --resolution 256
done
```

Freeze the chosen point:

```bash
: "${INTERVAL:?set the 8K-selected interval}"
: "${MAX_ORDER:?set the 8K-selected order}"
SELECTED_CONFIG="$CANDIDATE_CONFIG_DIR/pixelgen_i${INTERVAL}_k${MAX_ORDER}.yaml"
export INTERVAL MAX_ORDER SELECTED_CONFIG
test -s "$SELECTED_CONFIG"
sha256sum "$SELECTED_CONFIG"
```

## 9. Deferred single-GPU latency

```bash
RUNNER_CONFIG="$OUTPUT_ROOT/benchmark/pixelgen_runner_b1.json"
mkdir -p "$(dirname "$RUNNER_CONFIG")"
export RUNNER_CONFIG
python - <<'PY'
import json, os
from pathlib import Path
row = json.loads(Path(os.environ["MANIFEST"]).read_text(encoding="utf-8").splitlines()[0])
payload = {"model_config": os.environ["SELECTED_CONFIG"], "batch_size": 1,
           "sample_ids": [row["sample_id"]], "seeds": [row["seed"]],
           "class_ids": [row["class_id"]]}
Path(os.environ["RUNNER_CONFIG"]).write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
PY
export CUDA_VISIBLE_DEVICES=0
bash scripts/benchmark_single_gpu.sh \
  --runner-factory taylorseer_style.pixelgen_benchmark:build_benchmark_spec \
  --runner-config "$RUNNER_CONFIG" \
  --output-json "$OUTPUT_ROOT/benchmark/pixelgen_b1.json" \
  --warmup-batches 10 --measured-batches 30
```

Repeat at one common throughput batch. Primary speedup uses matched Full and
TaylorSeer in the same compile mode; report upstream compiled Full separately,
plus compile time, graph breaks, steady latency, and peak memory.

## 10. Deferred four-GPU 50K

The active compressed upstream result cannot be paired. Generate a new
manifest-backed Full:

```bash
FINAL_FULL_ROOT="$OUTPUT_ROOT/final/pixelgen_full"
FINAL_TAYLOR_ROOT="$OUTPUT_ROOT/final/pixelgen_taylorseer_i${INTERVAL}_k${MAX_ORDER}"
export FINAL_FULL_ROOT FINAL_TAYLOR_ROOT
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/launch_4gpu_50k.sh --config "$SMOKE_CONFIG_DIR/upstream_full.yaml" \
  --manifest "$MANIFEST" --output-root "$FINAL_FULL_ROOT"
bash scripts/launch_4gpu_50k.sh --config "$SELECTED_CONFIG" \
  --manifest "$MANIFEST" --output-root "$FINAL_TAYLOR_ROOT"
```

Resume only with identical arguments and `--resume`. Changed archived
config/manifest/sidecar inputs, metadata, or damaged images fail closed.

## 11. Validate outputs

Run once for each final root:

```bash
RUN_ROOT="$FINAL_TAYLOR_ROOT"
python scripts/validate_outputs.py --sample-dir "$RUN_ROOT/samples" \
  --metadata-dir "$RUN_ROOT/metadata" \
  --run-metadata "$RUN_ROOT/run_manifest.json" --manifest "$MANIFEST" \
  --expected-count 50000 --expected-per-class 50 \
  --expected-num-classes 1000 --resolution 256
```

## 12. Distribution metrics

```bash
: "${IMAGENET_REFERENCE_NPZ:?set local ImageNet-256 ADM reference}"
: "${ADM_EVALUATOR:?set local ADM evaluator.py}"
mkdir -p "$OUTPUT_ROOT/metrics"
bash scripts/evaluate_distribution.sh --sample-dir "$FINAL_FULL_ROOT/samples" \
  --manifest "$MANIFEST" --reference-npz "$IMAGENET_REFERENCE_NPZ" \
  --evaluator "$ADM_EVALUATOR" --run-metadata "$FINAL_FULL_ROOT/run_manifest.json" \
  --output-json "$OUTPUT_ROOT/metrics/pixelgen_full_distribution.json"
bash scripts/evaluate_distribution.sh --sample-dir "$FINAL_TAYLOR_ROOT/samples" \
  --manifest "$MANIFEST" --reference-npz "$IMAGENET_REFERENCE_NPZ" \
  --evaluator "$ADM_EVALUATOR" --run-metadata "$FINAL_TAYLOR_ROOT/run_manifest.json" \
  --output-json "$OUTPUT_ROOT/metrics/pixelgen_taylorseer_distribution.json"
python scripts/compare_distribution.py \
  --full-json "$OUTPUT_ROOT/metrics/pixelgen_full_distribution.json" \
  --taylorseer-json "$OUTPUT_ROOT/metrics/pixelgen_taylorseer_distribution.json" \
  --output-json "$OUTPUT_ROOT/metrics/pixelgen_distribution_delta.json"
```

The current upstream compressed Full may be separately evaluated for
distribution quality after validation, but never paired post hoc.

## 13. Strict paired metrics

```bash
bash scripts/evaluate_paired.sh --reference-dir "$FINAL_FULL_ROOT/samples" \
  --candidate-dir "$FINAL_TAYLOR_ROOT/samples" \
  --reference-manifest "$MANIFEST" --candidate-manifest "$MANIFEST" \
  --reference-run-metadata "$FINAL_FULL_ROOT/run_manifest.json" \
  --candidate-run-metadata "$FINAL_TAYLOR_ROOT/run_manifest.json" \
  --output-json "$OUTPUT_ROOT/metrics/pixelgen_paired.json" \
  --output-csv "$OUTPUT_ROOT/metrics/pixelgen_paired.csv" \
  --expected-count 50000 --expected-per-class 50 \
  --expected-num-classes 1000 --resolution 256 \
  --lpips-device cpu --lpips-batch-size 16
```

If local LPIPS/Alex weights are missing, `--skip-lpips` yields explicitly
partial PSNR/SSIM output only. Never substitute a backbone or download weights.

## 14. Aggregate and archive

Archive selected config/hash, manifest+sidecar/hash, source hash, run manifests,
rank JSONL/summaries/traces/logs, `four_gpu_wall_clock.json`, benchmark/memory
JSON, distribution/delta JSON, paired JSON/CSV, and software versions. Verify
99 combined forwards, q range, unconditional-first 2B order, Full ratio,
horizon/order, cache cleanup, EMA runtime isolation, and peak memory. Report
batch-1 latency, common-batch throughput, and four-GPU wall time separately.
Record failures/OOMs; never silently change batch/order or use Lite.
