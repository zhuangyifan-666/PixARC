# PixelGen SpeCa execution runbook

Run the gates in order. Gates 4–15 require a future, separately authorized GPU window and were **not executed** in this task. Never set `SPECA_GPU_TESTS_ALLOWED=1` merely because a GPU looks idle; obtain an allocation first. The wrappers refuse busy GPUs and never kill another process.

## 1. Environment audit and CPU gate

Do not assume the current directory:

```bash
: "${PIXARC_ROOT:?set the local PixARC checkout}"
PIXARC_ROOT="$(git -C "$PIXARC_ROOT" rev-parse --show-toplevel)"
BASE="$PIXARC_ROOT/PixelGen/baselines/speca-style"
UPSTREAM_PIXELGEN="$PIXARC_ROOT/third-party/PixelGen"
: "${PIXELGEN_PYTHON:?set the PixelGen-environment Python executable}"
: "${CHECKPOINT:?set the PixelGen checkpoint path}"
: "${OUTPUT_ROOT:?set an external output root}"
CHECKPOINT="$(realpath "$CHECKPOINT")"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
test -s "$CHECKPOINT" && test -d "$UPSTREAM_PIXELGEN"
[[ "$OUTPUT_ROOT/" != "$PIXARC_ROOT/"* ]]
mkdir -p "$OUTPUT_ROOT"
export PIXARC_ROOT BASE UPSTREAM_PIXELGEN PIXELGEN_PYTHON CHECKPOINT OUTPUT_ROOT
export PATH="$(dirname "$PIXELGEN_PYTHON"):$PATH"
export PYTHONPATH="$UPSTREAM_PIXELGEN:$BASE${PYTHONPATH:+:$PYTHONPATH}"
cd "$BASE"

git -C "$PIXARC_ROOT" rev-parse HEAD
git -C "$PIXARC_ROOT/baselines/Cache4Diffusion" rev-parse HEAD
git -C "$PIXARC_ROOT/baselines/TaylorSeer" rev-parse HEAD
git -C "$PIXARC_ROOT" rev-parse HEAD:third-party/JiT
git -C "$PIXARC_ROOT" rev-parse HEAD:third-party/PixelGen
git -C "$PIXARC_ROOT" status --short --untracked-files=no
bash scripts/inspect_active_runs.sh
```

Perform the process audit once. Do not read `/proc/*/environ`, attach, signal, reset, or scan an active output tree. Then hide CUDA for all CPU checks:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/pixarc-pixelgen-speca-pycache \
  "$PIXELGEN_PYTHON" -m compileall -q "$BASE"
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  "$PIXELGEN_PYTHON" -m pytest -q -p no:cacheprovider "$BASE/tests"
CUDA_VISIBLE_DEVICES="" "$PIXELGEN_PYTHON" scripts/compare_common_tool_interfaces.py \
  --pixarc-root "$PIXARC_ROOT" \
  --output-json "$OUTPUT_ROOT/pixelgen_common_interface_check.json"
for script in scripts/*.sh; do bash -n "$script"; done
git -C "$PIXARC_ROOT" diff --check
```

Require official scheduler/error fixtures, signed-gap/exact-only predictor parity, previous-error/no-rollback, combined-CFG/deepcopy/reset/state-dict, Heun counts, context, memory, manifest/sharding, paired-toy, metadata, and common-core tests. CPU success is not real-model/Lightning parity.

## 2. Immutable batch-1 manifests

Use four disjoint signed-63-bit seed ranges. PixelGen constructs deterministic noise with a CPU generator before transfer:

```bash
: "${SMOKE_BASE_SEED:?set an 8-image seed base}"
: "${PROXY_BASE_SEED:?set a disjoint 1K seed base}"
: "${VALIDATION_BASE_SEED:?set a disjoint 8K seed base}"
: "${FINAL_BASE_SEED:?set a disjoint 50K seed base}"
BATCH_SIZE=1
SMOKE_MANIFEST="$OUTPUT_ROOT/manifests/pixelgen_smoke8_b1.jsonl"
PROXY_MANIFEST="$OUTPUT_ROOT/manifests/pixelgen_imagenet1k_b1.jsonl"
VALIDATION_MANIFEST="$OUTPUT_ROOT/manifests/pixelgen_imagenet8k_b1.jsonl"
MANIFEST="$OUTPUT_ROOT/manifests/pixelgen_imagenet50k_b1.jsonl"
export BATCH_SIZE SMOKE_MANIFEST PROXY_MANIFEST VALIDATION_MANIFEST MANIFEST
mkdir -p "$OUTPUT_ROOT/manifests"

"$PIXELGEN_PYTHON" scripts/build_manifest.py --output "$SMOKE_MANIFEST" \
  --samples-per-class 1 --num-classes 8 --base-seed "$SMOKE_BASE_SEED" \
  --split-name smoke8 --world-size 1 --batch-size 1 \
  --generator-device cpu --noise-dtype float32 --noise-shape 3 256 256
"$PIXELGEN_PYTHON" scripts/build_manifest.py --output "$PROXY_MANIFEST" \
  --samples-per-class 1 --base-seed "$PROXY_BASE_SEED" \
  --split-name imagenet1k_proxy --world-size 4 --batch-size 1 \
  --generator-device cpu --noise-dtype float32 --noise-shape 3 256 256
"$PIXELGEN_PYTHON" scripts/build_manifest.py --output "$VALIDATION_MANIFEST" \
  --samples-per-class 8 --base-seed "$VALIDATION_BASE_SEED" \
  --split-name imagenet8k_validation --world-size 4 --batch-size 1 \
  --generator-device cpu --noise-dtype float32 --noise-shape 3 256 256
"$PIXELGEN_PYTHON" scripts/build_manifest.py --output "$MANIFEST" \
  --samples-per-class 50 --base-seed "$FINAL_BASE_SEED" \
  --split-name imagenet50k_final --world-size 4 --batch-size 1 \
  --generator-device cpu --noise-dtype float32 --noise-shape 3 256 256
```

Validate sidecars, class balance, one-sample groups, seeds, shards, and disjointness:

```bash
"$PIXELGEN_PYTHON" scripts/validate_manifest.py --manifest "$SMOKE_MANIFEST" \
  --expected-count 8 --expected-per-class 1 --expected-num-classes 8 \
  --world-size 1 --batch-size 1 --base-seed "$SMOKE_BASE_SEED" \
  --require-sidecar --disjoint-with "$PROXY_MANIFEST" \
  --disjoint-with "$VALIDATION_MANIFEST" --disjoint-with "$MANIFEST"
"$PIXELGEN_PYTHON" scripts/validate_manifest.py --manifest "$PROXY_MANIFEST" \
  --expected-count 1000 --expected-per-class 1 --expected-num-classes 1000 \
  --world-size 4 --batch-size 1 --base-seed "$PROXY_BASE_SEED" \
  --require-sidecar --disjoint-with "$SMOKE_MANIFEST" \
  --disjoint-with "$VALIDATION_MANIFEST" --disjoint-with "$MANIFEST"
"$PIXELGEN_PYTHON" scripts/validate_manifest.py --manifest "$VALIDATION_MANIFEST" \
  --expected-count 8000 --expected-per-class 8 --expected-num-classes 1000 \
  --world-size 4 --batch-size 1 --base-seed "$VALIDATION_BASE_SEED" \
  --require-sidecar --disjoint-with "$SMOKE_MANIFEST" \
  --disjoint-with "$PROXY_MANIFEST" --disjoint-with "$MANIFEST"
"$PIXELGEN_PYTHON" scripts/validate_manifest.py --manifest "$MANIFEST" \
  --expected-count 50000 --expected-per-class 50 --expected-num-classes 1000 \
  --world-size 4 --batch-size 1 --base-seed "$FINAL_BASE_SEED" \
  --require-sidecar --disjoint-with "$SMOKE_MANIFEST" \
  --disjoint-with "$PROXY_MANIFEST" --disjoint-with "$VALIDATION_MANIFEST"
sha256sum "$OUTPUT_ROOT"/manifests/*.jsonl "$OUTPUT_ROOT"/manifests/*.meta.json
```

Never edit a manifest/sidecar after use. A final shard must contain exactly 12,500 real samples.

## 3. Memory estimate and resolved configs

Set a provisional point for smoke/1K; released defaults are not claimed PixelGen-optimal:

```bash
: "${MAX_ORDER:?set provisional/selected max order}"
: "${BASE_THRESHOLD:?set provisional/selected base threshold}"
: "${DECAY_RATE:?set provisional/selected decay rate}"
: "${MIN_TAYLOR_STEPS:?set provisional/selected minimum Taylor span}"
: "${MAX_TAYLOR_STEPS:?set provisional/selected maximum Taylor span}"
[[ "$BATCH_SIZE" == 1 ]]
mkdir -p "$OUTPUT_ROOT/memory" "$OUTPUT_ROOT/configs/pixelgen"
"$PIXELGEN_PYTHON" scripts/estimate_cache_memory.py --preset pixelgen-xl-256 \
  --batch-size "$BATCH_SIZE" --max-order "$MAX_ORDER" \
  --cache-dtype bfloat16 --verify-layer -1 \
  --output-json "$OUTPUT_ROOT/memory/pixelgen_b1_k${MAX_ORDER}.json"
```

Materialize absolute-checkpoint configs and keep top-level SpeCa fields synchronized with denoiser init args. Re-run after 8K selection:

```bash
CONFIG_DIR="$OUTPUT_ROOT/configs/pixelgen"
export CONFIG_DIR MAX_ORDER BASE_THRESHOLD DECAY_RATE \
  MIN_TAYLOR_STEPS MAX_TAYLOR_STEPS
"$PIXELGEN_PYTHON" - <<'PY'
import os
from pathlib import Path
import yaml
base, out = Path(os.environ["BASE"]), Path(os.environ["CONFIG_DIR"])
out.mkdir(parents=True, exist_ok=True)
values = dict(max_order=int(os.environ["MAX_ORDER"]),
    base_threshold=float(os.environ["BASE_THRESHOLD"]),
    decay_rate=float(os.environ["DECAY_RATE"]),
    min_taylor_steps=int(os.environ["MIN_TAYLOR_STEPS"]),
    max_taylor_steps=int(os.environ["MAX_TAYLOR_STEPS"]))
cfgs={}
for name in ("upstream_full","instrumented_full","taylor_draft_fixed","shadow_verify","speca"):
    cfg=yaml.safe_load((base/"configs"/f"pixelgen_xl_256_{name}.yaml").read_text())
    cfg["checkpoint"]=os.environ["CHECKPOINT"]
    if name in {"shadow_verify","speca"}: cfg["speca"].update(values)
    if name=="taylor_draft_fixed": cfg["speca"]["max_order"]=values["max_order"]
    init=cfg["model"]["denoiser"]["init_args"]
    # scheduler_mode is validated/run-identity metadata; runtime scheduler type
    # is selected by mode and the model constructor has no such keyword.
    init.update({f"speca_{key}":value for key,value in cfg["speca"].items()
                 if key != "scheduler_mode"})
    cfgs["speca_selected" if name=="speca" else name]=cfg
for name,cfg in cfgs.items():
    (out/f"{name}.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
PY
UPSTREAM_CONFIG="$CONFIG_DIR/upstream_full.yaml"
FULL_CONFIG="$CONFIG_DIR/instrumented_full.yaml"
FIXED_CONFIG="$CONFIG_DIR/taylor_draft_fixed.yaml"
SHADOW_CONFIG="$CONFIG_DIR/shadow_verify.yaml"
SELECTED_CONFIG="$CONFIG_DIR/speca_selected.yaml"
export UPSTREAM_CONFIG FULL_CONFIG FIXED_CONFIG SHADOW_CONFIG SELECTED_CONFIG
sha256sum "$CONFIG_DIR"/*.yaml
```

The candidate must retain released scheduler, relative L1/`1e-10`, first-enhance 3, floor 0.01, last-block/all-token verification, batch-global gate, NFE coordinate, `interval: null`, no forced final Full, inherited cache dtype, real batch 1/effective batch 2, matched compile, exact Heun, timeshift/guidance, and `[unconditional, conditional]` order.

## 4. Deferred smoke

```bash
: "${CUDA_VISIBLE_DEVICES:?set exactly one allocated idle GPU}"
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]]
export CUDA_VISIBLE_DEVICES SPECA_GPU_TESTS_ALLOWED=1
bash scripts/inspect_active_runs.sh
bash scripts/run_deferred_smoke_tests.sh \
  --full-config "$FULL_CONFIG" --candidate-config "$SELECTED_CONFIG" \
  --manifest "$SMOKE_MANIFEST" \
  --output-root "$OUTPUT_ROOT/smoke/pixelgen_instrumented_vs_speca"
```

Require checkpoint/`ema_denoiser`, finite output, both actions/checks, 99 combined forwards, effective batch 2 in correct order, exact diagnostics, reset, and no NaN.

## 5. Deferred Full parity

```bash
bash scripts/run_deferred_smoke_tests.sh \
  --full-config "$UPSTREAM_CONFIG" --candidate-config "$FULL_CONFIG" \
  --manifest "$SMOKE_MANIFEST" \
  --output-root "$OUTPUT_ROOT/parity/pixelgen_upstream_vs_instrumented"
for side in full candidate; do
  "$PIXELGEN_PYTHON" scripts/validate_outputs.py \
    --sample-dir "$OUTPUT_ROOT/parity/pixelgen_upstream_vs_instrumented/$side/samples" \
    --metadata-dir "$OUTPUT_ROOT/parity/pixelgen_upstream_vs_instrumented/$side/metadata" \
    --run-metadata "$OUTPUT_ROOT/parity/pixelgen_upstream_vs_instrumented/$side/run_manifest.json" \
    --manifest "$SMOKE_MANIFEST" --expected-count 8 \
    --expected-per-class 1 --expected-num-classes 8 --resolution 256
done
FULL_PARITY_MAX_U8="${FULL_PARITY_MAX_U8:-1}"
FULL_PARITY_MEAN_U8="${FULL_PARITY_MEAN_U8:-0.01}"
export FULL_PARITY_MAX_U8 FULL_PARITY_MEAN_U8
"$PIXELGEN_PYTHON" - \
  "$OUTPUT_ROOT/parity/pixelgen_upstream_vs_instrumented/full/samples" \
  "$OUTPUT_ROOT/parity/pixelgen_upstream_vs_instrumented/candidate/samples" <<'PY'
import os,sys
from pathlib import Path
import numpy as np
from PIL import Image
a,b=map(Path,sys.argv[1:]); names=sorted(p.name for p in a.glob("*.png"))
assert names and names==sorted(p.name for p in b.glob("*.png"))
ds=[abs(np.asarray(Image.open(a/n).convert("RGB"),dtype=np.int16)-np.asarray(Image.open(b/n).convert("RGB"),dtype=np.int16)) for n in names]
mx=max(int(d.max()) for d in ds); mean=sum(float(d.sum()) for d in ds)/sum(d.size for d in ds)
print({"pairs":len(names),"max_abs_u8":mx,"mean_abs_u8":mean})
assert mx<=int(os.environ["FULL_PARITY_MAX_U8"])
assert mean<=float(os.environ["FULL_PARITY_MEAN_U8"])
PY
```

On failure, locate the first block/module/EMA or sampler mismatch and stop before tuning.

## 6. Deferred draft parity

```bash
CUDA_VISIBLE_DEVICES="" "$PIXELGEN_PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_taylor_predictor_parity.py tests/test_exact_only_history.py
CUDA_VISIBLE_DEVICES="" "$PIXELGEN_PYTHON" scripts/compare_taylor_predictor_parity.py \
  --max-order "$MAX_ORDER" --dtype float64 \
  --output-json "$OUTPUT_ROOT/parity/pixelgen_taylor_predictor_cpu.json"
bash scripts/run_deferred_smoke_tests.sh \
  --full-config "$FULL_CONFIG" --candidate-config "$FIXED_CONFIG" \
  --manifest "$SMOKE_MANIFEST" \
  --output-root "$OUTPUT_ROOT/parity/pixelgen_fixed_draft"
```

Require identical exact schedule/coordinate/order, factors, forecasts, output, tensor count, available order, and combined-`2B` state against the self-contained TaylorSeer oracle. Fixed-draft quality is not SpeCa.

## 7. Deferred verification semantics

```bash
bash scripts/run_shadow_diagnostic.sh --config "$SHADOW_CONFIG" \
  --manifest "$SMOKE_MANIFEST" --output-root "$OUTPUT_ROOT/shadow/pixelgen"
CUDA_VISIBLE_DEVICES="" "$PIXELGEN_PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_previous_error_semantics.py tests/test_no_current_rollback.py \
  tests/test_verification_layer.py tests/test_verification_prefix.py \
  tests/test_combined_cfg_error.py tests/test_combined_cfg_state.py \
  tests/test_deepcopy_state.py
```

Confirm speculative-prefix local exact work, current draft retention, next-NFE effect, all-token combined error, exact `return_layer`/`return_last`, independent EMA/deepcopy state, and reset. Shadow work is excluded from performance/50K.

## 8. Deferred 1K proxy

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 SPECA_GPU_TESTS_ALLOWED=1
bash scripts/inspect_active_runs.sh
PROXY_FULL_ROOT="$OUTPUT_ROOT/proxy1k/pixelgen_instrumented_full"
PROXY_SPECA_ROOT="$OUTPUT_ROOT/proxy1k/pixelgen_speca"
bash scripts/launch_4gpu_50k.sh --config "$FULL_CONFIG" \
  --manifest "$PROXY_MANIFEST" --output-root "$PROXY_FULL_ROOT" \
  --nonfinal-proxy --expected-records 1000 \
  --expected-per-class 1 --expected-num-classes 1000
bash scripts/launch_4gpu_50k.sh --config "$SELECTED_CONFIG" \
  --manifest "$PROXY_MANIFEST" --output-root "$PROXY_SPECA_ROOT" \
  --nonfinal-proxy --expected-records 1000 \
  --expected-per-class 1 --expected-num-classes 1000
for root in "$PROXY_FULL_ROOT" "$PROXY_SPECA_ROOT"; do
  "$PIXELGEN_PYTHON" scripts/validate_outputs.py --sample-dir "$root/samples" \
    --metadata-dir "$root/metadata" --run-metadata "$root/run_manifest.json" \
    --manifest "$PROXY_MANIFEST" --expected-count 1000 \
    --expected-per-class 1 --expected-num-classes 1000 --resolution 256
done
bash scripts/evaluate_paired.sh --reference-dir "$PROXY_FULL_ROOT/samples" \
  --candidate-dir "$PROXY_SPECA_ROOT/samples" \
  --reference-manifest "$PROXY_MANIFEST" --candidate-manifest "$PROXY_MANIFEST" \
  --reference-run-metadata "$PROXY_FULL_ROOT/run_manifest.json" \
  --candidate-run-metadata "$PROXY_SPECA_ROOT/run_manifest.json" \
  --output-json "$OUTPUT_ROOT/proxy1k/pixelgen_paired_psnr_ssim.json" \
  --output-csv "$OUTPUT_ROOT/proxy1k/pixelgen_paired_psnr_ssim.csv" \
  --expected-count 1000 --expected-per-class 1 --expected-num-classes 1000 \
  --resolution 256 --skip-lpips
```

Analyze quality tails, latency/memory, action/check/fail ratios, local-error versus final damage, solver stage, classes, and failures. Prune settings before 8K.

## 9. Deferred independent 8K calibration

Create a no-header CSV from 1K with six columns `name,max_order,base_threshold,decay_rate,min_taylor_steps,max_taylor_steps`, constrained by [`configs/speca_sweep_template.yaml`](configs/speca_sweep_template.yaml).

```bash
: "${CANDIDATE_MATRIX:?set the 1K-pruned six-column CSV path}"
test -s "$CANDIDATE_MATRIX"
CANDIDATE_CONFIG_DIR="$OUTPUT_ROOT/configs/pixelgen_8k_candidates"
export CANDIDATE_MATRIX CANDIDATE_CONFIG_DIR
"$PIXELGEN_PYTHON" - <<'PY'
import copy,csv,os
from pathlib import Path
import yaml
base=yaml.safe_load(Path(os.environ["SELECTED_CONFIG"]).read_text())
out=Path(os.environ["CANDIDATE_CONFIG_DIR"]); out.mkdir(parents=True,exist_ok=True)
for row in csv.reader(Path(os.environ["CANDIDATE_MATRIX"]).open()):
    if not row: continue
    if len(row)!=6: raise ValueError(row)
    name,k,b,d,mn,mx=row
    if not name.replace("-","_").isalnum(): raise ValueError(name)
    cfg=copy.deepcopy(base); cfg["speca"].update(max_order=int(k),base_threshold=float(b),decay_rate=float(d),min_taylor_steps=int(mn),max_taylor_steps=int(mx),mode="speca",interval=None)
    cfg["model"]["denoiser"]["init_args"].update({f"speca_{key}":value for key,value in cfg["speca"].items() if key!="scheduler_mode"})
    (out/f"{name}.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
PY
VALIDATION_FULL_ROOT="$OUTPUT_ROOT/validation8k/pixelgen_instrumented_full"
bash scripts/launch_4gpu_50k.sh --config "$FULL_CONFIG" \
  --manifest "$VALIDATION_MANIFEST" --output-root "$VALIDATION_FULL_ROOT" \
  --nonfinal-proxy --expected-records 8000 \
  --expected-per-class 8 --expected-num-classes 1000
for config in "$CANDIDATE_CONFIG_DIR"/*.yaml; do
  name="$(basename "$config" .yaml)"; run="$OUTPUT_ROOT/validation8k/$name"
  bash scripts/launch_4gpu_50k.sh --config "$config" \
    --manifest "$VALIDATION_MANIFEST" --output-root "$run" \
    --nonfinal-proxy --expected-records 8000 \
    --expected-per-class 8 --expected-num-classes 1000
  "$PIXELGEN_PYTHON" scripts/validate_outputs.py --sample-dir "$run/samples" \
    --metadata-dir "$run/metadata" --run-metadata "$run/run_manifest.json" \
    --manifest "$VALIDATION_MANIFEST" --expected-count 8000 \
    --expected-per-class 8 --expected-num-classes 1000 --resolution 256
done
```

Select by measured matched latency/speedup plus quality/behavior tails, never Taylor ratio alone. Tune PixelGen independently. Set the five final values, rerun gate 3, freeze the hash, and never tune on final 50K.

## 10. Deferred matched benchmark

```bash
RUNNER_CONFIG="$OUTPUT_ROOT/benchmark/pixelgen_runner_b1.json"
mkdir -p "$(dirname "$RUNNER_CONFIG")"; export RUNNER_CONFIG
"$PIXELGEN_PYTHON" - <<'PY'
import json,os
from pathlib import Path
row=json.loads(Path(os.environ["MANIFEST"]).read_text().splitlines()[0])
p={"model_config":os.environ["SELECTED_CONFIG"],"batch_size":1,"sample_ids":[row["sample_id"]],"seeds":[row["seed"]],"class_ids":[row["class_id"]],"config_origin_dir":str(Path(os.environ["SELECTED_CONFIG"]).parent)}
Path(os.environ["RUNNER_CONFIG"]).write_text(json.dumps(p,indent=2)+"\n")
PY
export CUDA_VISIBLE_DEVICES=0 SPECA_GPU_TESTS_ALLOWED=1
bash scripts/benchmark_single_gpu.sh \
  --runner-factory speca_style.pixelgen_benchmark:build_benchmark_spec \
  --runner-config "$RUNNER_CONFIG" \
  --output-json "$OUTPUT_ROOT/benchmark/pixelgen_b1_matched.json" \
  --warmup-batches 10 --measured-batches 30
```

Report compile time separately, latency quantiles, `speedup`, action counts, verification/reduction/sync overhead, cache, and CUDA peaks. The matched `instrumented_full` closure must report zero Taylor cache/history updates and zero verifier calls; only adaptive-SpeCa Full actions update factors. Do not divide an upstream-compiled Full by eager SpeCa. Common-batch throughput is grouped-batch and separate.

## 11. Deferred four-GPU 50K

The current batch-4 Full is not a strict pair. Generate a new real-batch-1/effective-2 matched Full:

`launch_4gpu_50k.sh` defaults to the final profile and always enforces exactly
50,000 records, 50 records/class, 1,000 classes, and four ranks. Custom count
arguments are rejected unless `--nonfinal-proxy` is present; proxy mode requires
all three explicit counts, checks their product, requires even four-way sharding,
and refuses 50K. Thus omitting proxy flags below cannot silently weaken the final
gate.

```bash
FINAL_FULL_ROOT="$OUTPUT_ROOT/final/pixelgen_instrumented_full_b1"
FINAL_SPECA_ROOT="$OUTPUT_ROOT/final/pixelgen_speca_b1"
export FINAL_FULL_ROOT FINAL_SPECA_ROOT
export CUDA_VISIBLE_DEVICES=0,1,2,3 SPECA_GPU_TESTS_ALLOWED=1
bash scripts/launch_4gpu_50k.sh --config "$FULL_CONFIG" \
  --manifest "$MANIFEST" --output-root "$FINAL_FULL_ROOT"
bash scripts/launch_4gpu_50k.sh --config "$SELECTED_CONFIG" \
  --manifest "$MANIFEST" --output-root "$FINAL_SPECA_ROOT"
```

For an interrupted matching run only, repeat the identical command with `--resume`. Changed archived inputs and incompatible/non-empty roots are rejected; every prediction batch resets SpeCa state.

## 12. Output validation

```bash
for root in "$FINAL_FULL_ROOT" "$FINAL_SPECA_ROOT"; do
  "$PIXELGEN_PYTHON" scripts/validate_outputs.py --sample-dir "$root/samples" \
    --metadata-dir "$root/metadata" --run-metadata "$root/run_manifest.json" \
    --manifest "$MANIFEST" --expected-count 50000 \
    --expected-per-class 50 --expected-num-classes 1000 --resolution 256
done
```

Do not evaluate until count/class balance, decode/RGB/uint8/256, manifest/config/checkpoint hashes, metadata, and shard coverage pass.

## 13. Distribution metrics

```bash
: "${IMAGENET_REFERENCE_NPZ:?set the local ImageNet-256 ADM reference NPZ}"
: "${ADM_EVALUATOR:?set the local ADM evaluator.py}"
test -s "$IMAGENET_REFERENCE_NPZ" && test -s "$ADM_EVALUATOR"
mkdir -p "$OUTPUT_ROOT/metrics"
bash scripts/evaluate_distribution.sh --sample-dir "$FINAL_FULL_ROOT/samples" \
  --manifest "$MANIFEST" --reference-npz "$IMAGENET_REFERENCE_NPZ" \
  --evaluator "$ADM_EVALUATOR" --run-metadata "$FINAL_FULL_ROOT/run_manifest.json" \
  --output-json "$OUTPUT_ROOT/metrics/pixelgen_full_distribution.json" \
  --expected-count 50000 --expected-per-class 50 --expected-num-classes 1000
bash scripts/evaluate_distribution.sh --sample-dir "$FINAL_SPECA_ROOT/samples" \
  --manifest "$MANIFEST" --reference-npz "$IMAGENET_REFERENCE_NPZ" \
  --evaluator "$ADM_EVALUATOR" --run-metadata "$FINAL_SPECA_ROOT/run_manifest.json" \
  --output-json "$OUTPUT_ROOT/metrics/pixelgen_speca_distribution.json" \
  --expected-count 50000 --expected-per-class 50 --expected-num-classes 1000
"$PIXELGEN_PYTHON" scripts/compare_distribution.py \
  --full-json "$OUTPUT_ROOT/metrics/pixelgen_full_distribution.json" \
  --speca-json "$OUTPUT_ROOT/metrics/pixelgen_speca_distribution.json" \
  --output-json "$OUTPUT_ROOT/metrics/pixelgen_distribution_delta.json"
```

No evaluator/reference download or fallback is allowed; evaluator time is not generation latency.

## 14. Strict paired metrics

Only the new manifest-backed batch-1 matched Full is eligible:

```bash
bash scripts/evaluate_paired.sh --reference-dir "$FINAL_FULL_ROOT/samples" \
  --candidate-dir "$FINAL_SPECA_ROOT/samples" \
  --reference-manifest "$MANIFEST" --candidate-manifest "$MANIFEST" \
  --reference-run-metadata "$FINAL_FULL_ROOT/run_manifest.json" \
  --candidate-run-metadata "$FINAL_SPECA_ROOT/run_manifest.json" \
  --output-json "$OUTPUT_ROOT/metrics/pixelgen_paired.json" \
  --output-csv "$OUTPUT_ROOT/metrics/pixelgen_paired.csv" \
  --expected-count 50000 --expected-per-class 50 --expected-num-classes 1000 \
  --resolution 256 --lpips-device cpu --lpips-batch-size 16
```

If local LPIPS/AlexNet assets are absent, add `--skip-lpips` and label the artifact PSNR/SSIM-only. Never install/download or pair the old batch-4 Full by filename.

## 15. Trace aggregation and archive

The launcher writes `four_gpu_wall_clock.json`. Use the self-contained trace aggregator, which deduplicates one trajectory per batch group and validates the requested method:

```bash
TRACE_JSON="$OUTPUT_ROOT/metrics/pixelgen_speca_trace_aggregate.json"; export TRACE_JSON
"$PIXELGEN_PYTHON" scripts/aggregate_speca_trace.py \
  --metadata-dir "$FINAL_SPECA_ROOT/metadata" --world-size 4 \
  --require-method speca --output-json "$TRACE_JSON"
sha256sum "$SELECTED_CONFIG" "$MANIFEST" "$MANIFEST.meta.json" \
  "$FINAL_SPECA_ROOT/run_manifest.json" "$TRACE_JSON"
```

Archive selected config/hash, manifest/sidecar/hash, source/config/checkpoint identities, run manifests, rank metadata/summaries/logs, wall clock, benchmark, analytic/observed memory, distribution/delta, paired JSON/CSV, and trace aggregation. Verify 99 combined calls, q 98→0, action/reason totals, next-error naming, combined CFG/deepcopy/reset, and maxima. Report real-batch-1 latency, any grouped throughput, and four-GPU wall time separately; record OOM/failure without changing batch, order, dtype, or method silently.
