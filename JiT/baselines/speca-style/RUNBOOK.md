# JiT SpeCa execution runbook

Run the gates in order. Gates 4–15 require a future, separately authorized GPU window and were **not executed** in this task. Never set `SPECA_GPU_TESTS_ALLOWED=1` merely because a GPU looks idle; obtain an allocation first. The wrappers refuse busy GPUs and never kill another process.

## 1. Environment audit and CPU gate

Do not assume the current directory:

```bash
: "${PIXARC_ROOT:?set the local PixARC checkout}"
PIXARC_ROOT="$(git -C "$PIXARC_ROOT" rev-parse --show-toplevel)"
BASE="$PIXARC_ROOT/JiT/baselines/speca-style"
UPSTREAM_JIT="$PIXARC_ROOT/third-party/JiT"
: "${JIT_PYTHON:?set the JiT-environment Python executable}"
: "${CHECKPOINT:?set the JiT-B/16 checkpoint path}"
: "${OUTPUT_ROOT:?set an external output root}"
CHECKPOINT="$(realpath "$CHECKPOINT")"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
test -s "$CHECKPOINT" && test -d "$UPSTREAM_JIT"
[[ "$OUTPUT_ROOT/" != "$PIXARC_ROOT/"* ]]
mkdir -p "$OUTPUT_ROOT"
export PIXARC_ROOT BASE UPSTREAM_JIT JIT_PYTHON CHECKPOINT OUTPUT_ROOT
export PATH="$(dirname "$JIT_PYTHON"):$PATH"
export PYTHONPATH="$UPSTREAM_JIT:$BASE${PYTHONPATH:+:$PYTHONPATH}"
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
  PYTHONPYCACHEPREFIX=/tmp/pixarc-jit-speca-pycache \
  "$JIT_PYTHON" -m compileall -q "$BASE"
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  "$JIT_PYTHON" -m pytest -q -p no:cacheprovider "$BASE/tests"
CUDA_VISIBLE_DEVICES="" "$JIT_PYTHON" scripts/compare_common_tool_interfaces.py \
  --pixarc-root "$PIXARC_ROOT" \
  --output-json "$OUTPUT_ROOT/jit_common_interface_check.json"
for script in scripts/*.sh; do bash -n "$script"; done
git -C "$PIXARC_ROOT" diff --check
```

Require official scheduler/error fixtures, signed-gap/exact-only predictor parity, previous-error/no-rollback, CFG stream aggregation, Heun counts, context, state-dict/reset, memory, manifest/sharding, paired-toy, metadata, and common-core tests. CPU success is not real-model parity.

## 2. Immutable batch-32 manifests

Use four disjoint signed-63-bit seed ranges:

```bash
: "${SMOKE_BASE_SEED:?set an 8-image seed base}"
: "${PROXY_BASE_SEED:?set a disjoint 1K seed base}"
: "${VALIDATION_BASE_SEED:?set a disjoint 8K seed base}"
: "${FINAL_BASE_SEED:?set a disjoint 50K seed base}"
BATCH_SIZE=32
SMOKE_MANIFEST="$OUTPUT_ROOT/manifests/jit_smoke8_b32.jsonl"
PROXY_MANIFEST="$OUTPUT_ROOT/manifests/jit_imagenet1k_b32.jsonl"
VALIDATION_MANIFEST="$OUTPUT_ROOT/manifests/jit_imagenet8k_b32.jsonl"
MANIFEST="$OUTPUT_ROOT/manifests/jit_imagenet50k_b32.jsonl"
export BATCH_SIZE SMOKE_MANIFEST PROXY_MANIFEST VALIDATION_MANIFEST MANIFEST
mkdir -p "$OUTPUT_ROOT/manifests"

"$JIT_PYTHON" scripts/build_manifest.py --output "$SMOKE_MANIFEST" \
  --samples-per-class 1 --num-classes 8 --base-seed "$SMOKE_BASE_SEED" \
  --split-name smoke8 --world-size 1 --batch-size 32 \
  --generator-device cuda --noise-dtype float32 --noise-shape 3 256 256
"$JIT_PYTHON" scripts/build_manifest.py --output "$PROXY_MANIFEST" \
  --samples-per-class 1 --base-seed "$PROXY_BASE_SEED" \
  --split-name imagenet1k_proxy --world-size 4 --batch-size 32 \
  --generator-device cuda --noise-dtype float32 --noise-shape 3 256 256
"$JIT_PYTHON" scripts/build_manifest.py --output "$VALIDATION_MANIFEST" \
  --samples-per-class 8 --base-seed "$VALIDATION_BASE_SEED" \
  --split-name imagenet8k_validation --world-size 4 --batch-size 32 \
  --generator-device cuda --noise-dtype float32 --noise-shape 3 256 256
"$JIT_PYTHON" scripts/build_manifest.py --output "$MANIFEST" \
  --samples-per-class 50 --base-seed "$FINAL_BASE_SEED" \
  --split-name imagenet50k_final --world-size 4 --batch-size 32 \
  --generator-device cuda --noise-dtype float32 --noise-shape 3 256 256
```

Validate sidecars, class balance, fixed batch-32 groups (with only the final group allowed to be partial), seed rules, four shards, and disjointness:

```bash
"$JIT_PYTHON" scripts/validate_manifest.py --manifest "$SMOKE_MANIFEST" \
  --expected-count 8 --expected-per-class 1 --expected-num-classes 8 \
  --world-size 1 --batch-size 32 --base-seed "$SMOKE_BASE_SEED" \
  --require-sidecar --disjoint-with "$PROXY_MANIFEST" \
  --disjoint-with "$VALIDATION_MANIFEST" --disjoint-with "$MANIFEST"
"$JIT_PYTHON" scripts/validate_manifest.py --manifest "$PROXY_MANIFEST" \
  --expected-count 1000 --expected-per-class 1 --expected-num-classes 1000 \
  --world-size 4 --batch-size 32 --base-seed "$PROXY_BASE_SEED" \
  --require-sidecar --disjoint-with "$SMOKE_MANIFEST" \
  --disjoint-with "$VALIDATION_MANIFEST" --disjoint-with "$MANIFEST"
"$JIT_PYTHON" scripts/validate_manifest.py --manifest "$VALIDATION_MANIFEST" \
  --expected-count 8000 --expected-per-class 8 --expected-num-classes 1000 \
  --world-size 4 --batch-size 32 --base-seed "$VALIDATION_BASE_SEED" \
  --require-sidecar --disjoint-with "$SMOKE_MANIFEST" \
  --disjoint-with "$PROXY_MANIFEST" --disjoint-with "$MANIFEST"
"$JIT_PYTHON" scripts/validate_manifest.py --manifest "$MANIFEST" \
  --expected-count 50000 --expected-per-class 50 --expected-num-classes 1000 \
  --world-size 4 --batch-size 32 --base-seed "$FINAL_BASE_SEED" \
  --require-sidecar --disjoint-with "$SMOKE_MANIFEST" \
  --disjoint-with "$PROXY_MANIFEST" --disjoint-with "$VALIDATION_MANIFEST"
sha256sum "$OUTPUT_ROOT"/manifests/*.jsonl "$OUTPUT_ROOT"/manifests/*.meta.json
```

Never edit a manifest/sidecar after use. A final shard must contain exactly 12,500 samples.

## 3. Memory estimate and resolved configs

Set a provisional point for smoke/1K; the official defaults are not claimed optimal:

```bash
: "${MAX_ORDER:?set provisional/selected max order}"
: "${BASE_THRESHOLD:?set provisional/selected base threshold}"
: "${DECAY_RATE:?set provisional/selected decay rate}"
: "${MIN_TAYLOR_STEPS:?set provisional/selected minimum Taylor span}"
: "${MAX_TAYLOR_STEPS:?set provisional/selected maximum Taylor span}"
[[ "$BATCH_SIZE" == 32 ]]
mkdir -p "$OUTPUT_ROOT/memory" "$OUTPUT_ROOT/configs/jit"
"$JIT_PYTHON" scripts/estimate_cache_memory.py --preset jit-b16-256 \
  --batch-size "$BATCH_SIZE" --max-order "$MAX_ORDER" \
  --cache-dtype bfloat16 --verify-layer -1 \
  --output-json "$OUTPUT_ROOT/memory/jit_b32_k${MAX_ORDER}.json"
```

Materialize absolute-checkpoint configs outside the repository. Re-run after 8K selection:

```bash
CONFIG_DIR="$OUTPUT_ROOT/configs/jit"
export CONFIG_DIR MAX_ORDER BASE_THRESHOLD DECAY_RATE \
  MIN_TAYLOR_STEPS MAX_TAYLOR_STEPS
"$JIT_PYTHON" - <<'PY'
import os
from pathlib import Path
import yaml
base, out = Path(os.environ["BASE"]), Path(os.environ["CONFIG_DIR"])
out.mkdir(parents=True, exist_ok=True)
cfgs = {}
for name in ("upstream_full", "instrumented_full", "taylor_draft_fixed", "shadow_verify"):
    cfg = yaml.safe_load((base/"configs"/f"jit_b16_256_{name}.yaml").read_text())
    cfg["model"]["checkpoint"] = os.environ["CHECKPOINT"]
    cfgs[name] = cfg
selected = yaml.safe_load((base/"configs/jit_b16_256_speca.yaml").read_text())
selected["model"]["checkpoint"] = os.environ["CHECKPOINT"]
values = dict(max_order=int(os.environ["MAX_ORDER"]),
    base_threshold=float(os.environ["BASE_THRESHOLD"]),
    decay_rate=float(os.environ["DECAY_RATE"]),
    min_taylor_steps=int(os.environ["MIN_TAYLOR_STEPS"]),
    max_taylor_steps=int(os.environ["MAX_TAYLOR_STEPS"]))
selected["speca"].update(values); cfgs["speca_selected"] = selected
cfgs["shadow_verify"]["speca"].update(values)
cfgs["taylor_draft_fixed"]["speca"]["max_order"] = values["max_order"]
for name, cfg in cfgs.items():
    (out/f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
UPSTREAM_CONFIG="$CONFIG_DIR/upstream_full.yaml"
FULL_CONFIG="$CONFIG_DIR/instrumented_full.yaml"
FIXED_CONFIG="$CONFIG_DIR/taylor_draft_fixed.yaml"
SHADOW_CONFIG="$CONFIG_DIR/shadow_verify.yaml"
SELECTED_CONFIG="$CONFIG_DIR/speca_selected.yaml"
export UPSTREAM_CONFIG FULL_CONFIG FIXED_CONFIG SHADOW_CONFIG SELECTED_CONFIG
sha256sum "$CONFIG_DIR"/*.yaml
```

The final candidate must retain released-code mode, relative L1/`1e-10`, first-enhance 3, floor 0.01, last-block/all-token verification, batch-global gate, NFE coordinate, `interval: null`, no final-Full addition, inherited cache dtype, batch 32, and matched compile mode. The threshold is selected under this grouped-batch protocol.

## 4. Deferred smoke

```bash
: "${CUDA_VISIBLE_DEVICES:?set exactly one allocated idle GPU}"
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]]
export CUDA_VISIBLE_DEVICES SPECA_GPU_TESTS_ALLOWED=1
bash scripts/inspect_active_runs.sh
bash scripts/run_deferred_smoke_tests.sh \
  --full-config "$FULL_CONFIG" --candidate-config "$SELECTED_CONFIG" \
  --manifest "$SMOKE_MANIFEST" \
  --output-root "$OUTPUT_ROOT/smoke/jit_instrumented_vs_speca"
```

Require checkpoint/EMA1, finite output, both actions, checking, 99 decisions/198 forwards, context, reset, and no NaN. Stop on failure.

## 5. Deferred Full parity

```bash
bash scripts/inspect_active_runs.sh
"$JIT_PYTHON" scripts/deferred_run_guard.py --config "$UPSTREAM_CONFIG" \
  --manifest "$SMOKE_MANIFEST" --max-records 8 --require-mode upstream_full
"$JIT_PYTHON" scripts/deferred_run_guard.py --config "$FULL_CONFIG" \
  --manifest "$SMOKE_MANIFEST" --max-records 8 --require-mode instrumented_full
SPECA_INVOCATION_ID="parity-upstream-$(date +%s%N)" \
  "$JIT_PYTHON" scripts/generate_shard.py --config "$UPSTREAM_CONFIG" \
  --config-origin-dir "$CONFIG_DIR" --manifest "$SMOKE_MANIFEST" \
  --shard-id 0 --world-size 1 \
  --output-root "$OUTPUT_ROOT/parity/jit_upstream_vs_instrumented/full" \
  --acknowledge-gpu-job
SPECA_INVOCATION_ID="parity-instrumented-$(date +%s%N)" \
  "$JIT_PYTHON" scripts/generate_shard.py --config "$FULL_CONFIG" \
  --config-origin-dir "$CONFIG_DIR" --manifest "$SMOKE_MANIFEST" \
  --shard-id 0 --world-size 1 \
  --output-root "$OUTPUT_ROOT/parity/jit_upstream_vs_instrumented/candidate" \
  --acknowledge-gpu-job
for side in full candidate; do
  "$JIT_PYTHON" scripts/validate_outputs.py \
    --sample-dir "$OUTPUT_ROOT/parity/jit_upstream_vs_instrumented/$side/samples" \
    --metadata-dir "$OUTPUT_ROOT/parity/jit_upstream_vs_instrumented/$side/metadata" \
    --run-metadata "$OUTPUT_ROOT/parity/jit_upstream_vs_instrumented/$side/run_manifest.json" \
    --manifest "$SMOKE_MANIFEST" --expected-count 8 \
    --expected-per-class 1 --expected-num-classes 8 --resolution 256
done
FULL_PARITY_MAX_U8="${FULL_PARITY_MAX_U8:-1}"
FULL_PARITY_MEAN_U8="${FULL_PARITY_MEAN_U8:-0.01}"
export FULL_PARITY_MAX_U8 FULL_PARITY_MEAN_U8
"$JIT_PYTHON" - \
  "$OUTPUT_ROOT/parity/jit_upstream_vs_instrumented/full/samples" \
  "$OUTPUT_ROOT/parity/jit_upstream_vs_instrumented/candidate/samples" <<'PY'
import os, sys
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

On failure, locate the first block/module mismatch; do not tune or generate 50K.

## 6. Deferred draft parity

```bash
CUDA_VISIBLE_DEVICES="" "$JIT_PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_taylor_predictor_parity.py tests/test_exact_only_history.py
CUDA_VISIBLE_DEVICES="" "$JIT_PYTHON" scripts/compare_taylor_predictor_parity.py \
  --max-order "$MAX_ORDER" --dtype float64 \
  --output-json "$OUTPUT_ROOT/parity/jit_taylor_predictor_cpu.json"
"$JIT_PYTHON" scripts/deferred_run_guard.py --config "$FIXED_CONFIG" \
  --manifest "$SMOKE_MANIFEST" --max-records 8 \
  --require-mode taylor_draft_fixed
SPECA_INVOCATION_ID="parity-fixed-$(date +%s%N)" \
  "$JIT_PYTHON" scripts/generate_shard.py --config "$FIXED_CONFIG" \
  --config-origin-dir "$CONFIG_DIR" --manifest "$SMOKE_MANIFEST" \
  --shard-id 0 --world-size 1 \
  --output-root "$OUTPUT_ROOT/parity/jit_fixed_draft" \
  --acknowledge-gpu-job
```

Require the same exact schedule/coordinate/order, factors, forecast, output, tensor count, and available order as the self-contained TaylorSeer oracle. Fixed-draft quality is not a SpeCa result.

## 7. Deferred verification semantics

```bash
bash scripts/run_shadow_diagnostic.sh --config "$SHADOW_CONFIG" \
  --manifest "$SMOKE_MANIFEST" --output-root "$OUTPUT_ROOT/shadow/jit"
CUDA_VISIBLE_DEVICES="" "$JIT_PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_previous_error_semantics.py tests/test_no_current_rollback.py \
  tests/test_verification_layer.py tests/test_verification_prefix.py \
  tests/test_cfg_error_aggregation.py
```

Confirm the last block is exact from the speculative prefix, cond/uncond sufficient statistics equal concatenation, current draft is retained, and a failure forces only the next NFE. Shadow work is excluded from performance and 50K.

## 8. Deferred 1K proxy

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 SPECA_GPU_TESTS_ALLOWED=1
bash scripts/inspect_active_runs.sh
PROXY_FULL_ROOT="$OUTPUT_ROOT/proxy1k/jit_instrumented_full"
PROXY_SPECA_ROOT="$OUTPUT_ROOT/proxy1k/jit_speca"
bash scripts/launch_4gpu_50k.sh --config "$FULL_CONFIG" \
  --manifest "$PROXY_MANIFEST" --output-root "$PROXY_FULL_ROOT" \
  --nonfinal-proxy --expected-records 1000 \
  --expected-per-class 1 --expected-num-classes 1000 \
  --gpu-ids 0,1,2,3
bash scripts/launch_4gpu_50k.sh --config "$SELECTED_CONFIG" \
  --manifest "$PROXY_MANIFEST" --output-root "$PROXY_SPECA_ROOT" \
  --nonfinal-proxy --expected-records 1000 \
  --expected-per-class 1 --expected-num-classes 1000 \
  --gpu-ids 0,1,2,3
for root in "$PROXY_FULL_ROOT" "$PROXY_SPECA_ROOT"; do
  "$JIT_PYTHON" scripts/validate_outputs.py --sample-dir "$root/samples" \
    --metadata-dir "$root/metadata" --run-metadata "$root/run_manifest.json" \
    --manifest "$PROXY_MANIFEST" --expected-count 1000 \
    --expected-per-class 1 --expected-num-classes 1000 --resolution 256
done
bash scripts/evaluate_paired.sh --reference-dir "$PROXY_FULL_ROOT/samples" \
  --candidate-dir "$PROXY_SPECA_ROOT/samples" \
  --reference-manifest "$PROXY_MANIFEST" --candidate-manifest "$PROXY_MANIFEST" \
  --reference-run-metadata "$PROXY_FULL_ROOT/run_manifest.json" \
  --candidate-run-metadata "$PROXY_SPECA_ROOT/run_manifest.json" \
  --output-json "$OUTPUT_ROOT/proxy1k/jit_paired_psnr_ssim.json" \
  --output-csv "$OUTPUT_ROOT/proxy1k/jit_paired_psnr_ssim.csv" \
  --expected-count 1000 --expected-per-class 1 --expected-num-classes 1000 \
  --resolution 256 --skip-lpips
```

Analyze quality tails, latency/memory, action/check/fail ratios, error-versus-terminal damage, solver stage, and failure examples. Prune nonsensical settings before 8K.

## 9. Deferred independent 8K calibration

Create a no-header CSV from the 1K screen with six columns: `name,max_order,base_threshold,decay_rate,min_taylor_steps,max_taylor_steps`; values must respect [`configs/speca_sweep_template.yaml`](configs/speca_sweep_template.yaml).

```bash
: "${CANDIDATE_MATRIX:?set the 1K-pruned six-column CSV path}"
test -s "$CANDIDATE_MATRIX"
CANDIDATE_CONFIG_DIR="$OUTPUT_ROOT/configs/jit_8k_candidates"
export CANDIDATE_MATRIX CANDIDATE_CONFIG_DIR
"$JIT_PYTHON" - <<'PY'
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
    (out/f"{name}.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
PY
VALIDATION_FULL_ROOT="$OUTPUT_ROOT/validation8k/jit_instrumented_full"
bash scripts/launch_4gpu_50k.sh --config "$FULL_CONFIG" \
  --manifest "$VALIDATION_MANIFEST" --output-root "$VALIDATION_FULL_ROOT" \
  --nonfinal-proxy --expected-records 8000 \
  --expected-per-class 8 --expected-num-classes 1000 \
  --gpu-ids 0,1,2,3
for config in "$CANDIDATE_CONFIG_DIR"/*.yaml; do
  name="$(basename "$config" .yaml)"; run="$OUTPUT_ROOT/validation8k/$name"
  bash scripts/launch_4gpu_50k.sh --config "$config" \
    --manifest "$VALIDATION_MANIFEST" --output-root "$run" \
    --nonfinal-proxy --expected-records 8000 \
    --expected-per-class 8 --expected-num-classes 1000 \
    --gpu-ids 0,1,2,3
  "$JIT_PYTHON" scripts/validate_outputs.py --sample-dir "$run/samples" \
    --metadata-dir "$run/metadata" --run-metadata "$run/run_manifest.json" \
    --manifest "$VALIDATION_MANIFEST" --expected-count 8000 \
    --expected-per-class 8 --expected-num-classes 1000 --resolution 256
done
```

Select points by measured matched latency/speedup plus quality and behavior tails, never Taylor ratio alone. Tune JiT independently. Set the five final environment values from the chosen row, rerun gate 3, freeze the selected hash, and never tune on 50K.

## 10. Deferred matched benchmark

```bash
RUNNER_CONFIG="$OUTPUT_ROOT/benchmark/jit_runner_b32.json"
mkdir -p "$(dirname "$RUNNER_CONFIG")"; export RUNNER_CONFIG
"$JIT_PYTHON" - <<'PY'
import json,os
from pathlib import Path
row=json.loads(Path(os.environ["MANIFEST"]).read_text().splitlines()[0])
p={"model_config":os.environ["SELECTED_CONFIG"],"batch_size":1,"sample_ids":[row["sample_id"]],"seeds":[row["seed"]],"class_ids":[row["class_id"]],"config_origin_dir":str(Path(os.environ["SELECTED_CONFIG"]).parent)}
Path(os.environ["RUNNER_CONFIG"]).write_text(json.dumps(p,indent=2)+"\n")
PY
export CUDA_VISIBLE_DEVICES=0 SPECA_GPU_TESTS_ALLOWED=1
bash scripts/benchmark_single_gpu.sh \
  --runner-factory speca_style.jit_benchmark:build_benchmark_spec \
  --runner-config "$RUNNER_CONFIG" \
  --output-json "$OUTPUT_ROOT/benchmark/jit_b32_matched.json" \
  --warmup-batches 10 --measured-batches 30
```

Report compile time separately, steady latency quantiles, `speedup`, action counts, verification/reduction/sync overhead, cache, and CUDA peaks. The matched `instrumented_full` closure must report zero Taylor cache/history updates and zero verifier calls; only adaptive-SpeCa Full actions update factors. Never divide upstream-compiled Full by eager SpeCa.

## 11. Deferred four-GPU 50K

Generate a new manifest-matched batch-32 Full; do not pair against legacy outputs that lack the immutable per-sample noise contract:

`launch_4gpu_50k.sh` defaults to the final profile and always enforces exactly
50,000 records, 50 records/class, 1,000 classes, and four ranks. Custom count
arguments are rejected unless `--nonfinal-proxy` is present; proxy mode requires
all three explicit counts, checks their product, requires even four-way sharding,
and refuses 50K. Thus omitting proxy flags below cannot silently weaken the final
gate.

```bash
FINAL_FULL_ROOT="$OUTPUT_ROOT/final/jit_instrumented_full_b32"
FINAL_SPECA_ROOT="$OUTPUT_ROOT/final/jit_speca_b32"
export FINAL_FULL_ROOT FINAL_SPECA_ROOT
export CUDA_VISIBLE_DEVICES=0,1,2,3 SPECA_GPU_TESTS_ALLOWED=1
bash scripts/launch_4gpu_50k.sh --config "$FULL_CONFIG" \
  --manifest "$MANIFEST" --output-root "$FINAL_FULL_ROOT" \
  --gpu-ids 0,1,2,3
bash scripts/launch_4gpu_50k.sh --config "$SELECTED_CONFIG" \
  --manifest "$MANIFEST" --output-root "$FINAL_SPECA_ROOT" \
  --gpu-ids 0,1,2,3
```

For an interrupted matching run only, repeat the identical command with `--resume`. Changed archived config/manifest/sidecar and incompatible non-empty roots are rejected.

## 12. Output validation

```bash
for root in "$FINAL_FULL_ROOT" "$FINAL_SPECA_ROOT"; do
  "$JIT_PYTHON" scripts/validate_outputs.py --sample-dir "$root/samples" \
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
  --output-json "$OUTPUT_ROOT/metrics/jit_full_distribution.json" \
  --expected-count 50000 --expected-per-class 50 --expected-num-classes 1000
bash scripts/evaluate_distribution.sh --sample-dir "$FINAL_SPECA_ROOT/samples" \
  --manifest "$MANIFEST" --reference-npz "$IMAGENET_REFERENCE_NPZ" \
  --evaluator "$ADM_EVALUATOR" --run-metadata "$FINAL_SPECA_ROOT/run_manifest.json" \
  --output-json "$OUTPUT_ROOT/metrics/jit_speca_distribution.json" \
  --expected-count 50000 --expected-per-class 50 --expected-num-classes 1000
"$JIT_PYTHON" scripts/compare_distribution.py \
  --full-json "$OUTPUT_ROOT/metrics/jit_full_distribution.json" \
  --speca-json "$OUTPUT_ROOT/metrics/jit_speca_distribution.json" \
  --output-json "$OUTPUT_ROOT/metrics/jit_distribution_delta.json"
```

No evaluator/reference download or fallback is allowed; evaluator time is not generation latency.

## 14. Strict paired metrics

Only the new manifest-backed matched Full is eligible:

```bash
bash scripts/evaluate_paired.sh --reference-dir "$FINAL_FULL_ROOT/samples" \
  --candidate-dir "$FINAL_SPECA_ROOT/samples" \
  --reference-manifest "$MANIFEST" --candidate-manifest "$MANIFEST" \
  --reference-run-metadata "$FINAL_FULL_ROOT/run_manifest.json" \
  --candidate-run-metadata "$FINAL_SPECA_ROOT/run_manifest.json" \
  --output-json "$OUTPUT_ROOT/metrics/jit_paired.json" \
  --output-csv "$OUTPUT_ROOT/metrics/jit_paired.csv" \
  --expected-count 50000 --expected-per-class 50 --expected-num-classes 1000 \
  --resolution 256 --lpips-device cpu --lpips-batch-size 16
```

If local LPIPS/AlexNet assets are absent, add `--skip-lpips` and label the artifact PSNR/SSIM-only. Never install/download or pair the old batch-32 Full by filename.

## 15. Trace aggregation and archive

The launcher writes `four_gpu_wall_clock.json`. Use the self-contained trace aggregator, which deduplicates one trajectory per batch group and validates the requested method:

```bash
TRACE_JSON="$OUTPUT_ROOT/metrics/jit_speca_trace_aggregate.json"; export TRACE_JSON
"$JIT_PYTHON" scripts/aggregate_speca_trace.py \
  --metadata-dir "$FINAL_SPECA_ROOT/metadata" --world-size 4 \
  --require-method speca --output-json "$TRACE_JSON"
sha256sum "$SELECTED_CONFIG" "$MANIFEST" "$MANIFEST.meta.json" \
  "$FINAL_SPECA_ROOT/run_manifest.json" "$TRACE_JSON"
```

Archive selected config/hash, manifest/sidecar/hash, source/config/checkpoint identities, run manifests, rank metadata/summaries/logs, wall clock, benchmark, analytic/observed memory, distribution/delta, paired JSON/CSV, and trace aggregation. Verify 99/198 calls, q 98→0, action/reason totals, failure-versus-next-Full naming, reset, and maxima. Report batch-32 per-image latency, throughput, and four-GPU wall time; record OOM/failure without changing batch, order, dtype, or method silently.
