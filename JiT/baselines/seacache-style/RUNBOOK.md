# JiT SeaCache-style execution runbook

Run phases in order. At the time this runbook was written PixelGen was ACTIVE
and JiT Full was SCHEDULED; therefore phases 4 onward were **deferred and not
executed**.

## 0. Safety gate

Read-only inspection:

```bash
ROOT="$(git rev-parse --show-toplevel)"
BASE="$ROOT/JiT/baselines/seacache-style"
cd "$BASE"
bash scripts/inspect_active_runs.sh
```

Do not proceed if any target GPU has a compute process, material memory use, or
material utilization. Never kill or reset another job. The launch tools refuse
GPU execution until the operator deliberately sets:

```bash
export SEACACHE_GPU_TESTS_ALLOWED=1
```

Set it only after all current reference jobs have finished and outputs were
validated.

## 1. Environment and immutable inputs

```bash
UPSTREAM_JIT="$ROOT/third-party/JiT"
CHECKPOINT="$ROOT/JiT/checkpoints/JiT-B-16-256/checkpoint-last.pth"
: "${JIT_PYTHON:?set the executable from the JiT Python environment}"
: "${OUTPUT_ROOT:?set an output directory outside the PixARC source tree}"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
[[ "$OUTPUT_ROOT/" != "$ROOT/"* ]] || {
  echo "OUTPUT_ROOT must be outside $ROOT" >&2
  return 2 2>/dev/null || exit 2
}
IMAGENET_REFERENCE_NPZ="${IMAGENET_REFERENCE_NPZ:?set local ImageNet-256 ADM NPZ}"
ADM_EVALUATOR="${ADM_EVALUATOR:?set local ADM evaluator.py}"

test -x "$JIT_PYTHON"
export ROOT BASE UPSTREAM_JIT CHECKPOINT OUTPUT_ROOT JIT_PYTHON
export PATH="$(dirname "$JIT_PYTHON"):$PATH"
export PYTHONPATH="$UPSTREAM_JIT:$BASE:${PYTHONPATH:-}"
test -s "$CHECKPOINT"
test -s "$IMAGENET_REFERENCE_NPZ"
test -f "$ADM_EVALUATOR"
mkdir -p "$OUTPUT_ROOT"
```

Do not download or replace a checkpoint/reference inside this workflow.

## 2. CPU-safe verification

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/pixarc-jit-seacache-pycache \
  "$JIT_PYTHON" -m compileall "$BASE"
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  "$JIT_PYTHON" -m pytest -q -p no:cacheprovider "$BASE/tests"
git -C "$ROOT" diff --check -- JiT/baselines/seacache-style
```

`git diff --check` omits untracked files. Inspect `git status --short`; after
intentionally staging new files, also run
`git -C "$ROOT" diff --cached --check`.

Final implementation-time result: `37 passed, 2 subtests passed`, zero failures and
zero skips. This does not replace GPU parity.

## 3. Freeze manifests

Choose mutually disjoint ranges for smoke, 1K proxy, 8K validation, and final
50K. The range length is the sample count. Every manifest remains outside the
source tree under `OUTPUT_ROOT`.

```bash
: "${SMOKE_BASE_SEED:?set 8-sample smoke base seed}"
: "${PROXY_BASE_SEED:?set 1K base seed}"
: "${VALIDATION_BASE_SEED:?set 8K base seed}"
: "${FINAL_BASE_SEED:?set disjoint 50K base seed}"

SMOKE_MANIFEST="$OUTPUT_ROOT/manifests/smoke8.jsonl"
PROXY_MANIFEST="$OUTPUT_ROOT/manifests/imagenet1k.jsonl"
VALIDATION_MANIFEST="$OUTPUT_ROOT/manifests/imagenet8k.jsonl"
FINAL_MANIFEST="$OUTPUT_ROOT/manifests/imagenet50k.jsonl"
export SMOKE_BASE_SEED SMOKE_MANIFEST PROXY_MANIFEST
export VALIDATION_MANIFEST FINAL_MANIFEST

python -c 'import os; from seacache_style.manifest import build_manifest, write_manifest; seed=int(os.environ["SMOKE_BASE_SEED"]); rows=build_manifest(samples_per_class=1, base_seed=seed, split_name="smoke8", world_size=1, batch_size=2, num_classes=8); write_manifest(rows, os.environ["SMOKE_MANIFEST"], base_seed=seed, world_size=1, batch_size=2)'

python scripts/build_manifest.py --output "$PROXY_MANIFEST" \
  --samples-per-class 1 --base-seed "$PROXY_BASE_SEED" \
  --split-name imagenet1k_proxy --world-size 4 --batch-size 32

python scripts/build_manifest.py --output "$VALIDATION_MANIFEST" \
  --samples-per-class 8 --base-seed "$VALIDATION_BASE_SEED" \
  --split-name imagenet8k_validation --world-size 4 --batch-size 32

python scripts/build_manifest.py --output "$FINAL_MANIFEST" \
  --samples-per-class 50 --base-seed "$FINAL_BASE_SEED" \
  --split-name imagenet50k_final --world-size 4 --batch-size 32

python scripts/validate_manifest.py --manifest "$SMOKE_MANIFEST" \
  --expected-count 8 --expected-per-class 1 --expected-num-classes 8 \
  --world-size 1 --batch-size 2 \
  --disjoint-with "$PROXY_MANIFEST" \
  --disjoint-with "$VALIDATION_MANIFEST" \
  --disjoint-with "$FINAL_MANIFEST"
python scripts/validate_manifest.py --manifest "$PROXY_MANIFEST" \
  --expected-count 1000 --expected-per-class 1 --expected-num-classes 1000 \
  --world-size 4 --batch-size 32 \
  --disjoint-with "$SMOKE_MANIFEST" \
  --disjoint-with "$VALIDATION_MANIFEST" \
  --disjoint-with "$FINAL_MANIFEST"
python scripts/validate_manifest.py --manifest "$VALIDATION_MANIFEST" \
  --expected-count 8000 --expected-per-class 8 --expected-num-classes 1000 \
  --world-size 4 --batch-size 32 \
  --disjoint-with "$SMOKE_MANIFEST" \
  --disjoint-with "$PROXY_MANIFEST" \
  --disjoint-with "$FINAL_MANIFEST"
python scripts/validate_manifest.py --manifest "$FINAL_MANIFEST" \
  --expected-count 50000 --expected-per-class 50 --expected-num-classes 1000 \
  --world-size 4 --batch-size 32 \
  --disjoint-with "$SMOKE_MANIFEST" \
  --disjoint-with "$PROXY_MANIFEST" \
  --disjoint-with "$VALIDATION_MANIFEST"
```

Archive each `.meta.json` sidecar and SHA-256. Do not edit a manifest after
generation starts.

## 4. Deferred 2-8 image smoke

Create external force-full and numeric diagnostic SeaCache configs. The large
diagnostic threshold is for proving reuse only and is prohibited as a reported
operating point:

```bash
FORCE_FULL_CONFIG="$OUTPUT_ROOT/configs/jit_force_full_smoke.yaml"
DIAGNOSTIC_CONFIG="$OUTPUT_ROOT/configs/jit_seacache_diagnostic.yaml"
export FORCE_FULL_CONFIG DIAGNOSTIC_CONFIG
python - <<'PY'
import copy
import os
from pathlib import Path

import yaml

base = yaml.safe_load(Path("configs/jit_b16_256_seacache.yaml").read_text(encoding="utf-8"))
for destination, mode, threshold in (
    (os.environ["FORCE_FULL_CONFIG"], "force_full_with_gate", 0.0),
    (os.environ["DIAGNOSTIC_CONFIG"], "seacache", 1_000_000_000.0),
):
    config = copy.deepcopy(base)
    config["model"]["checkpoint"] = os.environ["CHECKPOINT"]
    config["seacache"].update(mode=mode, threshold=threshold)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
```

After `inspect_active_runs.sh` is clean, expose exactly one allocated idle GPU.
The real smoke wrapper runs Full and force-full (`candidate`) on the same
manifest; then the direct command runs numeric diagnostic SeaCache:

```bash
: "${CUDA_VISIBLE_DEVICES:?set exactly one allocated idle GPU}"
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || {
  echo "smoke requires exactly one CUDA_VISIBLE_DEVICES entry" >&2
  return 2 2>/dev/null || exit 2
}
export CUDA_VISIBLE_DEVICES
export SEACACHE_GPU_TESTS_ALLOWED=1

SMOKE_PARITY_ROOT="$OUTPUT_ROOT/smoke/jit_full_vs_force_full"
export SMOKE_PARITY_ROOT
bash scripts/run_deferred_smoke_tests.sh \
  --full-config configs/jit_b16_256_full.yaml \
  --candidate-config "$FORCE_FULL_CONFIG" \
  --manifest "$SMOKE_MANIFEST" \
  --output-root "$SMOKE_PARITY_ROOT"

python scripts/generate_shard.py \
  --config "$DIAGNOSTIC_CONFIG" \
  --manifest "$SMOKE_MANIFEST" \
  --shard-id 0 --world-size 1 \
  --output-root "$OUTPUT_ROOT/smoke/jit_diagnostic_seacache" \
  --acknowledge-gpu-job
```

Verify:

```bash
for RUN_ROOT in \
  "$SMOKE_PARITY_ROOT/full" \
  "$SMOKE_PARITY_ROOT/candidate" \
  "$OUTPUT_ROOT/smoke/jit_diagnostic_seacache"
do
  python scripts/validate_outputs.py \
    --sample-dir "$RUN_ROOT/samples" \
    --manifest "$SMOKE_MANIFEST" \
    --metadata-dir "$RUN_ROOT/metadata" \
    --expected-count 8 \
    --expected-per-class 1 \
    --expected-num-classes 8 \
    --resolution 256
done
```

Required evidence: strict checkpoint keys, EMA1, finite output, Full versus
force-full equality/tolerance, first/final full, expected stream call count,
and empty state after trajectory. Record, do not assume, the tolerance.

## 5. Deferred real reuse diagnostic

Inspect `$OUTPUT_ROOT/smoke/jit_diagnostic_seacache/summaries`: require at
least one reuse, 99 calls per conditional and unconditional stream, and full
first/final calls. Reject the diagnostic if any output is non-finite or cache
state survives trajectory cleanup.

## 6. Deferred compile compatibility

Test Full, force-full, and numeric SeaCache under every jointly supported mode.
`matched_eager` unwraps each block/final compiled callable on the constructed
model instance only. `blockwise` preserves the upstream block/final wrappers
while SeaCache orchestration stays eager. JiT `upstream` also preserves those
wrappers because native JiT is blockwise compiled. Record the instance wrapper
count, correctness, graph breaks, recompiles, compile seconds, steady-state
latency, and memory. Never compare different mode labels in one speedup ratio.

## 7. Deferred 1K proxy

Run Full and candidate SeaCache numeric configs using `PROXY_MANIFEST` and the
four-GPU launcher. Validate outputs, then evaluate paired PSNR/SSIM/LPIPS and
record refresh ratio versus time. This stage finds correctness failures; it
does not select the final threshold.

## 8. Deferred 8K threshold selection

Materialize the coarse candidates declared by
`configs/threshold_sweep_template.yaml` without changing the template:

```bash
: "${VALIDATION_MANIFEST:?build the 8K validation manifest first}"
THRESHOLD_TEMPLATE="$BASE/configs/threshold_sweep_template.yaml"
BASE_SEACACHE_CONFIG="$BASE/configs/jit_b16_256_seacache.yaml"
CANDIDATE_CONFIG_DIR="$OUTPUT_ROOT/configs/jit_threshold_candidates"
export THRESHOLD_TEMPLATE BASE_SEACACHE_CONFIG CANDIDATE_CONFIG_DIR

python - <<'PY'
import copy
import os
from pathlib import Path

import yaml

template = yaml.safe_load(Path(os.environ["THRESHOLD_TEMPLATE"]).read_text(encoding="utf-8"))
base = yaml.safe_load(Path(os.environ["BASE_SEACACHE_CONFIG"]).read_text(encoding="utf-8"))
output = Path(os.environ["CANDIDATE_CONFIG_DIR"])
output.mkdir(parents=True, exist_ok=True)
for value in template["candidate_thresholds"]:
    threshold = float(value)
    config = copy.deepcopy(base)
    config["model"]["checkpoint"] = os.environ["CHECKPOINT"]
    config["seacache"].update(mode="seacache", threshold=threshold)
    tag = format(threshold, ".12g").replace("-", "m").replace(".", "p")
    path = output / f"jit_seacache_threshold_{tag}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(path)
PY
sha256sum "$CANDIDATE_CONFIG_DIR"/*.yaml
```

After parity is proven and four allocated GPUs are idle, generate the fixed
Full 8K reference and each candidate:

```bash
: "${CUDA_VISIBLE_DEVICES:?set exactly four comma-separated allocated idle GPUs}"
IFS=',' read -r -a VALIDATION_GPUS <<<"$CUDA_VISIBLE_DEVICES"
[[ ${#VALIDATION_GPUS[@]} -eq 4 ]] || {
  echo "CUDA_VISIBLE_DEVICES must contain exactly four entries" >&2
  return 2 2>/dev/null || exit 2
}
export CUDA_VISIBLE_DEVICES
export SEACACHE_GPU_TESTS_ALLOWED=1

VALIDATION_FULL_OUTPUT_ROOT="$OUTPUT_ROOT/validation8k/jit_full"
bash scripts/launch_4gpu_50k.sh \
  --config configs/jit_b16_256_full.yaml \
  --manifest "$VALIDATION_MANIFEST" \
  --output-root "$VALIDATION_FULL_OUTPUT_ROOT"
python scripts/validate_outputs.py \
  --sample-dir "$VALIDATION_FULL_OUTPUT_ROOT/samples" \
  --manifest "$VALIDATION_MANIFEST" \
  --metadata-dir "$VALIDATION_FULL_OUTPUT_ROOT/metadata" \
  --expected-count 8000 --expected-per-class 8 \
  --expected-num-classes 1000 --resolution 256

for CANDIDATE_CONFIG in "$CANDIDATE_CONFIG_DIR"/*.yaml; do
  CANDIDATE_NAME="$(basename "$CANDIDATE_CONFIG" .yaml)"
  CANDIDATE_OUTPUT_ROOT="$OUTPUT_ROOT/validation8k/$CANDIDATE_NAME"
  bash scripts/launch_4gpu_50k.sh \
    --config "$CANDIDATE_CONFIG" \
    --manifest "$VALIDATION_MANIFEST" \
    --output-root "$CANDIDATE_OUTPUT_ROOT"
  python scripts/validate_outputs.py \
    --sample-dir "$CANDIDATE_OUTPUT_ROOT/samples" \
    --manifest "$VALIDATION_MANIFEST" \
    --metadata-dir "$CANDIDATE_OUTPUT_ROOT/metadata" \
    --expected-count 8000 --expected-per-class 8 \
    --expected-num-classes 1000 --resolution 256
done
```

Keep `matched_eager`, batch 32/rank, four shards, manifest groups, EMA1, BF16,
Heun 50, CFG 3.0, and interval `[0.1,1.0]` unchanged. Compute unified and
paired quality against the manifest-driven Full 8K output and run matched
latency. Select from the measured quality-latency Pareto frontier, not cache
ratio.

Freeze the chosen numeric config as `$SELECTED_SEACACHE_CONFIG`, record its
SHA-256, and do not modify it during final 50K. Every candidate above is
**validation-only**: never tune, refine, or reselect on final 50K.

## 9. Deferred matched single-GPU benchmark

Generate concrete runner JSONs for batch 1 and the common throughput batch 32.
The selected model config must contain the frozen numeric threshold:

```bash
: "${SELECTED_SEACACHE_CONFIG:?set the frozen numeric config selected on 8K}"
: "${BENCHMARK_BASE_SEED:?set a benchmark-only signed 63-bit base seed}"
JIT_BENCHMARK_BATCH1_JSON="$OUTPUT_ROOT/benchmarks/jit_runner_batch1.json"
JIT_BENCHMARK_BATCH32_JSON="$OUTPUT_ROOT/benchmarks/jit_runner_batch32.json"
export SELECTED_SEACACHE_CONFIG BENCHMARK_BASE_SEED
export JIT_BENCHMARK_BATCH1_JSON JIT_BENCHMARK_BATCH32_JSON

python - <<'PY'
import json
import os
from pathlib import Path

model_config = Path(os.environ["SELECTED_SEACACHE_CONFIG"]).resolve(strict=True)
base_seed = int(os.environ["BENCHMARK_BASE_SEED"])
for variable, batch_size in (
    ("JIT_BENCHMARK_BATCH1_JSON", 1),
    ("JIT_BENCHMARK_BATCH32_JSON", 32),
):
    path = Path(os.environ[variable])
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_config": str(model_config),
        "batch_size": batch_size,
        "sample_ids": list(range(batch_size)),
        "seeds": [base_seed + index for index in range(batch_size)],
        "class_ids": [index % 1000 for index in range(batch_size)],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

: "${CUDA_VISIBLE_DEVICES:?set exactly one allocated idle GPU for benchmarking}"
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || {
  echo "benchmark requires exactly one CUDA_VISIBLE_DEVICES entry" >&2
  return 2 2>/dev/null || exit 2
}
export CUDA_VISIBLE_DEVICES
export SEACACHE_GPU_TESTS_ALLOWED=1

bash scripts/benchmark_single_gpu.sh \
  --runner-factory seacache_style.jit_benchmark:build_benchmark_spec \
  --runner-config "$JIT_BENCHMARK_BATCH1_JSON" \
  --output-json "$OUTPUT_ROOT/benchmarks/jit_batch1.json" \
  --warmup-batches 10 --measured-batches 30
bash scripts/benchmark_single_gpu.sh \
  --runner-factory seacache_style.jit_benchmark:build_benchmark_spec \
  --runner-config "$JIT_BENCHMARK_BATCH32_JSON" \
  --output-json "$OUTPUT_ROOT/benchmarks/jit_batch32.json" \
  --warmup-batches 10 --measured-batches 30
```

Do not proceed until this factory has passed deferred GPU parity. Report
both batch sizes. Each invocation measures Full then SeaCache in one process
with identical inputs. Compile time is separate. The primary speedup is median
Full ms/image divided by median SeaCache ms/image.

The enclosing latency uses CUDA Events and one end-boundary synchronization.
Gate wall time uses the host clock and includes required scalar sync; FFT and
residual `cache_io` use asynchronous CUDA Event pairs resolved after that
boundary, without per-component barriers. Exclude checkpoint load, first
compile, dataloader, CPU copy, PNG/file I/O and metrics.

## 10. Deferred four-GPU final generation

```bash
: "${SELECTED_SEACACHE_CONFIG:?set the frozen, hashed 8K-selected config}"
: "${CUDA_VISIBLE_DEVICES:?set exactly four comma-separated allocated idle GPUs}"
IFS=',' read -r -a FINAL_GPUS <<<"$CUDA_VISIBLE_DEVICES"
[[ ${#FINAL_GPUS[@]} -eq 4 ]] || {
  echo "CUDA_VISIBLE_DEVICES must contain exactly four entries" >&2
  return 2 2>/dev/null || exit 2
}
FULL_OUTPUT_ROOT="$OUTPUT_ROOT/final50k/full"
SEACACHE_OUTPUT_ROOT="$OUTPUT_ROOT/final50k/seacache"
export CUDA_VISIBLE_DEVICES FULL_OUTPUT_ROOT SEACACHE_OUTPUT_ROOT
export SEACACHE_GPU_TESTS_ALLOWED=1

bash scripts/launch_4gpu_50k.sh \
  --config configs/jit_b16_256_full.yaml \
  --manifest "$FINAL_MANIFEST" \
  --output-root "$FULL_OUTPUT_ROOT"

bash scripts/launch_4gpu_50k.sh \
  --config "$SELECTED_SEACACHE_CONFIG" \
  --manifest "$FINAL_MANIFEST" \
  --output-root "$SEACACHE_OUTPUT_ROOT"
```

Resume, if necessary, only with identical inputs:

```bash
: "${CUDA_VISIBLE_DEVICES:?set the same four allocated GPUs}"
IFS=',' read -r -a RESUME_GPUS <<<"$CUDA_VISIBLE_DEVICES"
[[ ${#RESUME_GPUS[@]} -eq 4 ]] || {
  echo "CUDA_VISIBLE_DEVICES must contain exactly four entries" >&2
  return 2 2>/dev/null || exit 2
}
: "${SELECTED_SEACACHE_CONFIG:?set the unchanged selected config}"
: "${FINAL_MANIFEST:?set the unchanged final manifest}"
: "${SEACACHE_OUTPUT_ROOT:?set the existing SeaCache output root}"
export CUDA_VISIBLE_DEVICES
export SEACACHE_GPU_TESTS_ALLOWED=1
bash scripts/launch_4gpu_50k.sh \
  --config "$SELECTED_SEACACHE_CONFIG" \
  --manifest "$FINAL_MANIFEST" \
  --output-root "$SEACACHE_OUTPUT_ROOT" --resume
```

Never resume with a changed threshold, batch size/grouping, checkpoint,
manifest content, `noise_scale`, compile mode, Git/SeaCache revision, PyTorch
version, or executable port source/config/dependency bytes. The run manifest
binds both raw/canonical manifest hashes and `port_source_sha256` (including
untracked executable files).

Every invocation preserves `four_gpu_wall_clock_<start_ns>.json` and refreshes
`four_gpu_wall_clock.json`. `images_per_second` covers only samples newly
generated during that invocation divided by that invocation's elapsed time;
the last resume file is not cumulative 50K throughput. Archive all timestamped
files. Output-root and per-physical-GPU coordination locks reject overlapping
local launchers. On INT/TERM/HUP the launcher sends no signal to ranks; it
keeps locks, waits for owned ranks to exit, and writes an interrupted
wall-clock record before exit 130 if any rank had started. After an uncatchable crash,
remove a stale `/tmp/pixarc-seacache-gpu-locks/*.lock` only after checking that
no corresponding rank is active. Ranks read the byte-checked archived config
and manifest, not the mutable source files; the source config directory is used
only for relative checkpoint resolution. Wall-clock baselines are counted from
durable metadata and current summaries must match the invocation ID. Any rank
may atomically create an identity-compatible run manifest without replacing a winner.

## 11. Validate final outputs

```bash
python scripts/validate_outputs.py --sample-dir "$FULL_OUTPUT_ROOT/samples" \
  --manifest "$FINAL_MANIFEST" --metadata-dir "$FULL_OUTPUT_ROOT/metadata" \
  --expected-count 50000 --expected-per-class 50 \
  --expected-num-classes 1000 --resolution 256
python scripts/validate_outputs.py --sample-dir "$SEACACHE_OUTPUT_ROOT/samples" \
  --manifest "$FINAL_MANIFEST" --metadata-dir "$SEACACHE_OUTPUT_ROOT/metadata" \
  --expected-count 50000 --expected-per-class 50 \
  --expected-num-classes 1000 --resolution 256
```

Confirm four rank metadata files and summaries exist and exactly cover the
manifest before metrics.

## 12. Unified distribution metrics

Run once for Full and once for SeaCache:

```bash
python -m seacache_style.distribution_metrics \
  --sample-dir "$FULL_OUTPUT_ROOT/samples" \
  --manifest "$FINAL_MANIFEST" \
  --reference-npz "$IMAGENET_REFERENCE_NPZ" \
  --evaluator "$ADM_EVALUATOR" \
  --output-json "$FULL_OUTPUT_ROOT/distribution_metrics.json" \
  --sample-npz "$FULL_OUTPUT_ROOT/samples_adm.npz" \
  --expected-count 50000 --expected-per-class 50 \
  --expected-num-classes 1000 --resolution 256

python -m seacache_style.distribution_metrics \
  --sample-dir "$SEACACHE_OUTPUT_ROOT/samples" \
  --manifest "$FINAL_MANIFEST" \
  --reference-npz "$IMAGENET_REFERENCE_NPZ" \
  --evaluator "$ADM_EVALUATOR" \
  --output-json "$SEACACHE_OUTPUT_ROOT/distribution_metrics.json" \
  --sample-npz "$SEACACHE_OUTPUT_ROOT/samples_adm.npz" \
  --expected-count 50000 --expected-per-class 50 \
  --expected-num-classes 1000 --resolution 256
```

Compute and report SeaCache-minus-Full deltas for FID, sFID, IS, precision,
and recall. Preserve raw evaluator output and evaluator revision.

The wrapper automatically reads `run_manifest.json` from each `samples/`
parent (or an explicit `--run-metadata`), verifies raw/canonical manifest and
complete run identity, archived input config/manifest, all rank metadata, and
sample-tree ownership. It records and validates ordered Full/SeaCache roles and
hashes both the reference NPZ and evaluator before use, checking them again
afterward. Relocated or legacy samples without that metadata fail closed.

```bash
python scripts/compare_distribution.py \
  --full-json "$FULL_OUTPUT_ROOT/distribution_metrics.json" \
  --seacache-json "$SEACACHE_OUTPUT_ROOT/distribution_metrics.json" \
  --output-json "$SEACACHE_OUTPUT_ROOT/distribution_deltas.json"
```

## 13. Strict paired metrics

```bash
python -m seacache_style.paired_metrics \
  --reference-dir "$FULL_OUTPUT_ROOT/samples" \
  --candidate-dir "$SEACACHE_OUTPUT_ROOT/samples" \
  --reference-manifest "$FINAL_MANIFEST" \
  --candidate-manifest "$FINAL_MANIFEST" \
  --reference-run-metadata "$FULL_OUTPUT_ROOT/run_manifest.json" \
  --candidate-run-metadata "$SEACACHE_OUTPUT_ROOT/run_manifest.json" \
  --output-json "$SEACACHE_OUTPUT_ROOT/paired_metrics.json" \
  --output-csv "$SEACACHE_OUTPUT_ROOT/paired_metrics.csv" \
  --resolution 256 --lpips-device cuda --lpips-batch-size 16
```

Any mismatch in IDs, class, seed, model/config hash, checkpoint/EMA,
sampler/config hash, CFG, `noise_scale`, dtype, raw/canonical manifest identity,
port source hash, revisions, PyTorch version, resolution, or postprocessing is
fatal. The command also verifies each PNG tree against its archived inputs and
four rank metadata files and rejects reversed or non-Full/SeaCache roles. Do not pair the legacy
scheduled JiT reference unless the conditional replay proof in
`BASELINE_COMPATIBILITY_REPORT.md` has passed.

## 14. Archive

Archive configs and hashes, manifests and sidecars, run manifests, rank
metadata/summaries/logs, every `four_gpu_wall_clock_<start_ns>.json`, validation
output, metric JSON/CSV, benchmark JSON, environment versions, and deferred-test
evidence. Do not commit generated PNG, NPZ, checkpoint, or large logs.
