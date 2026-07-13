# Unofficial SeaCache-style port for JiT

This directory provides a local, research-only SeaCache-style inference port
for JiT class-to-image generation. It is not an official SeaCache integration.

> Current safety status (audit snapshot, 2026-07-13 UTC): PixelGen reference
> generation was **ACTIVE** and JiT Full 50K was **SCHEDULED**. No GPU command
> in this directory was executed during implementation. All generation,
> benchmark, FID, and LPIPS commands below are deferred until the current jobs
> finish and all target GPUs are confirmed idle.

## Method and faithful semantics

The local port follows the audited SeaCache FLUX implementation at commit
`8dcf49097fcd37e39774fe7409cb3b9e0fdb4fe2`:

- probe FFT runs in float32 over patch-grid H/W and restores input dtype;
- the separable full-FFT Wiener gain uses mean normalization;
- JiT's noising convention gives `a=t`, `b=1-t`;
- relative L1 and gate decisions reduce over the complete batch;
- accumulated distance uses strict `< threshold` for reuse;
- first/final calls are full;
- official raw/filtered previous-probe write order is preserved;
- reuse adds the previous whole-body residual to current body input.

The cache boundary is:

```text
fresh patch embedding + position embedding
    -> cacheable all Transformer blocks
       (class context inserted at block 4 and removed after the body)
    -> fresh final RMSNorm/AdaLN/projection/unpatchify
```

The probe is the exact first attention-branch input:

```python
modulate(blocks[0].norm1(body_input), shift_msa, scale_msa)
```

JiT preserves upstream CFG as two forwards in this order:

```text
cond -> uncond -> CFG combination
```

The two branches use completely separate cache states. For 50-step Heun, each
stream executes `2*(50-1)+1=99` model calls, or 198 JiT forwards total.

## Modes

- `full`: directly calls upstream `JiT.forward`; no probe, FFT, gate, or cache.
- `force_full_with_gate`: executes the gate/probe path but always refreshes;
  use only for correctness parity, not the Full latency denominator.
- `seacache`: requires an explicit numeric threshold selected on the separate
  8K validation manifest.

No best threshold is supplied. The threshold in the SeaCache YAML is
deliberately null and must fail until replaced in a frozen selected config.

## Environment

From the PixARC repository:

```bash
ROOT="$(git rev-parse --show-toplevel)"
BASE="$ROOT/JiT/baselines/seacache-style"
UPSTREAM_JIT="$ROOT/third-party/JiT"
CHECKPOINT="$ROOT/JiT/checkpoints/JiT-B-16-256/checkpoint-last.pth"
: "${JIT_PYTHON:?set the executable from the JiT Python environment}"
: "${OUTPUT_ROOT:?set an output directory outside the PixARC source tree}"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
[[ "$OUTPUT_ROOT/" != "$ROOT/"* ]] || {
  echo "OUTPUT_ROOT must be outside $ROOT" >&2
  return 2 2>/dev/null || exit 2
}

test -x "$JIT_PYTHON"
export ROOT BASE UPSTREAM_JIT CHECKPOINT OUTPUT_ROOT JIT_PYTHON
export PATH="$(dirname "$JIT_PYTHON"):$PATH"
export PYTHONPATH="$UPSTREAM_JIT:$BASE:${PYTHONPATH:-}"
cd "$BASE"
test -s "$CHECKPOINT"
mkdir -p "$OUTPUT_ROOT"
```

The upstream `VisionRotaryEmbeddingFast` constructs CUDA tensors during model
instantiation. Import and mock tests are CPU-safe; a real JiT model is not.

Extra evaluation dependencies are listed in `requirements-extra.txt`. This
task did not install or upgrade packages. LPIPS is optional until paired LPIPS
evaluation. The ADM evaluator and ImageNet reference NPZ must already exist
locally; nothing is downloaded automatically.

## CPU verification

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/pixarc-jit-seacache-pycache \
  "$JIT_PYTHON" -m compileall "$BASE"
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  "$JIT_PYTHON" -m pytest -q -p no:cacheprovider "$BASE/tests"
git -C "$ROOT" diff --check -- JiT/baselines/seacache-style
```

`git diff --check` sees only tracked changes; it does not inspect untracked
files. Check `git status --short`, and after intentionally staging new files
also run `git -C "$ROOT" diff --cached --check`.

The final implementation-time run reported `37 passed, 2 subtests passed`, with no
failure or skip. SEA fixtures matched the audited local function exactly with
`rtol=0, atol=0`. This is not a GPU model-parity result.

## Deterministic manifests

Choose disjoint signed 63-bit seed ranges; do not reuse validation seeds for
final generation.

```bash
: "${VALIDATION_BASE_SEED:?set a validation base seed}"
: "${FINAL_BASE_SEED:?set a disjoint final base seed}"

VALIDATION_MANIFEST="$OUTPUT_ROOT/manifests/imagenet8k.jsonl"
FINAL_MANIFEST="$OUTPUT_ROOT/manifests/imagenet50k.jsonl"
export VALIDATION_MANIFEST FINAL_MANIFEST

python scripts/build_manifest.py \
  --output "$VALIDATION_MANIFEST" \
  --samples-per-class 8 \
  --base-seed "$VALIDATION_BASE_SEED" \
  --split-name imagenet8k_validation \
  --world-size 4 \
  --batch-size 32

python scripts/build_manifest.py \
  --output "$FINAL_MANIFEST" \
  --samples-per-class 50 \
  --base-seed "$FINAL_BASE_SEED" \
  --split-name imagenet50k_final \
  --world-size 4 \
  --batch-size 32

python scripts/validate_manifest.py \
  --manifest "$VALIDATION_MANIFEST" \
  --expected-count 8000 \
  --expected-per-class 8 \
  --expected-num-classes 1000 \
  --world-size 4 \
  --batch-size 32 \
  --disjoint-with "$FINAL_MANIFEST"

python scripts/validate_manifest.py \
  --manifest "$FINAL_MANIFEST" \
  --expected-count 50000 \
  --expected-per-class 50 \
  --expected-num-classes 1000 \
  --world-size 4 \
  --batch-size 32 \
  --disjoint-with "$VALIDATION_MANIFEST"
```

The sidecar records manifest SHA-256, PyTorch version, generator protocol,
class rule, seed base, sharding rule, world size, and batch size. Initial noise
is generated independently from each record seed, so rank order, resume, and
earlier samples do not change it.

## Deferred GPU smoke tests

Do not run these while any reference task is active. First inspect read-only:

```bash
bash scripts/inspect_active_runs.sh
```

The launcher and shard runner remain locked unless the operator explicitly
sets `SEACACHE_GPU_TESTS_ALLOWED=1`; neither tool kills existing processes.

Create an eight-sample, one-rank smoke manifest under `OUTPUT_ROOT`. Its seed
range must also be disjoint from validation and final generation:

```bash
: "${SMOKE_BASE_SEED:?set a smoke-only base seed}"
SMOKE_MANIFEST="$OUTPUT_ROOT/manifests/smoke8.jsonl"
export SMOKE_BASE_SEED SMOKE_MANIFEST
python -c 'import os; from seacache_style.manifest import build_manifest, write_manifest; seed=int(os.environ["SMOKE_BASE_SEED"]); rows=build_manifest(samples_per_class=1, base_seed=seed, split_name="smoke8", world_size=1, batch_size=2, num_classes=8); write_manifest(rows, os.environ["SMOKE_MANIFEST"], base_seed=seed, world_size=1, batch_size=2)'

python scripts/validate_manifest.py \
  --manifest "$SMOKE_MANIFEST" \
  --expected-count 8 \
  --expected-per-class 1 \
  --expected-num-classes 8 \
  --world-size 1 \
  --batch-size 2 \
  --disjoint-with "$VALIDATION_MANIFEST" \
  --disjoint-with "$FINAL_MANIFEST"
```

Create force-full and deliberately high-threshold diagnostic configs outside
the source tree. The diagnostic threshold is numeric only to prove real reuse;
it is not a candidate operating point.

```bash
FORCE_FULL_CONFIG="$OUTPUT_ROOT/configs/jit_force_full_smoke.yaml"
DIAGNOSTIC_CONFIG="$OUTPUT_ROOT/configs/jit_seacache_diagnostic.yaml"
export FORCE_FULL_CONFIG DIAGNOSTIC_CONFIG
python - <<'PY'
import copy
import os
from pathlib import Path

import yaml

source = Path("configs/jit_b16_256_seacache.yaml")
base = yaml.safe_load(source.read_text(encoding="utf-8"))
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

After the read-only inspection is clean, explicitly expose exactly one
allocated idle GPU. The shipped wrapper is a real two-run wrapper: here its
`candidate` output is the force-full run.

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

Validate all three outputs before inspecting parity and reuse summaries:

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

Required smoke evidence is strict checkpoint load, EMA1, Full versus
force-full parity, at least one diagnostic reuse while first/final calls stay
full, finite output, 99 calls per CFG stream for 50-step Heun, and empty state
after cleanup. These GPU checks remain **not executed** in this snapshot.

## Compile-mode semantics

- `matched_eager` binds each block and final layer's
  `_torchdynamo_orig_callable` on the constructed JiT instance only; it does
  not mutate upstream classes.
- `blockwise` preserves the upstream block/final `torch.compile` wrappers and
  keeps SeaCache orchestration eager.
- `upstream` also preserves those wrappers, matching JiT's native blockwise
  behavior; compare it only against SeaCache using the same label.

All modes are GPU-unverified. Never mix compile modes in a speedup ratio.

## Threshold selection

Use only the 8K validation manifest. The following command reads the coarse
candidates from `configs/threshold_sweep_template.yaml` and makes frozen model
configs under `OUTPUT_ROOT`; it does not edit the shipped template:

```bash
: "${CHECKPOINT:?set the JiT checkpoint}"
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

Only after smoke/parity succeeds and four allocated GPUs are idle, generate the
Full 8K reference once and each candidate on the same fixed manifest. The
launcher requires four explicit `CUDA_VISIBLE_DEVICES` entries and maps one
entry to each logical rank:

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
export VALIDATION_FULL_OUTPUT_ROOT
bash scripts/launch_4gpu_50k.sh \
  --config configs/jit_b16_256_full.yaml \
  --manifest "$VALIDATION_MANIFEST" \
  --output-root "$VALIDATION_FULL_OUTPUT_ROOT"
python scripts/validate_outputs.py \
  --sample-dir "$VALIDATION_FULL_OUTPUT_ROOT/samples" \
  --manifest "$VALIDATION_MANIFEST" \
  --metadata-dir "$VALIDATION_FULL_OUTPUT_ROOT/metadata" \
  --expected-count 8000 \
  --expected-per-class 8 \
  --expected-num-classes 1000 \
  --resolution 256

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
    --expected-count 8000 \
    --expected-per-class 8 \
    --expected-num-classes 1000 \
    --resolution 256
done
```

Keep checkpoint, manifest, batch grouping, dtype, CFG, sampler, steps, and
compile mode fixed. Measure quality and matched latency, then freeze selected
numeric configs with hashes. These generated files are **validation-only**;
cache ratio alone is not a selection objective, and final 50K data must never
be used to tune or refine the threshold.

## Matched latency and speedup

The latency engine uses the included JiT factory. Generate real runner JSONs
for batch 1 and the common throughput batch 32 from the frozen selected config:

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
  --output-json "$OUTPUT_ROOT/benchmarks/jit_latency_batch1.json" \
  --warmup-batches 10 \
  --measured-batches 30

bash scripts/benchmark_single_gpu.sh \
  --runner-factory seacache_style.jit_benchmark:build_benchmark_spec \
  --runner-config "$JIT_BENCHMARK_BATCH32_JSON" \
  --output-json "$OUTPUT_ROOT/benchmarks/jit_latency_batch32.json" \
  --warmup-batches 10 \
  --measured-batches 30
```

Each factory invocation measures matched Full then SeaCache on the same model,
inputs, and visible GPU. The factory exists but its real checkpoint and compile
behavior still require deferred GPU validation. Full and SeaCache must share
GPU, checkpoint/EMA, batch, noise, dtype, sampler, CFG, and compile mode. Main
speedup is median Full ms/image divided by median SeaCache ms/image. Four-GPU
50K wall time is production throughput, not this algorithmic speedup.

The enclosing latency uses CUDA Events and one end-boundary synchronization.
Gate wall time uses the host clock and includes the required scalar sync; FFT
and residual `cache_io` use asynchronous CUDA Event pairs resolved only after
that enclosing sync, so no per-component barrier is added. Checkpoint load,
first compile, dataloader, CPU copy, PNG/file I/O, and metrics are excluded.

## Deferred four-GPU 50K

After smoke, parity, 1K proxy, 8K sweep, and single-GPU benchmark pass:

```bash
: "${SELECTED_SEACACHE_CONFIG:?set the frozen, hashed numeric config selected on 8K}"
: "${CUDA_VISIBLE_DEVICES:?set exactly four comma-separated allocated idle GPUs}"
IFS=',' read -r -a FINAL_GPUS <<<"$CUDA_VISIBLE_DEVICES"
[[ ${#FINAL_GPUS[@]} -eq 4 ]] || {
  echo "CUDA_VISIBLE_DEVICES must contain exactly four entries" >&2
  return 2 2>/dev/null || exit 2
}
FULL_OUTPUT_ROOT="$OUTPUT_ROOT/final50k/jit_full"
SEACACHE_OUTPUT_ROOT="$OUTPUT_ROOT/final50k/jit_seacache"
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

Each process handles `sample_id % 4`, writes its own metadata/log/summary, and
uses fixed batch groups. Output roots must be outside Git and non-empty roots
are rejected unless `--resume` is explicit. Before launch, the config and
manifest are copied and byte-compared into `config_resolved.yaml` and
`input_manifest.jsonl`; every rank reads those archived bytes. The original
config directory is retained only to resolve a relative checkpoint path. All
ranks compete through a no-replace atomic create for a same-identity run manifest, so
its creation is not a rank-0 single point of failure.

Each launcher invocation writes `four_gpu_wall_clock.json` plus an immutable
`four_gpu_wall_clock_<start_ns>.json`. Its throughput is
`generated_this_invocation / elapsed_this_invocation`; on resume it is not a
cumulative 50K throughput. Only an uninterrupted first invocation measures the
whole production run directly. Resume baselines come from durable rank
metadata, and per-rank summaries carry the launcher invocation ID, so a stale
or missing pre-crash summary cannot be counted as current work. Output-root and
per-physical-GPU coordination
locks prevent overlapping local launchers. On INT/TERM/HUP the launcher sends
no signal to ranks; it keeps the locks and waits for owned ranks to exit. If
any rank had started, it writes an interrupted wall-clock record before exit 130.

## Resume and leak checks

Resume only with the exact same raw and canonical manifest identities, input
config hash, checkpoint identity, threshold, `noise_scale`, resolution, batch
groups, Git/SeaCache revisions, PyTorch version, and
`port_source_sha256`. The source hash covers executable Python/shell code,
configs, and `requirements-extra.txt`, including currently untracked files:

```bash
: "${SELECTED_SEACACHE_CONFIG:?set the unchanged selected config}"
: "${FINAL_MANIFEST:?set the unchanged final manifest}"
: "${SEACACHE_OUTPUT_ROOT:?set the existing SeaCache output root}"
: "${CUDA_VISIBLE_DEVICES:?set the same four allocated GPUs}"
IFS=',' read -r -a RESUME_GPUS <<<"$CUDA_VISIBLE_DEVICES"
[[ ${#RESUME_GPUS[@]} -eq 4 ]] || {
  echo "CUDA_VISIBLE_DEVICES must contain exactly four entries" >&2
  return 2 2>/dev/null || exit 2
}
export CUDA_VISIBLE_DEVICES
export SEACACHE_GPU_TESTS_ALLOWED=1
bash scripts/launch_4gpu_50k.sh \
  --config "$SELECTED_SEACACHE_CONFIG" \
  --manifest "$FINAL_MANIFEST" \
  --output-root "$SEACACHE_OUTPUT_ROOT" \
  --resume
```

A partially present group is an error; completed matching groups may be
skipped. Each group is a new trajectory. `generate()` ends and resets both
streams in `finally`; summaries must show expected call counts and no state may
remain active between batches. A changed environment or executable port file
requires a new output root rather than resume.

## Output validation

```bash
python scripts/validate_outputs.py \
  --sample-dir "$FULL_OUTPUT_ROOT/samples" \
  --manifest "$FINAL_MANIFEST" \
  --metadata-dir "$FULL_OUTPUT_ROOT/metadata" \
  --expected-count 50000 \
  --expected-per-class 50 \
  --expected-num-classes 1000 \
  --resolution 256

python scripts/validate_outputs.py \
  --sample-dir "$SEACACHE_OUTPUT_ROOT/samples" \
  --manifest "$FINAL_MANIFEST" \
  --metadata-dir "$SEACACHE_OUTPUT_ROOT/metadata" \
  --expected-count 50000 \
  --expected-per-class 50 \
  --expected-num-classes 1000 \
  --resolution 256
```

This checks numeric IDs, coverage, class balance, RGB, 256x256, uint8, and
decodability before metrics.

## Unified distribution metrics

Both methods must use the same explicit local ADM evaluator and ImageNet-256
reference NPZ:

```bash
: "${IMAGENET_REFERENCE_NPZ:?set the local ImageNet-256 ADM reference NPZ}"
: "${ADM_EVALUATOR:?set the local ADM evaluator.py}"
test -s "$IMAGENET_REFERENCE_NPZ"
test -f "$ADM_EVALUATOR"
python -m seacache_style.distribution_metrics \
  --sample-dir "$FULL_OUTPUT_ROOT/samples" \
  --manifest "$FINAL_MANIFEST" \
  --reference-npz "$IMAGENET_REFERENCE_NPZ" \
  --evaluator "$ADM_EVALUATOR" \
  --output-json "$FULL_OUTPUT_ROOT/distribution_metrics.json" \
  --sample-npz "$FULL_OUTPUT_ROOT/samples_adm.npz" \
  --expected-count 50000 \
  --expected-per-class 50 \
  --expected-num-classes 1000 \
  --resolution 256

python -m seacache_style.distribution_metrics \
  --sample-dir "$SEACACHE_OUTPUT_ROOT/samples" \
  --manifest "$FINAL_MANIFEST" \
  --reference-npz "$IMAGENET_REFERENCE_NPZ" \
  --evaluator "$ADM_EVALUATOR" \
  --output-json "$SEACACHE_OUTPUT_ROOT/distribution_metrics.json" \
  --sample-npz "$SEACACHE_OUTPUT_ROOT/samples_adm.npz" \
  --expected-count 50000 \
  --expected-per-class 50 \
  --expected-num-classes 1000 \
  --resolution 256
```

Report FID, sFID, IS, precision, recall, and SeaCache-minus-Full deltas. Metric
time is not generation latency.

The wrapper automatically reads `run_manifest.json` from each `samples/`
parent (or an explicit `--run-metadata`), verifies raw/canonical manifest and
complete run identity, requires the owned `samples/` tree, archived config and
manifest, and all four rank metadata files, and binds their config/checkpoint/
threshold fields before evaluation. It records the ordered Full/SeaCache roles
and hashes both the reference NPZ and evaluator before execution, rejecting a
concurrent change afterward. Relocated or legacy samples without that evidence
fail closed.

```bash
python scripts/compare_distribution.py \
  --full-json "$FULL_OUTPUT_ROOT/distribution_metrics.json" \
  --seacache-json "$SEACACHE_OUTPUT_ROOT/distribution_metrics.json" \
  --output-json "$SEACACHE_OUTPUT_ROOT/distribution_deltas.json"
```

## Strict paired metrics

Only run after manifest and run-metadata validation proves identical sample ID,
class, seed, noise protocol/scale, model and sampler config hashes, steps, CFG,
dtype, checkpoint/EMA, source/revision identity, resolution, and
postprocessing. The paired entry also requires each directory's archived inputs
and complete per-rank metadata, and enforces ordered `full -> seacache` roles:

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
  --resolution 256 \
  --lpips-device cuda \
  --lpips-batch-size 16
```

The scheduled legacy JiT reference is only conditionally compatible; see
`BASELINE_COMPATIBILITY_REPORT.md`. Do not pair by filename alone.

## Current reference compatibility and limitations

- Compatibility: `PAIRED_METRICS_CONDITIONALLY_COMPATIBLE`.
- The scheduled reference uses continuous per-rank RNG streams and saves no
  per-image seed/noise hash.
- Batch-global decisions change with batch size and grouping.
- Threshold is intentionally unselected.
- Real JiT, checkpoint/EMA, compile, latency, and generation tests are deferred.
- SeaCache's audited clone has no license file; see `NOTICE.md`.

Detailed audit, ordered operations, and implementation answers are in
`AUDIT.md`, `RUNBOOK.md`, and `IMPLEMENTATION_REPORT.md`.
