# Unofficial SeaCache-style port for PixelGen

This directory adds a manifest-driven SeaCache-style class-to-image inference
path without modifying `third-party/PixelGen`. At the audit snapshot PixelGen
Full 50K was **ACTIVE** and JiT was **SCHEDULED**. No GPU command below was run.

## Faithful integration

- SEA uses float32 full FFT over patch-grid H/W, mean normalization and restores
  dtype; JiT time maps to `a=t,b=1-t`.
- Gate distance is batch-global. With PixelGen batch 4, CFG forms one effective
  2B=8 model batch and one `combined_cfg` cache state.
- Cache covers all Transformer blocks between patch+position image tokens and
  post-context-removal image tokens. Final modulation/projection/unpatchify are
  always fresh.
- `exact_henu=true`, 50 steps gives 99 combined forwards.
- Guidance is 2.25 for shifted time `t>0.1 and t<=0.9`; timeshift is 2.0.
- `return_layer` or `return_last` forces full body and preserves upstream tuple
  results.
- Runtime state is absent from checkpoints; deepcopy produces independent empty
  denoiser/EMA state.

The probe is exactly the first block attention-branch input, after patch
embedding and the upstream in-place positional add:

```python
condition = self.t_embedder(t) + self.y_embedder(y)
body_input = self.x_embedder(x)
body_input += self.pos_embed
shift_msa, scale_msa, *_ = self.blocks[0].adaLN_modulation(condition).chunk(6, dim=-1)
probe_raw = modulate(self.blocks[0].norm1(body_input), shift_msa, scale_msa)
```

Modes are `full`, `force_full_with_gate`, and `seacache`. Full directly uses the
upstream-equivalent forward. SeaCache requires a numeric threshold selected on
the independent 8K manifest; the shipped value is deliberately null.

## Setup and CPU tests

```bash
ROOT="$(git rev-parse --show-toplevel)"
BASE="$ROOT/PixelGen/baselines/seacache-style"
UPSTREAM="$ROOT/third-party/PixelGen"
CHECKPOINT="$ROOT/PixelGen/checkpoints/PixelGen_XL_160ep.ckpt"
: "${PIXELGEN_PYTHON:?set the executable from the PixelGen Python environment}"
: "${OUTPUT_ROOT:?set an output directory outside the PixARC source tree}"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
[[ "$OUTPUT_ROOT/" != "$ROOT/"* ]] || {
  echo "OUTPUT_ROOT must be outside $ROOT" >&2
  return 2 2>/dev/null || exit 2
}
test -x "$PIXELGEN_PYTHON"
export ROOT BASE UPSTREAM CHECKPOINT OUTPUT_ROOT PIXELGEN_PYTHON
export PATH="$(dirname "$PIXELGEN_PYTHON"):$PATH"
export PYTHONPATH="$UPSTREAM:$BASE:${PYTHONPATH:-}"
cd "$BASE"
test -s "$CHECKPOINT"
mkdir -p "$OUTPUT_ROOT"

CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/pixarc-pixelgen-seacache-pycache \
  "$PIXELGEN_PYTHON" -m compileall "$BASE"
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/matplotlib-seacache-test \
  "$PIXELGEN_PYTHON" -m unittest discover -s "$BASE/tests" -v
git -C "$ROOT" diff --check -- PixelGen/baselines/seacache-style
```

`git diff --check` does not inspect untracked files. Check
`git status --short`; after intentionally staging new files, also run
`git -C "$ROOT" diff --cached --check`.

The final CPU run in the local PixelGen environment passed 47 tests with zero
failures or skips. It did not instantiate a real CUDA JiT.

Real model/checkpoint/EMA and compile parity remain GPU-deferred because
upstream model construction uses CUDA.

Both shipped inference configs use
`seacache_style.pixelgen_lightning.InferenceOnlyTrainer`. Prediction never
calls the diffusion trainer, so this parameter-free placeholder avoids
training-only LPIPS/DINO imports and `torch.hub` access; it raises if used for
training.

## Immutable manifests

Choose mutually disjoint smoke, validation, and final seed ranges. Every
manifest is under `OUTPUT_ROOT`, outside the source tree:

```bash
: "${SMOKE_BASE_SEED:?set an 8-sample smoke-only base seed}"
: "${VALIDATION_BASE_SEED:?set 8K base seed}"
: "${FINAL_BASE_SEED:?set disjoint 50K base seed}"
SMOKE_MANIFEST="$OUTPUT_ROOT/manifests/pixelgen_smoke8.jsonl"
VALIDATION_MANIFEST="$OUTPUT_ROOT/manifests/pixelgen8k.jsonl"
FINAL_MANIFEST="$OUTPUT_ROOT/manifests/pixelgen50k.jsonl"
export SMOKE_BASE_SEED SMOKE_MANIFEST VALIDATION_MANIFEST FINAL_MANIFEST

python -c 'import os; from seacache_style.manifest import build_manifest, write_manifest; seed=int(os.environ["SMOKE_BASE_SEED"]); rows=build_manifest(samples_per_class=1, base_seed=seed, split_name="pixelgen_smoke8", world_size=1, batch_size=4, num_classes=8); write_manifest(rows, os.environ["SMOKE_MANIFEST"], base_seed=seed, world_size=1, batch_size=4)'

python scripts/build_manifest.py --output "$VALIDATION_MANIFEST" \
  --samples-per-class 8 --base-seed "$VALIDATION_BASE_SEED" \
  --split-name pixelgen8k_validation --world-size 4 --batch-size 4
python scripts/build_manifest.py --output "$FINAL_MANIFEST" \
  --samples-per-class 50 --base-seed "$FINAL_BASE_SEED" \
  --split-name pixelgen50k_final --world-size 4 --batch-size 4
python scripts/validate_manifest.py --manifest "$VALIDATION_MANIFEST" \
  --expected-count 8000 --expected-per-class 8 --expected-num-classes 1000 \
  --world-size 4 --batch-size 4 \
  --disjoint-with "$SMOKE_MANIFEST" --disjoint-with "$FINAL_MANIFEST"
python scripts/validate_manifest.py --manifest "$FINAL_MANIFEST" \
  --expected-count 50000 --expected-per-class 50 --expected-num-classes 1000 \
  --world-size 4 --batch-size 4 \
  --disjoint-with "$SMOKE_MANIFEST" --disjoint-with "$VALIDATION_MANIFEST"
python scripts/validate_manifest.py --manifest "$SMOKE_MANIFEST" \
  --expected-count 8 --expected-per-class 1 --expected-num-classes 8 \
  --world-size 1 --batch-size 4 \
  --disjoint-with "$VALIDATION_MANIFEST" --disjoint-with "$FINAL_MANIFEST"
```

Per-sample CPU generators make noise independent of rank launch, resume and
earlier RNG use. Fixed `batch_group_id` preserves the batch-global gate.

## Deferred smoke

First run `bash scripts/inspect_active_runs.sh`. Proceed only after all target
GPUs are idle; launchers require `SEACACHE_GPU_TESTS_ALLOWED=1` and never kill
jobs.

Create copyable force-full and deliberately high-threshold diagnostic configs
outside the source tree. The diagnostic numeric threshold exists only to prove
that reuse can occur; it is not a selectable operating point.

```bash
FORCE_FULL_CONFIG="$OUTPUT_ROOT/configs/pixelgen_force_full_smoke.yaml"
DIAGNOSTIC_CONFIG="$OUTPUT_ROOT/configs/pixelgen_seacache_diagnostic.yaml"
export FORCE_FULL_CONFIG DIAGNOSTIC_CONFIG
python - <<'PY'
import copy
import os
from pathlib import Path

import yaml

base = yaml.safe_load(Path("configs/pixelgen_xl_256_seacache.yaml").read_text(encoding="utf-8"))
for destination, mode, threshold in (
    (os.environ["FORCE_FULL_CONFIG"], "force_full_with_gate", 0.0),
    (os.environ["DIAGNOSTIC_CONFIG"], "seacache", 1_000_000_000.0),
):
    config = copy.deepcopy(base)
    config["checkpoint"] = os.environ["CHECKPOINT"]
    config["seacache"].update(mode=mode, threshold=threshold)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
```

Expose exactly one allocated idle GPU. The real wrapper runs Full and
force-full (`candidate`) with identical inputs; the direct command then runs
numeric diagnostic SeaCache:

```bash
: "${CUDA_VISIBLE_DEVICES:?set exactly one allocated idle GPU}"
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || {
  echo "smoke requires exactly one CUDA_VISIBLE_DEVICES entry" >&2
  return 2 2>/dev/null || exit 2
}
export CUDA_VISIBLE_DEVICES
export SEACACHE_GPU_TESTS_ALLOWED=1

SMOKE_PARITY_ROOT="$OUTPUT_ROOT/smoke/pixelgen_full_vs_force_full"
export SMOKE_PARITY_ROOT
bash scripts/run_deferred_smoke_tests.sh \
  --full-config configs/pixelgen_xl_256_full.yaml \
  --candidate-config "$FORCE_FULL_CONFIG" \
  --manifest "$SMOKE_MANIFEST" \
  --output-root "$SMOKE_PARITY_ROOT"

python scripts/generate_shard.py \
  --config "$DIAGNOSTIC_CONFIG" \
  --manifest "$SMOKE_MANIFEST" \
  --shard-id 0 --world-size 1 \
  --output-root "$OUTPUT_ROOT/smoke/pixelgen_diagnostic_seacache" \
  --acknowledge-gpu-job
```

```bash
for RUN_ROOT in \
  "$SMOKE_PARITY_ROOT/full" \
  "$SMOKE_PARITY_ROOT/candidate" \
  "$OUTPUT_ROOT/smoke/pixelgen_diagnostic_seacache"
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

Require strict checkpoint/EMA load, Full versus force-full parity, at least one
diagnostic reuse with first/final calls full, 99 combined calls, finite output,
and state cleanup before the 1K/8K stages.

## Compile-mode semantics

- `matched_eager` unwraps the upstream compiled block callables on each
  constructed denoiser and EMA instance and skips outer compilation; the
  upstream final layer is already eager.
- `blockwise` preserves the upstream block wrappers but skips Lightning's
  outer denoiser/EMA compile.
- `upstream` preserves block wrappers and delegates to upstream
  `configure_model()`, including its outer compilation.

Use one mode for both Full and SeaCache in any speedup ratio. All three remain
GPU-unverified.

## Deferred 8K threshold selection

Materialize coarse candidates from `configs/threshold_sweep_template.yaml` as
external model configs:

```bash
: "${VALIDATION_MANIFEST:?build the 8K validation manifest first}"
THRESHOLD_TEMPLATE="$BASE/configs/threshold_sweep_template.yaml"
BASE_SEACACHE_CONFIG="$BASE/configs/pixelgen_xl_256_seacache.yaml"
CANDIDATE_CONFIG_DIR="$OUTPUT_ROOT/configs/pixelgen_threshold_candidates"
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
    config["checkpoint"] = os.environ["CHECKPOINT"]
    config["seacache"].update(mode="seacache", threshold=threshold)
    tag = format(threshold, ".12g").replace("-", "m").replace(".", "p")
    path = output / f"pixelgen_seacache_threshold_{tag}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(path)
PY
sha256sum "$CANDIDATE_CONFIG_DIR"/*.yaml
```

After parity succeeds, expose exactly four allocated idle GPUs. The launcher
maps the four explicit entries to logical ranks 0-3:

```bash
: "${CUDA_VISIBLE_DEVICES:?set exactly four comma-separated allocated idle GPUs}"
IFS=',' read -r -a VALIDATION_GPUS <<<"$CUDA_VISIBLE_DEVICES"
[[ ${#VALIDATION_GPUS[@]} -eq 4 ]] || {
  echo "CUDA_VISIBLE_DEVICES must contain exactly four entries" >&2
  return 2 2>/dev/null || exit 2
}
export CUDA_VISIBLE_DEVICES
export SEACACHE_GPU_TESTS_ALLOWED=1

VALIDATION_FULL_OUTPUT_ROOT="$OUTPUT_ROOT/validation8k/pixelgen_full"
bash scripts/launch_4gpu_50k.sh \
  --config configs/pixelgen_xl_256_full.yaml \
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

Keep batch 4/rank, effective CFG batch 8, exact Heun 50, guidance 2.25,
interval `(0.1,0.9]`, timeshift 2, BF16, EMA, checkpoint, grouping, and compile
mode fixed. Select by measured quality-latency Pareto results, never by cache
ratio. These configs are **validation-only**: freeze and hash the selected
numeric config, and never tune on final 50K.

## Deferred matched single-GPU benchmark

Generate real runner JSONs for batch 1 and the common throughput batch 4:

```bash
: "${SELECTED_SEACACHE_CONFIG:?set the frozen numeric config selected on 8K}"
: "${BENCHMARK_BASE_SEED:?set a benchmark-only signed 63-bit base seed}"
PIXELGEN_BENCHMARK_BATCH1_JSON="$OUTPUT_ROOT/benchmarks/pixelgen_runner_batch1.json"
PIXELGEN_BENCHMARK_BATCH4_JSON="$OUTPUT_ROOT/benchmarks/pixelgen_runner_batch4.json"
export SELECTED_SEACACHE_CONFIG BENCHMARK_BASE_SEED
export PIXELGEN_BENCHMARK_BATCH1_JSON PIXELGEN_BENCHMARK_BATCH4_JSON

python - <<'PY'
import json
import os
from pathlib import Path

model_config = Path(os.environ["SELECTED_SEACACHE_CONFIG"]).resolve(strict=True)
base_seed = int(os.environ["BENCHMARK_BASE_SEED"])
for variable, batch_size in (
    ("PIXELGEN_BENCHMARK_BATCH1_JSON", 1),
    ("PIXELGEN_BENCHMARK_BATCH4_JSON", 4),
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
  --runner-factory seacache_style.pixelgen_benchmark:build_benchmark_spec \
  --runner-config "$PIXELGEN_BENCHMARK_BATCH1_JSON" \
  --output-json "$OUTPUT_ROOT/benchmarks/pixelgen_latency_batch1.json" \
  --warmup-batches 10 --measured-batches 30
bash scripts/benchmark_single_gpu.sh \
  --runner-factory seacache_style.pixelgen_benchmark:build_benchmark_spec \
  --runner-config "$PIXELGEN_BENCHMARK_BATCH4_JSON" \
  --output-json "$OUTPUT_ROOT/benchmarks/pixelgen_latency_batch4.json" \
  --warmup-batches 10 --measured-batches 30
```

Each factory invocation measures Full then SeaCache in one process with the
same GPU, checkpoint/EMA, batch inputs, noise, dtype, sampler, and compile mode.
For batch 4 the effective CFG batch is 8. Main speedup is median Full divided
by median SeaCache ms/image; 4-GPU wall time is separate production throughput.
The factory remains GPU-deferred until real checkpoint/parity testing.

The enclosing latency uses CUDA Events and one end-boundary synchronization.
Gate wall time uses the host clock and includes required scalar sync; FFT and
residual `cache_io` use asynchronous CUDA Event pairs resolved only after that
boundary, so no per-component barrier is added. Checkpoint load, first compile,
dataloader, CPU copy, PNG/file I/O, and metrics are excluded.

## Deferred final 50K

After threshold selection and matched benchmark:

```bash
: "${SELECTED_SEACACHE_CONFIG:?set the frozen, hashed numeric config selected on 8K}"
: "${CUDA_VISIBLE_DEVICES:?set exactly four comma-separated allocated idle GPUs}"
IFS=',' read -r -a FINAL_GPUS <<<"$CUDA_VISIBLE_DEVICES"
[[ ${#FINAL_GPUS[@]} -eq 4 ]] || {
  echo "CUDA_VISIBLE_DEVICES must contain exactly four entries" >&2
  return 2 2>/dev/null || exit 2
}
FULL_OUTPUT_ROOT="$OUTPUT_ROOT/final50k/pixelgen_full"
SEACACHE_OUTPUT_ROOT="$OUTPUT_ROOT/final50k/pixelgen_seacache"
export CUDA_VISIBLE_DEVICES FULL_OUTPUT_ROOT SEACACHE_OUTPUT_ROOT
export SEACACHE_GPU_TESTS_ALLOWED=1

bash scripts/launch_4gpu_50k.sh \
  --config configs/pixelgen_xl_256_full.yaml \
  --manifest "$FINAL_MANIFEST" --output-root "$FULL_OUTPUT_ROOT"
bash scripts/launch_4gpu_50k.sh \
  --config "$SELECTED_SEACACHE_CONFIG" \
  --manifest "$FINAL_MANIFEST" --output-root "$SEACACHE_OUTPUT_ROOT"
```

Safe resume adds `--resume` and requires identical raw/canonical manifest
identity, batch grouping, input config hash, checkpoint identity, threshold,
`noise_scale`, resolution, Git/SeaCache revisions, PyTorch version, and
`port_source_sha256`. That source hash includes executable Python/shell code,
configs and `requirements-extra.txt`, including untracked files. Before resume,
guard the same four-item `CUDA_VISIBLE_DEVICES`, unchanged config/manifest, and
existing output root. A changed environment/source requires a new output root.
A new Lightning batch always starts a new trajectory and cleanup runs in
`finally`.

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
  --output-root "$SEACACHE_OUTPUT_ROOT" \
  --resume
```

The launcher preserves `four_gpu_wall_clock_<start_ns>.json` per invocation and
updates `four_gpu_wall_clock.json`. Throughput is based only on images newly
generated in that invocation and its elapsed time; a resume result is not a
cumulative 50K throughput. Output-root and per-physical-GPU coordination locks
reject overlapping local launchers. On INT/TERM/HUP it sends no signal to
ranks; it holds the locks, waits for owned ranks to exit, and writes an
interrupted wall-clock record before exit 130 if any rank had started. Ranks read
byte-checked `config_resolved.yaml` and `input_manifest.jsonl` archives; the
source config directory is retained only for relative checkpoint resolution.
Resume baselines use durable rank metadata, and summaries must carry the current
launcher invocation ID. Any rank may atomically create an identity-compatible run
manifest, and a fully completed/no-op shard writes its summary without entering
Lightning's zero-batch prediction loop.

## Validation and metrics

```bash
python scripts/validate_outputs.py --sample-dir "$FULL_OUTPUT_ROOT/samples" \
  --manifest "$FINAL_MANIFEST" --metadata-dir "$FULL_OUTPUT_ROOT/metadata" \
  --expected-count 50000 --expected-per-class 50 \
  --expected-num-classes 1000 --resolution 256
python scripts/validate_outputs.py --sample-dir "$SEACACHE_OUTPUT_ROOT/samples" \
  --manifest "$FINAL_MANIFEST" --metadata-dir "$SEACACHE_OUTPUT_ROOT/metadata" \
  --expected-count 50000 --expected-per-class 50 \
  --expected-num-classes 1000 --resolution 256

: "${IMAGENET_REFERENCE_NPZ:?set the local ImageNet-256 ADM reference NPZ}"
: "${ADM_EVALUATOR:?set the local ADM evaluator.py}"
test -s "$IMAGENET_REFERENCE_NPZ"
test -f "$ADM_EVALUATOR"
python -m seacache_style.distribution_metrics \
  --sample-dir "$FULL_OUTPUT_ROOT/samples" --manifest "$FINAL_MANIFEST" \
  --reference-npz "$IMAGENET_REFERENCE_NPZ" --evaluator "$ADM_EVALUATOR" \
  --output-json "$FULL_OUTPUT_ROOT/distribution.json" \
  --sample-npz "$FULL_OUTPUT_ROOT/samples_adm.npz" \
  --expected-count 50000 --expected-per-class 50 \
  --expected-num-classes 1000 --resolution 256

python -m seacache_style.distribution_metrics \
  --sample-dir "$SEACACHE_OUTPUT_ROOT/samples" --manifest "$FINAL_MANIFEST" \
  --reference-npz "$IMAGENET_REFERENCE_NPZ" --evaluator "$ADM_EVALUATOR" \
  --output-json "$SEACACHE_OUTPUT_ROOT/distribution.json" \
  --sample-npz "$SEACACHE_OUTPUT_ROOT/samples_adm.npz" \
  --expected-count 50000 --expected-per-class 50 \
  --expected-num-classes 1000 --resolution 256

python -m seacache_style.paired_metrics \
  --reference-dir "$FULL_OUTPUT_ROOT/samples" \
  --candidate-dir "$SEACACHE_OUTPUT_ROOT/samples" \
  --reference-manifest "$FINAL_MANIFEST" --candidate-manifest "$FINAL_MANIFEST" \
  --reference-run-metadata "$FULL_OUTPUT_ROOT/run_manifest.json" \
  --candidate-run-metadata "$SEACACHE_OUTPUT_ROOT/run_manifest.json" \
  --output-json "$SEACACHE_OUTPUT_ROOT/paired.json" \
  --output-csv "$SEACACHE_OUTPUT_ROOT/paired.csv" \
  --resolution 256 --lpips-device cuda --lpips-batch-size 16
```

Use the same ADM evaluator/reference for Full and SeaCache and report absolute
FID/sFID/IS/precision/recall plus deltas. Paired PSNR/SSIM/LPIPS requires exact
metadata equality; the currently active compressed reference is
`PAIRED_METRICS_BLOCKED`.

The wrapper automatically reads `run_manifest.json` from each `samples/`
parent (or an explicit `--run-metadata`), verifies raw/canonical manifest and
complete run identity, archived config/manifest, sample-tree ownership, and all
four rank metadata files. It records ordered Full/SeaCache roles and hashes the
reference NPZ and evaluator before execution, rejecting concurrent changes
afterward. Paired metrics enforce the same artifact binding and ordered roles.
Relocated or legacy samples without that identity fail closed. The active upstream compressed NPZ
is not an input to this PNG/run-manifest wrapper; any distribution-only legacy
ADM conversion must be separately documented and validated and still cannot
support paired metrics.

```bash
python scripts/compare_distribution.py \
  --full-json "$FULL_OUTPUT_ROOT/distribution.json" \
  --seacache-json "$SEACACHE_OUTPUT_ROOT/distribution.json" \
  --output-json "$SEACACHE_OUTPUT_ROOT/distribution_deltas.json"
```

## Limitations

Threshold, GPU parity, compile compatibility, latency and SeaCache 50K are not
yet measured. Batch size/grouping affects decisions. Neither local SeaCache nor
PixelGen contains a license file; see `NOTICE.md`. Detailed evidence and order
are in `AUDIT.md`, `RUNBOOK.md`, and `IMPLEMENTATION_REPORT.md`.
