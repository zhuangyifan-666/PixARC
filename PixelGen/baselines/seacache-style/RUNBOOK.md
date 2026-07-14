# PixelGen ordered runbook

Phases 4 onward are deferred while PixelGen is ACTIVE and JiT is SCHEDULED.

## 1. Safety and environment

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
bash scripts/inspect_active_runs.sh
```

Do not set `SEACACHE_GPU_TESTS_ALLOWED=1` until every target GPU is idle.

## 2. CPU gate

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/pixarc-pixelgen-seacache-pycache \
  "$PIXELGEN_PYTHON" -m compileall "$BASE"
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/matplotlib-seacache-test \
  "$PIXELGEN_PYTHON" -m unittest discover -s "$BASE/tests" -v
git -C "$ROOT" diff --check -- PixelGen/baselines/seacache-style
```

`git diff --check` omits untracked files. Inspect `git status --short`; after
intentionally staging new files, also run
`git -C "$ROOT" diff --cached --check`.

The configs use `InferenceOnlyTrainer`, a parameter-free prediction component
that avoids training-only LPIPS/DINO imports and `torch.hub` access. It cannot
be used for training.

## 3. Manifests

Create all manifests outside the source tree under `OUTPUT_ROOT`, using
mutually disjoint seed ranges:

```bash
: "${SMOKE_BASE_SEED:?set 8-sample smoke base seed}"
: "${PROXY_BASE_SEED:?set 1K proxy base seed}"
: "${VALIDATION_BASE_SEED:?set 8K validation base seed}"
: "${FINAL_BASE_SEED:?set 50K final base seed}"
SMOKE_MANIFEST="$OUTPUT_ROOT/manifests/pixelgen_smoke8.jsonl"
PROXY_MANIFEST="$OUTPUT_ROOT/manifests/pixelgen1k.jsonl"
VALIDATION_MANIFEST="$OUTPUT_ROOT/manifests/pixelgen8k.jsonl"
FINAL_MANIFEST="$OUTPUT_ROOT/manifests/pixelgen50k.jsonl"
export SMOKE_BASE_SEED SMOKE_MANIFEST PROXY_MANIFEST
export VALIDATION_MANIFEST FINAL_MANIFEST

python -c 'import os; from seacache_style.manifest import build_manifest, write_manifest; seed=int(os.environ["SMOKE_BASE_SEED"]); rows=build_manifest(samples_per_class=1, base_seed=seed, split_name="pixelgen_smoke8", world_size=1, batch_size=4, num_classes=8); write_manifest(rows, os.environ["SMOKE_MANIFEST"], base_seed=seed, world_size=1, batch_size=4)'
python scripts/build_manifest.py --output "$PROXY_MANIFEST" \
  --samples-per-class 1 --base-seed "$PROXY_BASE_SEED" \
  --split-name pixelgen1k_proxy --world-size 4 --batch-size 4
python scripts/build_manifest.py --output "$VALIDATION_MANIFEST" \
  --samples-per-class 8 --base-seed "$VALIDATION_BASE_SEED" \
  --split-name pixelgen8k_validation --world-size 4 --batch-size 4
python scripts/build_manifest.py --output "$FINAL_MANIFEST" \
  --samples-per-class 50 --base-seed "$FINAL_BASE_SEED" \
  --split-name pixelgen50k_final --world-size 4 --batch-size 4

python scripts/validate_manifest.py --manifest "$SMOKE_MANIFEST" \
  --expected-count 8 --expected-per-class 1 --expected-num-classes 8 \
  --world-size 1 --batch-size 4 \
  --disjoint-with "$PROXY_MANIFEST" \
  --disjoint-with "$VALIDATION_MANIFEST" \
  --disjoint-with "$FINAL_MANIFEST"
python scripts/validate_manifest.py --manifest "$PROXY_MANIFEST" \
  --expected-count 1000 --expected-per-class 1 --expected-num-classes 1000 \
  --world-size 4 --batch-size 4 \
  --disjoint-with "$SMOKE_MANIFEST" \
  --disjoint-with "$VALIDATION_MANIFEST" \
  --disjoint-with "$FINAL_MANIFEST"
python scripts/validate_manifest.py --manifest "$VALIDATION_MANIFEST" \
  --expected-count 8000 --expected-per-class 8 --expected-num-classes 1000 \
  --world-size 4 --batch-size 4 \
  --disjoint-with "$SMOKE_MANIFEST" \
  --disjoint-with "$PROXY_MANIFEST" \
  --disjoint-with "$FINAL_MANIFEST"
python scripts/validate_manifest.py --manifest "$FINAL_MANIFEST" \
  --expected-count 50000 --expected-per-class 50 --expected-num-classes 1000 \
  --world-size 4 --batch-size 4 \
  --disjoint-with "$SMOKE_MANIFEST" \
  --disjoint-with "$PROXY_MANIFEST" \
  --disjoint-with "$VALIDATION_MANIFEST"
```

Archive each SHA-256 sidecar; never edit a manifest after generation starts.

## 4. Deferred smoke

The implemented probe is the first block attention-branch input:

```python
condition = self.t_embedder(t) + self.y_embedder(y)
body_input = self.x_embedder(x)
body_input += self.pos_embed
shift_msa, scale_msa, *_ = self.blocks[0].adaLN_modulation(condition).chunk(6, dim=-1)
probe_raw = modulate(self.blocks[0].norm1(body_input), shift_msa, scale_msa)
```

Create external force-full and large-threshold numeric diagnostic YAMLs:

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
    config["runtime"]["batch_size"] = 4
    config["data"]["train_batch_size"] = 4
    config["data"]["pred_batch_size"] = 4
    config["seacache"].update(mode=mode, threshold=threshold)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
```

After read-only GPU inspection is clean, expose exactly one allocated idle GPU.
The real wrapper runs Full and force-full (`candidate`); the direct command
runs diagnostic SeaCache:

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

for RUN_ROOT in \
  "$SMOKE_PARITY_ROOT/full" \
  "$SMOKE_PARITY_ROOT/candidate" \
  "$OUTPUT_ROOT/smoke/pixelgen_diagnostic_seacache"
do
  python scripts/validate_outputs.py \
    --sample-dir "$RUN_ROOT/samples" \
    --manifest "$SMOKE_MANIFEST" \
    --metadata-dir "$RUN_ROOT/metadata" \
    --expected-count 8 --expected-per-class 1 \
    --expected-num-classes 8 --resolution 256
done
```

Required evidence: checkpoint/EMA load, Full/force-full parity, real reuse,
first/final full, 99 combined calls, preserved diagnostic returns, finite
output, deepcopy isolation and cleanup. The `1e9` threshold is diagnostic only,
never a selected operating point.

## 5. Deferred compile matrix

Test `matched_eager`, `blockwise`, and `upstream` for Full, force-full, and
numeric SeaCache. `matched_eager` unwraps the upstream compiled block callables
on each constructed denoiser and EMA instance and skips outer compilation; the
upstream final layer is already eager.
`blockwise` preserves upstream block wrappers but skips Lightning's outer
compile. `upstream` preserves the wrappers and delegates to upstream outer
denoiser/EMA compilation. Record correctness, graph breaks, compile seconds,
recompiles, steady latency, wrapper counts, and memory. Never mix modes in a
speedup ratio.

## 6. Deferred 1K and 8K

Materialize the coarse candidates from the threshold template into external
configs with an absolute checkpoint:

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

Use `PROXY_MANIFEST` first to reject broken configs. Then, only after parity
passes and four allocated GPUs are idle, generate the Full 8K reference and
the candidate sweep:

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

Keep batch 4/rank, effective 2B=8, exact Heun 50, guidance 2.25, interval
`(0.1,0.9]`, timeshift 2, BF16, EMA, grouping, checkpoint, and compile mode
fixed. Select numeric points from measured quality-latency Pareto results,
never cache ratio. Candidate configs are **validation-only**; freeze and hash
`SELECTED_SEACACHE_CONFIG`, and never tune or reselect on final 50K.

## 7. Deferred matched benchmark

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
  --output-json "$OUTPUT_ROOT/benchmarks/pixelgen_batch1.json" \
  --warmup-batches 10 --measured-batches 30
bash scripts/benchmark_single_gpu.sh \
  --runner-factory seacache_style.pixelgen_benchmark:build_benchmark_spec \
  --runner-config "$PIXELGEN_BENCHMARK_BATCH4_JSON" \
  --output-json "$OUTPUT_ROOT/benchmarks/pixelgen_batch4.json" \
  --warmup-batches 10 --measured-batches 30
```

Each invocation measures Full then SeaCache in one process on one visible GPU
with identical inputs. Report batch 1 and common throughput batch 4. The
enclosing latency uses CUDA Events and one end-boundary synchronization. Gate
wall time uses the host clock and includes required scalar sync; FFT and
residual `cache_io` use asynchronous CUDA Event pairs resolved after that
boundary, with no per-component barrier. Exclude checkpoint load, first
compile, dataloader, CPU copy, PNG/file I/O and metrics.

## 8. Deferred final 50K

```bash
: "${SELECTED_SEACACHE_CONFIG:?set the frozen, hashed config selected on 8K}"
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
bash scripts/launch_4gpu_50k.sh --config configs/pixelgen_xl_256_full.yaml \
  --manifest "$FINAL_MANIFEST" --output-root "$FULL_OUTPUT_ROOT"
bash scripts/launch_4gpu_50k.sh --config "$SELECTED_SEACACHE_CONFIG" \
  --manifest "$FINAL_MANIFEST" --output-root "$SEACACHE_OUTPUT_ROOT"
```

Resume only with `--resume`, identical raw/canonical manifest identity, input
config, checkpoint, threshold, `noise_scale`, batch grouping, compile mode,
Git/SeaCache revisions, PyTorch version and `port_source_sha256`, plus the same
explicit four-GPU mapping. The source hash includes untracked executable port
files, configs and `requirements-extra.txt`:

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

Every invocation preserves `four_gpu_wall_clock_<start_ns>.json` and updates
`four_gpu_wall_clock.json`. `images_per_second` uses only images newly generated
in that invocation and that invocation's elapsed time, so the latest resume
file is not cumulative 50K throughput. Archive every timestamped file.
Output-root and per-physical-GPU coordination locks reject overlapping local
launchers. On INT/TERM/HUP it sends no signal to ranks and waits while holding
the locks, then writes an interrupted wall-clock record before exit 130 if at
least one rank has started; before that point it exits 130 without a timing
record. After an uncatchable crash, remove a stale
`/tmp/pixarc-seacache-gpu-locks/*.lock` only after checking no corresponding
rank is active. Every rank reads byte-checked archived config/manifest bytes;
the source config directory is used only for relative checkpoint resolution.
Resume baselines come from durable metadata, and only summaries carrying the
current invocation ID contribute to wall-clock validation. Any rank may
atomically create a same-identity run manifest; a no-op shard writes a current
summary directly instead of relying on Lightning zero-batch hooks.

## 9. Validate and evaluate

Validate both sample directories before any metric:

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

Run distribution metrics for Full and SeaCache against the same explicit local
ADM artifacts; nothing is downloaded automatically:

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
  --output-json "$FULL_OUTPUT_ROOT/distribution.json" \
  --sample-npz "$FULL_OUTPUT_ROOT/samples_adm.npz" \
  --expected-count 50000 --expected-per-class 50 \
  --expected-num-classes 1000 --resolution 256
python -m seacache_style.distribution_metrics \
  --sample-dir "$SEACACHE_OUTPUT_ROOT/samples" \
  --manifest "$FINAL_MANIFEST" \
  --reference-npz "$IMAGENET_REFERENCE_NPZ" \
  --evaluator "$ADM_EVALUATOR" \
  --output-json "$SEACACHE_OUTPUT_ROOT/distribution.json" \
  --sample-npz "$SEACACHE_OUTPUT_ROOT/samples_adm.npz" \
  --expected-count 50000 --expected-per-class 50 \
  --expected-num-classes 1000 --resolution 256
python scripts/compare_distribution.py \
  --full-json "$FULL_OUTPUT_ROOT/distribution.json" \
  --seacache-json "$SEACACHE_OUTPUT_ROOT/distribution.json" \
  --output-json "$SEACACHE_OUTPUT_ROOT/distribution_deltas.json"
```

Run paired metrics only for the new manifest-driven Full/SeaCache pair:

```bash
python -m seacache_style.paired_metrics \
  --reference-dir "$FULL_OUTPUT_ROOT/samples" \
  --candidate-dir "$SEACACHE_OUTPUT_ROOT/samples" \
  --reference-manifest "$FINAL_MANIFEST" \
  --candidate-manifest "$FINAL_MANIFEST" \
  --reference-run-metadata "$FULL_OUTPUT_ROOT/run_manifest.json" \
  --candidate-run-metadata "$SEACACHE_OUTPUT_ROOT/run_manifest.json" \
  --output-json "$SEACACHE_OUTPUT_ROOT/paired.json" \
  --output-csv "$SEACACHE_OUTPUT_ROOT/paired.csv" \
  --resolution 256 --lpips-device cuda --lpips-batch-size 16
```

The distribution wrapper automatically reads `run_manifest.json` from each
`samples/` parent (or an explicit `--run-metadata`), verifies raw/canonical
manifest, complete run identity, archived inputs, sample-tree ownership, and
all rank metadata. It records ordered Full/SeaCache roles and hashes both the
reference NPZ and evaluator before use, checking again afterward. Paired
metrics enforce the same artifact binding and reject reversed roles. Relocated
or legacy samples without that identity fail closed. The active compressed Full
NPZ may be eligible only for a separately documented legacy ADM conversion; it
is not accepted by this PNG/run-manifest wrapper, and strict pairing is blocked.

Archive configs/hashes, manifests, run manifests, four rank metadata files,
summaries/logs, every `four_gpu_wall_clock_<start_ns>.json`, validation reports,
metric JSON/CSV and benchmark JSON. Do not commit PNG, NPZ, checkpoint or large
logs.
