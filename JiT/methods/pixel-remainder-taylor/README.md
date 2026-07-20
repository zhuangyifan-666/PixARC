# Pixel-Remainder Taylor

This directory is the canonical shared implementation. It does not modify the
four frozen baselines or either `third-party` tree. JiT executes two B-sized CFG
model calls per NFE (198 calls for exact 50-step Heun); PixelGen executes one
combined `[unconditional, conditional]` 2B call per NFE (99 calls). Pixel
planning runs only after guidance and never invokes the model.

Adaptive feature and pixel forecasts use the most recent exact anchors with
non-uniform Lagrange extrapolation and FP32 accumulation before one output
cast. This is required because dynamically selected Full spans do not form a
uniform grid. The debug-only fixed parity mode keeps the frozen TaylorSeer
recursive arithmetic so its output remains directly replayable.

The only primary sweep controls are `tau` and `max_taylor_span`. Feature history
is fixed at order two, pixel history at order three, warmup at three Full NFE,
pooling at 8, and batch reduction at the real-batch mean. `instrumented_full`
and `fixed_schedule_parity` are smoke/debug configurations and must not be used
as extra 1K trajectories.

## Local CPU verification

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
python -m pytest -q -p no:cacheprovider \
  JiT/methods/pixel-remainder-taylor/tests
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
python -m pytest -q -p no:cacheprovider \
  PixelGen/methods/pixel-remainder-taylor/tests
```

## Remote prerequisites

Run from the PixARC checkout after `git pull`. Keep generated images outside the
checkout. The generator deliberately refuses repository-local output roots.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PIXEL_REMAINDER_GPU_RUN_ALLOWED=1
export RUN_ROOT=/absolute/path/outside/PixARC/prt
export ADM_REFERENCE=/absolute/path/imagenet_256_adm_reference.npz
export ADM_EVALUATOR=/absolute/path/ADM/evaluator.py
export PIXEL_REMAINDER_PYTHON=/absolute/path/to/the/model-family/python
```

Check checkpoint paths in the selected YAML files. JiT manifests must record
`generator_device=cuda`; PixelGen manifests must record `generator_device=cpu`.

## Required smoke sequence

Create eight-sample manifests without touching the 1K manifests:

```bash
mkdir -p "$RUN_ROOT/protocol"
python JiT/baselines/taylorseer-style/scripts/build_manifest.py \
  --output "$RUN_ROOT/protocol/jit_smoke8.jsonl" --samples-per-class 1 \
  --num-classes 8 --base-seed 0 --split-name prt-smoke8 --world-size 4 \
  --batch-size 32 --generator-device cuda
python PixelGen/baselines/taylorseer-style/scripts/build_manifest.py \
  --output "$RUN_ROOT/protocol/pixelgen_smoke8.jsonl" --samples-per-class 1 \
  --num-classes 8 --base-seed 0 --split-name prt-smoke8 --world-size 4 \
  --batch-size 4 --generator-device cpu
```

For each model, run `instrumented_full`, `fixed_i3_k2`, then the `t0p01_h3`
dynamic config with the launcher below and `--expected-count 8`. Compare the
first two outputs against already-existing matched Full and TaylorSeer images;
do not generate a new baseline. The validator checks 99 NFE, full traces, image
integrity, and the 198/99 forward contract. Before a 1K run, inspect the dynamic
trace and require at least one nonzero `selected_span`.

Example dynamic smoke:

```bash
bash JiT/methods/pixel-remainder-taylor/scripts/launch_4gpu.sh \
  --model JiT \
  --config JiT/methods/pixel-remainder-taylor/configs/jit_b16_256_prt_t0p01_h3.yaml \
  --manifest "$RUN_ROOT/protocol/jit_smoke8.jsonl" \
  --output-root "$RUN_ROOT/smoke/jit_t0p01" --expected-count 8
```

Repeat with `--model PixelGen`, its config, and its CPU-noise manifest. Use a
new empty output directory per configuration. `--resume` is accepted only for
a validated partial run and never overwrites complete groups.

## New-method-only 1K sweep

Use the frozen manifests already checked into `results/.../protocol`; no
baseline command appears in this sweep. Run all three configs for each model:

```bash
export PIXEL_REMAINDER_PYTHON=/absolute/path/to/jit/python
for TAU in 01 02 04; do
  bash JiT/methods/pixel-remainder-taylor/scripts/launch_4gpu.sh \
    --model JiT \
    --config "JiT/methods/pixel-remainder-taylor/configs/jit_b16_256_prt_t0p${TAU}_h3.yaml" \
    --manifest results/JiT/baselines/taylorseer/protocol/manifest_1k.jsonl \
    --output-root "$RUN_ROOT/1k/jit_prt_t0p${TAU}_h3" --expected-count 1000

done

export PIXEL_REMAINDER_PYTHON=/absolute/path/to/pixelgen/python
for TAU in 01 02 04; do
  bash JiT/methods/pixel-remainder-taylor/scripts/launch_4gpu.sh \
    --model PixelGen \
    --config "PixelGen/methods/pixel-remainder-taylor/configs/pixelgen_xl_256_prt_t0p${TAU}_h3.yaml" \
    --manifest results/PixelGen/baselines/taylorseer/protocol/manifest_1k.jsonl \
    --output-root "$RUN_ROOT/1k/pixelgen_prt_t0p${TAU}_h3" --expected-count 1000
done
```

The restart-safe cumulative launcher wall clock is in `launcher_timing.json`;
individual invocations are immutable records below `launcher_invocations/`.
Each output contains immutable input snapshots,
per-image metadata, per-rank summaries, and complete JSONL trajectory traces.

## Evaluation and final tables

`evaluate_1k.py` uses the same metric primitives as the frozen baseline suite,
but has a PRT-specific identity validator instead of mislabeling the candidate
as TaylorSeer. It requires an existing matched Full image root, the ADM
reference NPZ and evaluator. Example:

```bash
python JiT/methods/pixel-remainder-taylor/scripts/evaluate_1k.py \
  --model JiT --run prt_t0p01_h3 \
  --candidate-root "$RUN_ROOT/1k/jit_prt_t0p01_h3" \
  --reference-root /absolute/path/to/existing/jit_full_1k \
  --manifest results/JiT/baselines/taylorseer/protocol/manifest_1k.jsonl \
  --reference-manifest results/JiT/baselines/taylorseer/protocol/manifest_1k.jsonl \
  --reference-npz "$ADM_REFERENCE" --evaluator "$ADM_EVALUATOR" \
  --trace "$RUN_ROOT/1k/jit_prt_t0p01_h3/traces/rank_*.jsonl" \
  --output-dir "$RUN_ROOT/eval"
```

Repeat for six measured runs. Then merge only their measured CSV files:

```bash
python JiT/methods/pixel-remainder-taylor/scripts/merge_results.py \
  --summary "$RUN_ROOT/eval/*_summary.csv" \
  --trace "$RUN_ROOT/eval/*_trace.csv" \
  --results-root results
```

This creates the four required files:

- `results/pixel_remainder_taylor_1k_summary.csv`
- `results/pixel_remainder_taylor_1k_trace.csv`
- `results/pixel_remainder_taylor_1k_comparison.csv`
- `results/PIXEL_REMAINDER_TAYLOR_1K_REPORT.md`

The comparison builder reads the existing TaylorSeer, SeaCache, SpeCa and
DiCache CSVs; it never starts their generators. It marks speed/FID,
speed/LPIPS and speed/MSE Pareto frontiers. Missing measurements remain empty.
