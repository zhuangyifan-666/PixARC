# Pixel-Remainder Taylor: 1K and 50K runbook

All generation roots must be outside the Git checkout. The launcher refuses a
dirty executable worktree, a non-empty run without `--resume`, a manifest count
mismatch, busy/aliased GPUs, or fewer/more than four physical GPUs.

Each run archives the exact authoring YAML as `input_config.yaml` and a separate
canonical, parent-expanded, portable YAML as `config_resolved.yaml`. The latter
has no `extends`, has `template_only: false`, and contains an absolute existing
checkpoint. On resume, both raw bytes and a newly resolved parent chain must
match the archived files. The launcher owns these files; generators only read
and validate them.

## Environment

```bash
export REPAIR=/mnt/iset/nfs-main/private/zhuangyifan/PixARC_prt_ready_20260720
export CONFIG_SOURCE=/mnt/iset/nfs-main/private/zhuangyifan/PixARC
export RUN_ROOT=/mnt/iset/nfs-main/private/zhuangyifan/PixARC-runs/pixel-remainder-taylor-repair
export LAUNCH="$REPAIR/JiT/methods/pixel-remainder-taylor/scripts/launch_4gpu.sh"
export JIT_PYTHON=/root/miniconda3/envs/jit/bin/python
export PIXELGEN_PYTHON=/root/miniconda3/envs/pixelgen/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PIXEL_REMAINDER_GPU_RUN_ALLOWED=1
mkdir -p "$RUN_ROOT/protocol"
```

The repaired worktree intentionally does not copy ignored checkpoints. The
configuration source above is the existing checkout that owns the two verified
checkpoint files; the launcher snapshots the exact YAML bytes into each output.

## CPU gate and smoke manifests

```bash
cd "$REPAIR"
JiT/methods/pixel-remainder-taylor/scripts/run_all_cpu_tests.sh
JiT/methods/pixel-remainder-taylor/scripts/preflight_all_configs.sh

"$JIT_PYTHON" JiT/methods/pixel-remainder-taylor/scripts/build_smoke_manifest.py \
  --manifest results/JiT/baselines/taylorseer/protocol/manifest_1k.jsonl \
  --output "$RUN_ROOT/protocol/jit_smoke128.jsonl" \
  --world-size 4 --batch-size 32 --generator-device cuda

"$PIXELGEN_PYTHON" JiT/methods/pixel-remainder-taylor/scripts/build_smoke_manifest.py \
  --manifest results/PixelGen/baselines/taylorseer/protocol/manifest_1k.jsonl \
  --output "$RUN_ROOT/protocol/pixelgen_smoke16.jsonl" \
  --world-size 4 --batch-size 4 --generator-device cpu
```

These select sample IDs 0–127 for JiT and 0–15 for PixelGen, which are one
complete real batch group per shard while preserving the frozen IDs/classes,
seeds, group IDs, and positions.

## Required four-GPU smoke matrix

Run all four modes for each model family. A failed command is a hard gate.

```bash
cd "$REPAIR"
export PIXEL_REMAINDER_PYTHON="$JIT_PYTHON"
for SPEC in \
  full:jit_b16_256_instrumented_full.yaml \
  fixed_i3_k2:jit_b16_256_fixed_i3_k2.yaml \
  dynamic_t0p01:jit_b16_256_prt_t0p01_h3.yaml \
  dynamic_t0p04:jit_b16_256_prt_t0p04_h3.yaml
do
  NAME=${SPEC%%:*}; CONFIG=${SPEC#*:}
  "$LAUNCH" --model JiT \
    --config "$CONFIG_SOURCE/JiT/methods/pixel-remainder-taylor/configs/$CONFIG" \
    --manifest "$RUN_ROOT/protocol/jit_smoke128.jsonl" \
    --output-root "$RUN_ROOT/smoke/jit_$NAME" --expected-count 128
done

export PIXEL_REMAINDER_PYTHON="$PIXELGEN_PYTHON"
for SPEC in \
  full:pixelgen_xl_256_instrumented_full.yaml \
  fixed_i3_k2:pixelgen_xl_256_fixed_i3_k2.yaml \
  dynamic_t0p01:pixelgen_xl_256_prt_t0p01_h3.yaml \
  dynamic_t0p04:pixelgen_xl_256_prt_t0p04_h3.yaml
do
  NAME=${SPEC%%:*}; CONFIG=${SPEC#*:}
  "$LAUNCH" --model PixelGen \
    --config "$CONFIG_SOURCE/PixelGen/methods/pixel-remainder-taylor/configs/$CONFIG" \
    --manifest "$RUN_ROOT/protocol/pixelgen_smoke16.jsonl" \
    --output-root "$RUN_ROOT/smoke/pixelgen_$NAME" --expected-count 16
done
```

Validate the two dynamic runs as a single protocol gate.  A conservative lower
tau is allowed to remain all-Full, but at least one run must execute Taylor and
the Taylor ratio must not decrease as tau increases:

```bash
"$JIT_PYTHON" JiT/methods/pixel-remainder-taylor/scripts/validate_dynamic_matrix.py \
  --lower-run "$RUN_ROOT/smoke/jit_dynamic_t0p01" \
  --upper-run "$RUN_ROOT/smoke/jit_dynamic_t0p04"
"$PIXELGEN_PYTHON" JiT/methods/pixel-remainder-taylor/scripts/validate_dynamic_matrix.py \
  --lower-run "$RUN_ROOT/smoke/pixelgen_dynamic_t0p01" \
  --upper-run "$RUN_ROOT/smoke/pixelgen_dynamic_t0p04"
```

Pixel-identical Full and fixed-schedule parity checks use the existing frozen
baseline images and the smoke manifests:

```bash
"$JIT_PYTHON" JiT/methods/pixel-remainder-taylor/scripts/compare_image_trees.py \
  --candidate-root "$RUN_ROOT/smoke/jit_full/samples" \
  --reference-root /mnt/iset/nfs-main/private/zhuangyifan/PixARC-runs/taylorseer/jit_1k/runs/full/samples \
  --manifest "$RUN_ROOT/protocol/jit_smoke128.jsonl"
"$JIT_PYTHON" JiT/methods/pixel-remainder-taylor/scripts/compare_image_trees.py \
  --candidate-root "$RUN_ROOT/smoke/jit_fixed_i3_k2/samples" \
  --reference-root /mnt/iset/nfs-main/private/zhuangyifan/PixARC-runs/taylorseer/jit_1k/runs/i3_k2/samples \
  --manifest "$RUN_ROOT/protocol/jit_smoke128.jsonl"
"$JIT_PYTHON" JiT/methods/pixel-remainder-taylor/scripts/compare_image_trees.py \
  --candidate-root "$RUN_ROOT/smoke/pixelgen_full/samples" \
  --reference-root /mnt/iset/nfs-main/private/zhuangyifan/PixARC-runs/taylorseer/pixelgen_1k/runs/full/samples \
  --manifest "$RUN_ROOT/protocol/pixelgen_smoke16.jsonl"
"$JIT_PYTHON" JiT/methods/pixel-remainder-taylor/scripts/compare_image_trees.py \
  --candidate-root "$RUN_ROOT/smoke/pixelgen_fixed_i3_k2/samples" \
  --reference-root /mnt/iset/nfs-main/private/zhuangyifan/PixARC-runs/taylorseer/pixelgen_1k/runs/i3_k2/samples \
  --manifest "$RUN_ROOT/protocol/pixelgen_smoke16.jsonl"
```

To exercise recovery, interrupt only between launcher invocations (the launcher
does not signal external or owned ranks), then repeat the identical command with
`--resume`. Inspect `launcher_invocations/*.json` and
`launcher_timing.json`; throughput must use `cumulative_elapsed_seconds`.

Do not start 1K unless all CPU tests, Full/fixed/dynamic smoke runs, parity
checks, forward counts, resume validation, and the nondecreasing average Taylor
ratio from tau 0.01 to 0.04 pass.

Every formal command below reruns the no-GPU input preflight inside the launcher
before it inspects or locks a GPU. Add `--resume` only to the exact same command
and output root after a controlled incomplete invocation. Raw YAML formatting,
PixelGen parent semantics, manifest, sidecar, checkpoint size, code tree and
runtime identity must remain unchanged.

## Six new-method 1K runs

```bash
cd "$REPAIR"
export PIXEL_REMAINDER_PYTHON="$JIT_PYTHON"
for TAU in 01 02 04; do
  "$LAUNCH" --model JiT \
    --config "$CONFIG_SOURCE/JiT/methods/pixel-remainder-taylor/configs/jit_b16_256_prt_t0p${TAU}_h3.yaml" \
    --manifest results/JiT/baselines/taylorseer/protocol/manifest_1k.jsonl \
    --output-root "$RUN_ROOT/1k/jit_prt_t0p${TAU}_h3" --expected-count 1000
done

export PIXEL_REMAINDER_PYTHON="$PIXELGEN_PYTHON"
for TAU in 01 02 04; do
  "$LAUNCH" --model PixelGen \
    --config "$CONFIG_SOURCE/PixelGen/methods/pixel-remainder-taylor/configs/pixelgen_xl_256_prt_t0p${TAU}_h3.yaml" \
    --manifest results/PixelGen/baselines/taylorseer/protocol/manifest_1k.jsonl \
    --output-root "$RUN_ROOT/1k/pixelgen_prt_t0p${TAU}_h3" --expected-count 1000
done
```

Add `--resume` to an identical command after a failed/incomplete invocation.
Never rerun Full, TaylorSeer, SeaCache, SpeCa, or DiCache baselines.

## Generic evaluation

```bash
export ADM_EVALUATOR=/root/third_party/guided-diffusion/evaluations/evaluator.py
export IMAGENET_REFERENCE_NPZ=/root/.cache/adm_eval/VIRTUAL_imagenet256_labeled.npz
mkdir -p "$RUN_ROOT/eval"

"$JIT_PYTHON" JiT/methods/pixel-remainder-taylor/scripts/evaluate_run.py \
  --model JiT --run prt_t0p01_h3 \
  --candidate-root "$RUN_ROOT/1k/jit_prt_t0p01_h3" \
  --manifest results/JiT/baselines/taylorseer/protocol/manifest_1k.jsonl \
  --expected-count 1000 --reference-npz "$IMAGENET_REFERENCE_NPZ" \
  --evaluator "$ADM_EVALUATOR" \
  --timing "$RUN_ROOT/1k/jit_prt_t0p01_h3/launcher_timing.json" \
  --trace "$RUN_ROOT/1k/jit_prt_t0p01_h3/traces/rank_*.jsonl" \
  --paired-reference-root /mnt/iset/nfs-main/private/zhuangyifan/PixARC-runs/taylorseer/jit_1k/runs/full \
  --reference-manifest results/JiT/baselines/taylorseer/protocol/manifest_1k.jsonl \
  --baseline-summary results/taylorseer_1k_summary.csv \
  --output-dir "$RUN_ROOT/eval"
```

Repeat for all six runs/model identities. For a 50K run, set
`--expected-count 50000`; omit paired-reference arguments when no matched Full
tree exists, and omit `--baseline-summary` unless a trusted matched 50K Full
timing is available. The evaluator then reports raw images/s without inventing a
speedup.

Merge measured per-run CSVs only after evaluation:

```bash
"$JIT_PYTHON" JiT/methods/pixel-remainder-taylor/scripts/merge_results.py \
  --summary "$RUN_ROOT/eval/*_summary.csv" \
  --trace "$RUN_ROOT/eval/*_trace.csv" --results-root results
```

## Build and run 50K after selecting one tau

The base seeds below are examples and must remain explicit. Both commands reject
any overlap with the frozen 1K tuning seeds.

```bash
cd "$REPAIR"
"$JIT_PYTHON" JiT/methods/pixel-remainder-taylor/scripts/build_manifest.py \
  --output "$RUN_ROOT/protocol/jit_50k.jsonl" --samples-per-class 50 \
  --num-classes 1000 --base-seed 303000000000 --split-name prt-50k \
  --world-size 4 --batch-size 32 --generator-device cuda \
  --disjoint-from results/JiT/baselines/taylorseer/protocol/manifest_1k.jsonl
"$PIXELGEN_PYTHON" JiT/methods/pixel-remainder-taylor/scripts/build_manifest.py \
  --output "$RUN_ROOT/protocol/pixelgen_50k.jsonl" --samples-per-class 50 \
  --num-classes 1000 --base-seed 404000000000 --split-name prt-50k \
  --world-size 4 --batch-size 4 --generator-device cpu \
  --disjoint-from results/PixelGen/baselines/taylorseer/protocol/manifest_1k.jsonl

export SELECTED_TAU=0.02
"$JIT_PYTHON" JiT/methods/pixel-remainder-taylor/scripts/materialize_50k_config.py \
  --model JiT \
  --base "$CONFIG_SOURCE/JiT/methods/pixel-remainder-taylor/configs/jit_b16_256_prt_t0p02_h3.yaml" \
  --output "$RUN_ROOT/protocol/jit_selected_50k.yaml" \
  --tau "$SELECTED_TAU" --max-taylor-span 3
"$JIT_PYTHON" JiT/methods/pixel-remainder-taylor/scripts/materialize_50k_config.py \
  --model PixelGen \
  --base "$CONFIG_SOURCE/PixelGen/methods/pixel-remainder-taylor/configs/pixelgen_xl_256_prt_t0p02_h3.yaml" \
  --output "$RUN_ROOT/protocol/pixelgen_selected_50k.yaml" \
  --tau "$SELECTED_TAU" --max-taylor-span 3

export PIXEL_REMAINDER_PYTHON="$JIT_PYTHON"
"$LAUNCH" --model JiT --config "$RUN_ROOT/protocol/jit_selected_50k.yaml" \
  --manifest "$RUN_ROOT/protocol/jit_50k.jsonl" \
  --output-root "$RUN_ROOT/50k/jit_prt_selected_h3" --expected-count 50000
export PIXEL_REMAINDER_PYTHON="$PIXELGEN_PYTHON"
"$LAUNCH" --model PixelGen --config "$RUN_ROOT/protocol/pixelgen_selected_50k.yaml" \
  --manifest "$RUN_ROOT/protocol/pixelgen_50k.jsonl" \
  --output-root "$RUN_ROOT/50k/pixelgen_prt_selected_h3" --expected-count 50000
```

The materialized 50K YAML exposes only `tau` and `max_taylor_span` as quality
controls and always uses `trace_mode: summary`. The core sampler and recovery,
timing, validation, aggregation, and evaluation code are identical to 1K.
