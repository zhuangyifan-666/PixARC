# Pixel-Remainder Taylor formal 1K commands

These commands are prepared but were not started. They are authorized only
after `CODEX_REPAIR_REPORT.md` says `1K readiness: PASS`, including all JiT and
PixelGen four-GPU Full/fixed/dynamic, pixel-alignment, and recovery gates.
Never use these commands to rerun a baseline.

```bash
cd /mnt/iset/nfs-main/private/zhuangyifan/PixARC_prt_configfix_20260720T110522Z
export REPAIR="$PWD"
export CONFIG_SOURCE=/mnt/iset/nfs-main/private/zhuangyifan/PixARC
export RUN_ROOT=/mnt/iset/nfs-main/private/zhuangyifan/PixARC-runs/pixel-remainder-taylor-configfix
export LAUNCH="$REPAIR/JiT/methods/pixel-remainder-taylor/scripts/launch_4gpu.sh"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PIXEL_REMAINDER_GPU_RUN_ALLOWED=1
mkdir -p "$RUN_ROOT/1k"

nvidia-smi -L
JiT/methods/pixel-remainder-taylor/scripts/run_all_cpu_tests.sh
JiT/methods/pixel-remainder-taylor/scripts/preflight_all_configs.sh
```

## JiT tau=0.01

```bash
cd "$REPAIR"
export PIXEL_REMAINDER_PYTHON=/root/miniconda3/envs/jit/bin/python
"$LAUNCH" --model JiT \
  --config "$CONFIG_SOURCE/JiT/methods/pixel-remainder-taylor/configs/jit_b16_256_prt_t0p01_h3.yaml" \
  --manifest "$REPAIR/results/JiT/baselines/taylorseer/protocol/manifest_1k.jsonl" \
  --output-root "$RUN_ROOT/1k/jit_prt_t0p01_h3" --expected-count 1000
```

## JiT tau=0.02

```bash
cd "$REPAIR"
export PIXEL_REMAINDER_PYTHON=/root/miniconda3/envs/jit/bin/python
"$LAUNCH" --model JiT \
  --config "$CONFIG_SOURCE/JiT/methods/pixel-remainder-taylor/configs/jit_b16_256_prt_t0p02_h3.yaml" \
  --manifest "$REPAIR/results/JiT/baselines/taylorseer/protocol/manifest_1k.jsonl" \
  --output-root "$RUN_ROOT/1k/jit_prt_t0p02_h3" --expected-count 1000
```

## JiT tau=0.04

```bash
cd "$REPAIR"
export PIXEL_REMAINDER_PYTHON=/root/miniconda3/envs/jit/bin/python
"$LAUNCH" --model JiT \
  --config "$CONFIG_SOURCE/JiT/methods/pixel-remainder-taylor/configs/jit_b16_256_prt_t0p04_h3.yaml" \
  --manifest "$REPAIR/results/JiT/baselines/taylorseer/protocol/manifest_1k.jsonl" \
  --output-root "$RUN_ROOT/1k/jit_prt_t0p04_h3" --expected-count 1000
```

## PixelGen tau=0.01

```bash
cd "$REPAIR"
export PIXEL_REMAINDER_PYTHON=/root/miniconda3/envs/pixelgen/bin/python
"$LAUNCH" --model PixelGen \
  --config "$CONFIG_SOURCE/PixelGen/methods/pixel-remainder-taylor/configs/pixelgen_xl_256_prt_t0p01_h3.yaml" \
  --manifest "$REPAIR/results/PixelGen/baselines/taylorseer/protocol/manifest_1k.jsonl" \
  --output-root "$RUN_ROOT/1k/pixelgen_prt_t0p01_h3" --expected-count 1000
```

## PixelGen tau=0.02

```bash
cd "$REPAIR"
export PIXEL_REMAINDER_PYTHON=/root/miniconda3/envs/pixelgen/bin/python
"$LAUNCH" --model PixelGen \
  --config "$CONFIG_SOURCE/PixelGen/methods/pixel-remainder-taylor/configs/pixelgen_xl_256_prt_t0p02_h3.yaml" \
  --manifest "$REPAIR/results/PixelGen/baselines/taylorseer/protocol/manifest_1k.jsonl" \
  --output-root "$RUN_ROOT/1k/pixelgen_prt_t0p02_h3" --expected-count 1000
```

## PixelGen tau=0.04

```bash
cd "$REPAIR"
export PIXEL_REMAINDER_PYTHON=/root/miniconda3/envs/pixelgen/bin/python
"$LAUNCH" --model PixelGen \
  --config "$CONFIG_SOURCE/PixelGen/methods/pixel-remainder-taylor/configs/pixelgen_xl_256_prt_t0p04_h3.yaml" \
  --manifest "$REPAIR/results/PixelGen/baselines/taylorseer/protocol/manifest_1k.jsonl" \
  --output-root "$RUN_ROOT/1k/pixelgen_prt_t0p04_h3" --expected-count 1000
```

After a failed or safely interrupted invocation, repeat that exact command with
`--resume`. Never add `--resume` to a new output root, and never change the
authoring YAML, PixelGen parent YAML, manifest, sidecar, checkpoint, Python, or
code commit between invocations.
