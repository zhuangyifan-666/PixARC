# CODEX repair report

## Status

- 1K readiness: **FAIL** — CPU/code gates pass, but the mandatory four-GPU
  Full/fixed/dynamic smoke matrix could not run in this container.
- 50K infrastructure readiness: **PASS** — generic manifests, disjoint seeds,
  summary tracing, arbitrary expected counts, cumulative recovery timing,
  validation, aggregation, configuration materialization, and generic evaluation
  are implemented and CPU-tested. This does not waive the 1K GPU gate.
- Original repository HEAD: `444fba31d0542b7b738c06b28525e7287d8eab99`
- Repaired code commit: `4cda54517aa86e0c259d008a33eed406224e21eb`
- Repair branch: `codex/prt-ready-20260720`
- Repair worktree: `/mnt/iset/nfs-main/private/zhuangyifan/PixARC_prt_ready_20260720`

The original checkout was preserved. Before work it contained one tracked
deletion and one untracked prompt; its status/diffs were saved beside the
original repository as `pixarc_before_codex_status.txt`,
`pixarc_before_codex_worktree.patch`, and `pixarc_before_codex_index.patch`.
An older dirty `PixARC_prt_ready` worktree was also preserved and not reused.

## Frozen manifest identity

| Model | manifest SHA256 | sidecar SHA256 | records | batch/world | RNG device |
|---|---|---|---:|---|---|
| JiT | `e8ddfb2a2470661b7fbc46bd9077c2432195ae2b6986a5b466a760f68797bc1c` | `2762c6930dbf06979d2e7d9fd10fd8f749335df5aa7a92198f1afc5e84457f54` | 1000 | 32/4 | CUDA |
| PixelGen | `31536470eacf69e07ccd72305e7866957d15859b2091eec7daed2a309cedf5c0` | `bad5838c17778acd0ab32234e2c68888b1dca965b34f8590c44251964880840f` | 1000 | 4/4 | CPU |

Both tracked manifests contain LF bytes (no CRLF) and use the frozen
`manifest_1k.meta.json` sidecar. The shared resolver accepts the canonical and
frozen names, rejects missing or byte-conflicting dual sidecars, and still calls
the original strict TaylorSeer validator.

## Executable tree identity

The run-manifest hashing contract includes all `.py/.sh/.yaml/.yml` files below
`pixel_remainder_taylor/`, `scripts/`, and `configs/`, including untracked files
at execution time.

- Shared JiT/method tree SHA256:
  `e1391280c962a4ef572c31ce9e20da328b7f4f7ddc023ad08851e5b9ce281dcd`
- PixelGen adapter tree SHA256:
  `528e3b282634216c8c4d90e07ea25902a49b0ed651790e0e678ba43ac0d844bd`
- PixelGen combined canonical identity SHA256:
  `b76a4773c9be9d127eb4a28706ec1035cd4923d647fe8eff916332320c661ad6`

## Repairs completed

- Dynamic feature and pixel forecasts use the newest exact anchors with
  non-uniform Lagrange extrapolation; constant through cubic decreasing-grid
  tests and the required `[10,7,3] -> 2` quadratic counterexample pass.
- FP16/BF16 forecasts accumulate once in FP32 and cast once; duplicate,
  non-finite, insufficient, and ill-conditioned histories fail closed.
- Every Taylor NFE is preflighted before any cached model branch. An unsafe
  forecast is converted to Full without an extra network forward.
- Fixed `interval=3, order=2` parity retains the legacy recursive predictor.
- Pixel risk uses protected-minus-selected forecasts at the same target,
  per-image low/high ratios, real-batch mean decisions, diagnostic maxima,
  remaining-NFE span caps, and one compact device-to-host decision transfer.
- `sampler.t_eps` was corrected to `self.t_eps`; a real `_guided_velocity` call
  verifies epsilon clamping, `[unconditional, conditional]` ordering, and B-sized
  output.
- PixelGen `LightningCLI` now binds upstream `DataModule`; behavior tests capture
  the two received classes and round-trip the generated prediction YAML.
- Dummy-model sampler tests execute the actual JiT and PixelGen Heun paths:
  99 NFE, 198 separate JiT forwards, and 99 combined PixelGen forwards, including
  predictor/corrector/final-Euler lifecycle and exception cleanup.
- Full and summary tracing produce identical actions, outputs, and counts.
  Summary stores aggregate spans/stages/risks/timing/cache/call contracts without
  nested per-NFE JSON.
- Run identity now records config/manifest/sidecar/checkpoint/code-tree identities,
  Python/PyTorch/CUDA, predictor backend, fixed interpolation safety constant,
  model/sampling/CFG/batch/forward contracts, trace mode, and image protocol.
- Recovery uses immutable `launcher_invocations/*.json` plus cumulative
  `launcher_timing.json`; tests cover partial failure, resume, no duplicate IDs,
  immutable input rejection, and a completed no-op resume.
- The launcher validates arbitrary positive expected counts, snapshots inputs,
  requires a clean code tree and explicit GPU approval, checks UUID aliasing,
  PIDs/utilization/memory, takes output and global GPU locks, never signals
  external jobs, records interrupted invocations incomplete, and supports a
  model-family-specific absolute Python.
- Generic smoke/50K manifest builders, immutable summary configuration
  materialization, generic output validator/aggregator/evaluator, image parity
  comparator, and `RUNBOOK_1K_50K.md` were added.
- No baseline was rerun. No file below `third-party/` or `results/` was modified.

## Test evidence

### CPU and syntax

```text
JiT/methods/pixel-remainder-taylor/scripts/run_all_cpu_tests.sh
  JiT:      44 passed in 4.27s
  PixelGen: 8 passed in 1.45s
  PixelGen production manifest validator: PASS (1000 records)

PYTHONPYCACHEPREFIX=/tmp/pixarc-prt-compile-cache-release \
  /root/miniconda3/envs/jit/bin/python -m compileall -q \
  JiT/methods/pixel-remainder-taylor \
  PixelGen/methods/pixel-remainder-taylor
  result: PASS

bash -n JiT/methods/pixel-remainder-taylor/scripts/*.sh
  result: PASS

git diff --check
  result: PASS
```

`ruff`, `pyflakes`, and `shellcheck` were not installed in the available PATH or
JiT environment; no packages or global environment were modified.

### Real 50K protocol dry run

Actual external artifacts were written under
`/tmp/pixarc_prt_50k_validation` without model sampling:

- JiT 50K: 50,000 records, 50/class, 12,500/shard, batch 32, disjoint from 1K,
  manifest SHA256
  `46c7dc3274ef44a9c27b5d313732ad46c71cc9e58f99ccd5bd96a5a2a8c4ef10`.
- PixelGen 50K: 50,000 records, 50/class, 12,500/shard, batch 4, disjoint from
  1K, manifest SHA256
  `fe1a36706e7f6283873b96352f5f3498a573268547e3b89e1ae7ac3d1df3edc5`.
- Materialized JiT and PixelGen YAMLs both round-tripped with `tau=0.02`,
  `max_taylor_span=3`, `trace_mode=summary`, and absolute verified checkpoint
  paths.
- Smoke manifests were also materialized: JiT 128 records (32/shard), SHA256
  `1d4baa9e96a555f896083d039d5f7d85718829472fee839e12d3ac1f111acc6b`;
  PixelGen 16 records (4/shard), SHA256
  `988ff8dff68db358a82a828060da969d527459d8b7a3bdbf2ff155532f0cbc2b`.

## Exact GPU blocker evidence

The mandatory GPU smoke matrix was not attempted because the container exposes
no GPU device and cannot communicate with the NVIDIA driver:

```text
$ nvidia-smi -L
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
nvidia_smi_exit=9

$ ls -l /dev/nvidia*
ls: cannot access '/dev/nvidia*': No such file or directory

JiT torch 2.5.1+cu124:     cuda_available=False, device_count=0
PixelGen torch 2.7.1+cu126: cuda_available=False, device_count=0
```

The other external inputs do exist:

- JiT checkpoint: 1,576,057,392 bytes at
  `/mnt/iset/nfs-main/private/zhuangyifan/PixARC/JiT/checkpoints/JiT-B-16-256/checkpoint-last.pth`.
- PixelGen checkpoint: 2,722,303,321 bytes at
  `/mnt/iset/nfs-main/private/zhuangyifan/PixARC/PixelGen/checkpoints/PixelGen_XL_160ep.ckpt`.
- Existing Full and `i3_k2` reference trees exist for both families; each Full
  tree contains 1000 PNGs.
- ADM evaluator exists at
  `/root/third_party/guided-diffusion/evaluations/evaluator.py` and the reference
  NPZ exists at `/root/.cache/adm_eval/VIRTUAL_imagenet256_labeled.npz`.

Consequently the following mandatory items remain unverified: four-GPU
instrumented Full pixel parity, fixed-schedule pixel parity, dynamic tau 0.01
and 0.04 behavior/monotonicity, GPU resume, real GPU memory/timing, all six 1K
candidate runs, candidate evaluation, and final comparison CSVs. No result was
fabricated.

## Remaining commands

The complete copy-paste sequence is in
`JiT/methods/pixel-remainder-taylor/RUNBOOK_1K_50K.md`. Once four idle GPUs are
visible, start with:

```bash
cd /mnt/iset/nfs-main/private/zhuangyifan/PixARC_prt_ready_20260720
export REPAIR=$PWD
export CONFIG_SOURCE=/mnt/iset/nfs-main/private/zhuangyifan/PixARC
export RUN_ROOT=/mnt/iset/nfs-main/private/zhuangyifan/PixARC-runs/pixel-remainder-taylor-repair
export LAUNCH="$REPAIR/JiT/methods/pixel-remainder-taylor/scripts/launch_4gpu.sh"
export JIT_PYTHON=/root/miniconda3/envs/jit/bin/python
export PIXELGEN_PYTHON=/root/miniconda3/envs/pixelgen/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PIXEL_REMAINDER_GPU_RUN_ALLOWED=1

nvidia-smi -L
JiT/methods/pixel-remainder-taylor/scripts/run_all_cpu_tests.sh

# Then execute, in order, the RUNBOOK sections:
#   "CPU gate and smoke manifests"
#   "Required four-GPU smoke matrix"
#   pixel-identical Full/fixed checks and one --resume exercise
# Only after every gate passes: "Six new-method 1K runs"
# Then: "Generic evaluation"
```

Do not run the six 1K commands until the smoke and parity sections have passed.
