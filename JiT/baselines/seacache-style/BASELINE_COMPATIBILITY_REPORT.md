# JiT Full-reference compatibility report

## Decision

```text
PAIRED_METRICS_CONDITIONALLY_COMPATIBLE
```

At audit time the JiT run was **SCHEDULED**, not completed. This classification
describes the planned/current reference protocol if that exact scheduled run
finishes successfully and its command/log/environment evidence is retained.
It is not a claim that a complete result set already exists.

## What is recoverable

- Class ID: recoverable as `sample_id//50` from the class-major generator.
- Sample ID: recoverable from numeric filename.
- Base seed and rank seed: base seed defaults to 0; rank seed is `0+rank`.
- World size and batch grouping: the retained launcher fixes four ranks and
  per-rank batch 32.
- Sampler: linear-time Heun 50 plus final Euler, 99 calls per CFG stream.
- CFG: scale 3.0, interval `[0.1,1.0]`.
- Dtype: BF16 autocast with float32 initial `torch.randn` state.
- Checkpoint/EMA: scheduled JiT-B/16 checkpoint and `model_ema1`.
- PNG conversion: the exact round/clip/RGB conversion is in
  `third-party/JiT/engine_jit.py:172-185`.

## What is not stored in the output

- Per-image seed;
- initial-noise tensor or hash;
- rank and local RNG offset;
- batch-group metadata;
- checkpoint/config hash;
- environment and PyTorch RNG version inside each image.

Same filenames alone do not establish paired initial noise.

## Conditions for using the result in paired metrics

All of the following must be satisfied:

1. Preserve the scheduled launcher, complete log, PixARC commit, PyTorch/CUDA
   versions, checkpoint identity/size, and EMA selection.
2. Reconstruct four independent CUDA RNG streams seeded `0,1,2,3`.
3. Reproduce exactly 391 full draws of shape `[32,3,256,256]` per rank,
   including the final padding positions, and map rank/batch/offset back to
   numeric sample IDs exactly as `engine_jit.py` does.
4. Feed reconstructed noise explicitly to the SeaCache run so cache decisions
   cannot alter future RNG consumption.
5. Keep world size, rank mapping, per-rank batch 32, class grouping, sampler,
   steps, CFG, interval, dtype, checkpoint/EMA, and postprocessing identical.
6. Perform a deferred GPU replay check before accepting PSNR/SSIM/LPIPS.

If any condition cannot be proven, paired metrics against this reference are
`PAIRED_METRICS_BLOCKED`.

## Distribution metrics

If the run completes and passes 50K RGB/256/uint8 validation, it remains valid
for absolute FID, sFID, IS, precision, and recall even if strict pairing cannot
be proven. Both JiT Full and SeaCache must use the same explicit local ADM
evaluator and ImageNet reference NPZ.

## Recommended paired baseline

The robust protocol is to generate a new Full result after GPU availability
using the immutable per-sample manifest in this directory, then run SeaCache
with the same manifest and explicit noise helper. A new per-sample-seed manifest
does **not** retroactively match the scheduled rank-stream reference.

