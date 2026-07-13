# PixelGen reference compatibility

## Decision

```text
PAIRED_METRICS_BLOCKED
```

The active Full reference can still be used for FID/sFID/IS/precision/recall
after its compressed 50K result is complete and validated. It cannot be used
as a strict PSNR/SSIM/LPIPS reference because the output does not establish a
complete sample-ID/class/seed/initial-noise mapping. Same array position or
preview filename is not proof of identical Gaussian noise.

The exact command/config records model, checkpoint, four ranks, batch 4,
exact-Heun 50, CFG 2.25, interval `(0.1,0.9]`, timeshift 2, and BF16, but a
continuous distributed RNG stream and compressed NPZ do not provide stable
per-sample replay evidence. No post-hoc pairing is permitted.

For paired evaluation, generate a new Full baseline and SeaCache candidate from
the same immutable manifest using independent per-sample seeds, identical fixed
batch groups, checkpoint/EMA, sampler, CFG, dtype, compile mode and
postprocessing. The formal runner uses one `combined_cfg` state and records
metadata. Batch 4/rank is the initial fixed protocol; changing it requires new
Full and threshold validation.

