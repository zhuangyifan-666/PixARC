# PixelGen Full-reference compatibility

## Decision

```text
PAIRED_METRICS_BLOCKED
```

The active upstream reference is not a strict PSNR/SSIM/LPIPS baseline. Its
compressed output does not establish a complete mapping among stable sample
ID, class, per-image seed, initial Gaussian tensor/hash, rank RNG offset, and
batch group. Same NPZ position or preview filename is not evidence of identical
noise. A manifest-backed Full rerun is required.

## Pairing evidence matrix

| Required equality | Current evidence | Status |
|---|---|---|
| class ID | upstream dataset/config known, no complete archived per-sample mapping | blocked |
| initial Gaussian noise | distributed RNG, no per-image seed/hash | blocked |
| checkpoint | `PixelGen_XL_160ep.ckpt` recorded | known |
| EMA | prediction defaults to EMA | known |
| sampler | exact `HeunSamplerJiT` | known |
| steps | 50, yielding 99 combined forwards | known |
| CFG | 2.25 on `(0.1,0.9]` | known |
| timeshift | 2.0 | known |
| dtype | BF16-mixed | known |
| batch/RNG grouping | DDP four devices, batch 4; exact per-item consumption not archived | blocked |
| postprocessing | callback/decoder path known, identity per sample unproven | blocked |

These blocked fields cannot be repaired by indexing the NPZ after the fact.

## Distribution metrics

After completion and validation of exactly 50,000 RGB uint8 256x256 samples,
the current output may be used for FID, sFID, IS, precision, and recall. Every
method must use the same local ADM evaluator, ImageNet reference NPZ, numeric
sample ordering, preprocessing, and count. Distribution validity does not
authorize paired metrics.

## Required paired rerun

After GPUs are idle, build one immutable manifest with independent per-sample
seeds. Generate both `upstream_full` and TaylorSeer with explicit manifest
noise, the same frozen batch groups, checkpoint/EMA, exact-Heun sampler,
guidance, timeshift, dtype, compile mode, and image postprocessing. Archive
`input_manifest.jsonl`, its sidecar/SHA-256, `config_resolved.yaml`,
`run_manifest.json`, rank metadata, and source/config hashes.

The paired evaluator must fail on any missing/duplicate sample or metadata
mismatch. It must never silently fall back to the current compressed Full
reference.

