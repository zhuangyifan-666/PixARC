# JiT Full-reference compatibility

## Decision

```text
PAIRED_METRICS_CONDITIONALLY_COMPATIBLE
```

At the audit snapshot the JiT reference was queued, not running or complete.
The decision describes the queued protocol if it completes and its exact
launcher/log/environment are retained. Strict pairing remains **pending a GPU
replay proof**. If any required reconstruction evidence is missing, the
decision becomes `PAIRED_METRICS_BLOCKED`.

## Pairing evidence matrix

| Required equality | Current evidence | Status |
|---|---|---|
| class ID | class-major generator and numeric ID | recoverable |
| initial Gaussian noise | continuous per-rank RNG, no per-image hash | unproven |
| checkpoint | queued path known; preserve size/identity | conditional |
| EMA | EMA1 recorded | known |
| sampler | JiT Heun loop | known |
| steps | 50, yielding 99 NFE | known |
| CFG | 3.0, interval `[0.1,1.0]` | known |
| timeshift | not an independent JiT option in this path | known/not applicable |
| dtype | BF16 autocast; initial state float32 | known |
| batch/RNG grouping | four ranks, batch 32; padding consumes RNG | reconstructable only with exact replay |
| postprocessing | upstream engine code known | known |

Matching filenames do not prove matching noise.

## Conditions for accepting paired metrics

Preserve the exact command/log, PixARC revision, checkpoint identity, EMA,
PyTorch/CUDA versions, rank mapping, world size four, batch 32, and padding.
Reconstruct four CUDA RNG streams seeded 0,1,2,3 and their full-batch draws,
including padded positions. Feed reconstructed tensors explicitly to a local
Full replay and verify image equality before pairing them with TaylorSeer.
Sampler, compile mode, dtype, sample grouping, and postprocessing must match.

Do not force pairing by filename. If replay fails, generate a new Full run from
the immutable manifest in this port and use the same explicit per-sample noise
for the candidate.

## Distribution metrics

After the queued run completes, it may still be used for FID, sFID, IS,
precision, and recall if it passes the 50K RGB/256/uint8 protocol and uses the
same local evaluator and ImageNet reference NPZ as every candidate. This does
not make it a valid PSNR/SSIM/LPIPS reference.

## Recommended final protocol

The robust paired baseline is a new `upstream_full` manifest-backed run and a
TaylorSeer run from the exact same frozen manifest/config/checkpoint. Archive
`input_manifest.jsonl`, its sidecar/SHA-256, `config_resolved.yaml`,
`run_manifest.json`, rank metadata, and source/config hashes. Never edit these
inputs after generation begins.

